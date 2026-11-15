# crownforge-bot

A neural-network agent for the [Crownforge](https://github.com/elicpeter/crownforge)
battle simulator, with the self-play training system that produces it.

The agent plays any deck against any deck. It sees only what a player sees and
outputs one simulator action per decision.

## Status

The whole pipeline runs end to end and is covered by tests: environment bridge,
network, PPO, opponent league, paired evaluation, the training command with
resume, and seeded play. No strong model has been trained yet; that is the long
run described under [Training](#training). The NVIDIA path has not been run.

This is the second design. The first used a larger recurrent transformer and
learned about 350 decisions per second on an M2 Max. This one learned 1,600 to
2,000 in a three-iteration check, and each decision covers nearly twice the
game time. [Design](#design) explains what changed and why.

## Setup

Requires Rust, Python 3.11 or later and PyTorch. The simulator is found through
a symlink named `sim`, which is not checked in:

```sh
ln -s /path/to/crownforge sim        # a crownforge checkout
python3 -m venv .venv --system-site-packages && source .venv/bin/activate
pip install maturin pytest
maturin develop --release            # builds the Rust bridge into the venv
python -m pytest -q
```

Developed against crownforge `9d5bff4e`. Rebuild with `maturin develop --release`
after the simulator changes, then run the tests: they compare the bot's public
planes against the simulator's hidden state.

## Training

```sh
python -m cfbot.train --run runs/main --set budget_hours=330   # start a two-week run
python -m cfbot.train --run runs/main --resume                 # continue after a stop
scripts/run_forever.sh runs/main --set budget_hours=330        # restart after crashes
```

Ctrl-C checkpoints and exits. A checkpoint holds the model, its averaged
weights, the optimizer, the league and the deck statistics, and a run directory
can be moved between machines: a run started on Apple Silicon continues on an
NVIDIA GPU with `--resume`. The model size is fixed when a run starts
(`--preset small|base|large`); every other setting can change on resume with
`--set key=value`. [config.py](python/cfbot/config.py) lists them all.

Every schedule (learning rate, entropy, reward shaping, deck mix, scripted-bot
share) follows the run's progress from 0 to 1. With `budget_hours`, progress is
training time spent over the budget, and the run stops when the budget is
spent. Without it, progress is `iteration / total_iters`.

Each line of `runs/<name>/metrics.jsonl` is one iteration. The numbers to
watch:

| Field | Meaning |
|---|---|
| `gauntlet_elo`, `g_*` | Rating and scores from the gauntlet (below). The strength measurement. |
| `lineage_elo` | Rating against frozen copies of the agent's own past. It shows the agent still beats its history; it cannot tell progress from a cycle. |
| `win_random`, `win_spend8.0`, `win_spend5.0` | Score against the scripted bots in training games. All three place random cards, so these saturate early. |
| `explained_variance` | How well the critic predicts outcomes; rises toward 1. |
| `entropy`, `kl`, `kl_stop` | Policy randomness and step size on decisions that had a choice. `kl_stop` is 1 when the policy stopped updating early in an iteration. |
| `play_rate`, `choice_frac` | Share of the learner's real decisions that were a play, and share of its rows that had a choice at all. |
| `trade`, `leak` | Elixir value of damage dealt per decision, and elixir wasted at the cap. |
| `return_clamped` | Share of value targets outside the value range. Should stay near 0. |
| `rejected`, `crashes`, `overflow`, `hazard_overflow`, `deck_retries` | Should stay 0. |
| `alarms` | Present when a check failed: non-finite gradients, a policy that stopped playing, collapsed entropy, halved throughput, simulator crashes. Also printed as `ALARM`. |

### Measuring strength

Every 2% of the run the agent plays a gauntlet: the three scripted bots and
frozen snapshots of itself taken at doubling iterations, always on the same 32
deck pairs and seeds, each pair played from both seats. Results go to
`gauntlet.jsonl`, and one Bradley-Terry rating scale is fitted over all of them
with the random bot at 0.

```sh
python -m cfbot.evaluate --ratings runs/main                       # rating of every gauntlet so far
python -m cfbot.evaluate runs/main/checkpoint.pt random --pairs 200
python -m cfbot.evaluate A.pt B.pt --ema                           # averaged weights
```

A direct comparison plays each deck pair twice with seats swapped and the same
seed, counts exactly one game per battle slot, and reports simulator crashes
separately instead of scoring them.

To probe how exploitable a checkpoint is, train a fresh agent against it alone
and watch `win_fixed`:

```sh
python -m cfbot.train --run runs/probe --set fixed_opponent=runs/main/checkpoint.pt
```

### Throughput and memory

On an M2 Max (32 GB) a three-iteration `base` run (3.9M parameters) learned
1,600 to 2,000 decisions per second, counting collection and the update. That
is early play on small boards; later iterations are slower. The network, not
the simulator, sets the pace.

Two things to know on Apple GPUs. Kernels are compiled per tensor shape, so the
code pads to a few fixed sizes. And `minibatch` must fit in memory: going over
physical memory makes training hundreds of times slower instead of failing. The
base model keeps about 4 MB of activations per decision at full entity width,
so 2,048 is the limit on a 16 GB GPU. On an NVIDIA GPU mixed precision is used
automatically and peak memory is logged as `gpu_mem_gb`.

## Using a trained agent

```python
from cfbot.agent import Agent
from cfbot.env import Env

env = Env(num_envs=1)
agent = Agent.load("runs/main/checkpoint.pt", env.spec, seed=0)
obs = env.reset(seed=7)
while True:
    actions = agent.act(obs)          # one flat action index per player row
    obs, info = env.step(actions)
    if info["done"][0]:
        break
```

`Agent` reads only public planes, keeps no hidden memory and is reproducible:
the same checkpoint, seed and observations give the same actions on the same
machine. It samples from the policy with a seeded generator by default.
`mode="greedy"` always takes the most likely action, but "wait" is usually the
most likely action even just before a play, so greedy play can stall.

For a program outside Python, export the policy:

```sh
pip install onnx onnxscript onnxruntime
python -m cfbot.export runs/main/checkpoint.pt policy.onnx
```

The graph maps the public observation planes to the log-probability of all
4,611 simulator actions (`-inf` where illegal) and a value estimate. Action
indices are the ones `crownforge_train::grid::decode_observed` accepts.

The public planes are built by the bridge, which counts the opponent's plays
and elixir from what both players can see. A program that controls one seat
only has to do the same counting to feed the exported graph.

## Design

The design borrows from open-source Rocket League bots (RLGym, Necto, Nexto,
Lucy-SKG), which are also trained by self-play on a headless simulator with
modest hardware. Their consistent lessons: a small, fast network trained on far
more games beats a large one; reward design and auxiliary prediction tasks buy
more than architecture; actor and critic are separate networks; batches are
large.

Clash Royale differs in ways that matter, so not everything carries over:

| Rocket League | Clash Royale | Consequence here |
|---|---|---|
| Fully observable | Opponent hand, deck and elixir are hidden | Critic sees the hidden state in training; the policy gets counted elixir, the revealed cards and where each sits in the card cycle; an auxiliary head predicts the opponent's hand |
| Fixed cars and ball | 114 cards, eight per deck | Card and unit embeddings built from learned vectors plus static stats; random, mutated and pooled decks |
| Continuous control at 15 Hz | Few, discrete, costly decisions | Steps skip ahead while nobody can act; decisions are discounted by game time |
| 90 discrete actions | Card choice times 576 tiles | Two-step action: pick a card, then a tile from a board image |
| Kinematic rewards | Economy and tower damage | Shaped reward in elixir terms, see below |
| Replay start states | Every match passes through all phases | No start-state injection; not needed |

### Observation

The Rust bridge ([src/](src)) steps many battles in parallel and writes arrays
for both players of each battle. The shared planes follow the simulator's
feature schema version 2 (same planes, columns, tokens and units):

- `global`: elapsed seconds, own elixir, ready ability buttons;
- `entities`: one row per visible unit or tower with kind, side, unit, level,
  position, health, shield and status flags, in a canonical sorted order;
- `hand`, `own_deck`;
- `recent_plays`: the last 32 card plays by either side, from the simulator's
  public play journal. An opponent's spell or tunnelling troop shows the card
  but not where it was aimed;
- `revealed_deck`: opponent cards seen so far, with last-seen time and count.

The bot adds what a memory-less policy would otherwise have to remember or
could not see:

- `entity_extras`: deploy timer, stun timer and velocity of each unit;
- `abilities`: which unit each ability button belongs to, its state and charges;
- `deck_extras`, `reveal_extras`: card cycle position. The cycle is a queue of
  four, so a card played fewer than four plays ago is still waiting and every
  other card is in the hand or arriving. This holds for the opponent too;
- `hazards`: spell areas, bombs, tunnel trails;
- `extras`: the opponent's elixir counted from regeneration, card plays,
  abilities and death rewards; time within the second; own hand refill timer;
- the legal action mask.

The opponent's actual hand, deck and elixir are exported separately (`priv_*`)
for the critic. Tests check that the policy ignores them, that the public
planes are identical when only the opponent's hidden deck differs, and that
the cycle and elixir counts agree with the hidden truth.

### Network

[model.py](python/cfbot/model.py). Actor:

1. Each unit becomes a token; one self-attention layer lets units see each
   other.
2. Units and hazards are scattered onto a 16 x 9 board image, with a count of
   units per cell and side, refined by depthwise-convolution blocks.
3. Seven query tokens (one global, four hand cards, two ability buttons) read
   the units, hazards, both decks and the play history through two
   cross-attention layers. This is the Perceiver layout Necto and Lucy-SKG
   use, at a fraction of the cost of self-attention over every token.
4. A pointer over [wait, hand cards, abilities] picks what to do. The chosen
   card's query then modulates the board image to pick one of 576 tiles.
   Exact legality masks from the simulator apply to both steps.

There is no recurrent memory. What a player has to remember is in the input.

The critic is a separate, smaller network with its own embeddings. It also
reads the opponent's hidden state, so it exists only in training. The actor
keeps a small public value head and two auxiliary heads that predict the
opponent's hand and elixir. Those heads send only a tenth of their gradient
back into the actor's encoder, so they inform it without outweighing the
policy loss there.

### Training

[train.py](python/cfbot/train.py) runs PPO over a league
([league.py](python/cfbot/league.py)):

- **Opponents.** 60% of battles are self-play, where both seats learn (about
  three quarters of the training rows). The rest are against frozen past
  snapshots and scripted bots, whose share shrinks over time. One snapshot
  slot always holds a recent snapshot; the others hold different snapshots,
  drawn mostly by how often the agent still loses to them. Old snapshots are
  thinned so the kept ones stay spread over the run.
- **Decks.** Each side draws a random playable deck, a pool deck, or a pool
  deck with one to three cards replaced. The pool starts as twenty familiar
  archetypes ([decks.py](python/cfbot/decks.py)). Nothing shows these are strong
  in this simulator, so random decks stay the largest source: 70% at the start
  and 45% from 15% of the run on. Self-play results per card and per deck are
  written to `deck_stats.json`.
- **Reward.** +1 for a win and -1 for a loss, plus three shaped terms that
  are zero-sum between the players and anneal to zero over the first quarter
  of the run: tower health (as a potential), elixir value of damage dealt to
  enemy troops minus damage taken, and elixir leaked at the cap. Only bodies a
  card deploys itself count toward the trade term; spawned units and troops
  that destroy themselves do not.
- **Time.** A step fast-forwards while neither player can do anything but
  wait, and rewards are discounted by the game time a step covered.
- **Updates.** One pass over each rollout in minibatches of 2,048 decisions.
  The actor trains on decisions that had a choice; rows where only waiting was
  legal carry no policy gradient and go to the critic alone. An averaged copy
  of the actor's weights is kept for evaluation and export (`--ema`).
- **Mirroring.** Off by default, because the simulator is not exactly
  left-right symmetric. `flip_augment` shows the acting policy a mirrored view
  for half its decisions and mirrors the chosen tile back, so every stored
  transition is one that really happened.

### Decks and levels in the bridge

`Env.queue_match` sets an exact deck pair and seed for a battle's next episode,
`Env.set_pool` replaces the pool with optional weights, and `Env.random_decks`
draws playable decks. Evaluation is built on these.

Training uses the bundled level-11 rules. `rules="complete"` loads the
simulator's bundled bank with every card level, and `level_jitter=(low, high)`
then draws a level offset per card for each battle; `rules` also accepts a path
to a rules document for another base level. A checkpoint records the rules it
was trained on, and loading it under other rules warns and continues.

## Simulator notes

- A deck with three or more ability units (champions) is refused: the
  simulator has two ability slots per side. Random decks hold at most one.
- A simulator panic ends only the affected battle: it is reported as outcome 3,
  counted in `crashes`, excluded from training and ratings, and its decks are
  listed by `Env.crashes()`.
- Seven cards the simulator marks as incomplete drafts are left out of random
  decks (`DRAFT_CARDS` in [config.py](python/cfbot/config.py)).
- Observations do not include projectiles in flight, so a spell is seen only
  through its recorded play and its effect.
- The elixir count misses ability refunds, and the tower troop is always
  Princess.

## Layout

| Path | Contents |
|---|---|
| `src/` | Rust bridge: batched battles, featurizer, cycle and elixir tracker, deck sampler |
| `python/cfbot/` | Model, PPO, league, decks, training, evaluation, agent, export |
| `tests/` | End-to-end tests on the CPU |
| `scripts/` | Environment smoke test, gradient-step profiler, restart loop |
