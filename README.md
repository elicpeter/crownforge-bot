# crownforge-bot

A neural-network agent for the [Crownforge](https://github.com/elicpeter/crownforge)
battle simulator, with the self-play training system that produces it.

The agent plays any deck against any deck. It sees only what a player sees and
outputs one simulator action per decision.

## Status

The whole pipeline runs end to end and is covered by tests: environment bridge,
network, PPO, opponent league, evaluation, seeded play and ONNX export. No
strong model has been trained yet; that is the long run described under
[Training](#training).

This is the second design. The first used a larger recurrent transformer and
learned about 350 decisions per second on an M2 Max. This one learns about
1,300, and each decision covers nearly twice the game time.
[Design](#design) explains what changed and why.

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

Developed against crownforge `471dcd6f` (rules `october19-level11-shared-v205`).
Rebuild with `maturin develop --release` after the simulator changes. See
[Moving to the simulator's own encoder](#moving-to-the-simulators-own-encoder)
before pointing `sim` at a newer checkout.

## Training

```sh
python -m cfbot.train --run runs/main                 # start
python -m cfbot.train --run runs/main --resume        # continue after a stop
python -m cfbot.train --run runs/main --resume --set minibatch=4096
```

Ctrl-C checkpoints and exits. A checkpoint holds the model, optimizer and
league, and is portable between machines: a run started on Apple Silicon
continues on an NVIDIA GPU with `--resume`. The model size is fixed when a run
starts (`--preset small|base|large`); every other setting can change on resume
with `--set key=value`. [config.py](python/cfbot/config.py) lists them all.

Each line of `runs/<name>/metrics.jsonl` is one iteration. The numbers to
watch:

| Field | Meaning |
|---|---|
| `elo` | Rating against the agent's own past snapshots. Should keep rising. |
| `win_random`, `win_spend8.0`, `win_spend5.0` | Score against the scripted bots. |
| `explained_variance` | How well the critic predicts outcomes; rises toward 1. |
| `entropy`, `kl` | Policy randomness and step size; `kl` stays near `target_kl` or below. |
| `trade`, `leak` | Elixir value of damage dealt per decision, and elixir wasted at the cap. |
| `ticks`, `forced_wait` | Game ticks per decision, and the share of rows that could only wait. |
| `rejected`, `crashes`, `overflow` | Should stay 0. See [Simulator notes](#simulator-notes). |

Measure strength directly at any time:

```sh
python -m cfbot.evaluate runs/main/checkpoint.pt random --games 400
python -m cfbot.evaluate runs/main/checkpoint.pt runs/main/snapshots/00010.pt
```

### Throughput and memory

Measured on an M2 Max (32 GB):

| Preset | Parameters | Acting, 512 rows | Gradient step |
|---|---|---|---|
| `small` | 1.9M | about 15,000 decisions/s | about 3,400 decisions/s |
| `base` | 3.8M | about 7,700 decisions/s | about 2,100 decisions/s, 6 GB per 1,024 |
| `large` | 8.9M | about 3,700 decisions/s | not measured |

A full `base` iteration (collect, then one pass over the data) learns about
1,300 decisions per second. The simulator itself is several times faster, so
the network is the bottleneck.

Two things to know on Apple GPUs. Tensor shapes must stay constant, because
kernels are compiled per shape; the code pads to fixed sizes for that reason.
And `minibatch` must fit in memory: going over physical memory makes training
hundreds of times slower instead of failing. On an NVIDIA GPU mixed precision
is used automatically and `minibatch` can be raised; that path has not been run
yet.

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

`Agent` reads only public inputs, keeps no hidden memory and is reproducible:
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
| Fully observable | Opponent hand, deck and elixir are hidden | Critic sees the hidden state in training; the policy gets counted elixir and the revealed cards; an auxiliary head predicts the opponent's hand |
| Fixed cars and ball | 114 cards, any eight per deck | Card and unit embeddings built from learned vectors plus static stats; a deck curriculum |
| Continuous control at 15 Hz | Few, discrete, costly decisions | Steps skip ahead while nobody can act; decisions are discounted by game time |
| 90 discrete actions | Card choice times 576 tiles | Two-step action: pick a card, then a tile from a board image |
| Kinematic rewards | Economy and tower damage | Shaped reward in elixir terms, see below |
| Replay start states | Every match passes through all phases | No start-state injection; not needed |

### Observation

The Rust bridge ([src/](src)) steps many battles in parallel and writes arrays
for both players of each battle. The policy planes follow the simulator's own
feature schema version 2 exactly (same planes, columns, tokens and units):

- `global`: elapsed seconds, own elixir, ready ability buttons;
- `entities`: one row per visible unit or tower with kind, side, unit, level,
  position, health, shield and status flags, in a canonical sorted order;
- `hand`, `own_deck`;
- `recent_plays`: the last 32 card plays by either side;
- `revealed_deck`: opponent cards seen so far, with last-seen time and count.

Three planes are added by the bot: `hazards` (spell areas, bombs, tunnel
trails), `extras` (the opponent's elixir counted from regeneration and observed
spending) and the legal action mask. The opponent's actual hand, deck and
elixir are exported separately (`priv_*`) for the critic, and a test checks
that they cannot affect the policy output.

### Network

[model.py](python/cfbot/model.py). Actor:

1. Each unit becomes a token; one self-attention layer lets units see each
   other.
2. Units and hazards are scattered onto a 16 x 9 board image, refined by
   depthwise-convolution blocks.
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
opponent's hand and elixir.

### Training

[train.py](python/cfbot/train.py) runs PPO over a league
([league.py](python/cfbot/league.py)):

- **Opponents.** 60% self-play; the rest against frozen past snapshots,
  sampled so that snapshots the agent still loses to come up more often, and
  scripted bots, whose share shrinks over time.
- **Decks.** Twenty archetype decks ([decks.py](python/cfbot/decks.py)) mixed
  with random decks. Training starts at 90% archetypes and anneals to 30%,
  because uniformly random decks are mostly incoherent and teach little about
  real matchups.
- **Reward.** +1 for a win and -1 for a loss, plus three shaped terms that
  are zero-sum between the players and anneal to zero: tower health (as a
  potential), elixir value of damage dealt to enemy troops minus damage taken,
  and elixir leaked at the cap. Self-destructing troops are excluded from the
  trade term.
- **Time.** A step fast-forwards while neither player can do anything but
  wait, and rewards are discounted by the game time a step covered.
- **Updates.** One pass over each rollout in minibatches of 2,048 decisions,
  with left-right mirrored copies as augmentation.

## Moving to the simulator's own encoder

Crownforge main now ships `crownforge_train::encoding` (the schema this bot's
planes copy), public play history, deck validation, variable decision cadence
and panic quarantine. When `sim` points at such a checkout, the bridge can drop
its own copies: `featurize` in [src/features.rs](src/features.rs) calls
`encode_policy`, [src/tracker.rs](src/tracker.rs) is replaced by
`PlayerObservation::public_history`, and the panic guard in
[src/lib.rs](src/lib.rs) by `quarantine_panics`. The network and its
checkpoints do not change. The bot-owned planes (`hazards`, counted elixir)
and the reward terms stay in the bridge.

## Simulator notes

Found while building this; none are worked around silently.

- At `471dcd6f`, a deck with three or more ability units (champions) panics the
  simulator at `engine/battle.rs` (two ability slots per side). Random decks
  hold at most one champion, and `deck_pool` rejects such decks.
- A simulator panic ends only the affected battle: it is reported as outcome 3,
  counted in `crashes`, rebuilt, and its decks are listed by `Env.crashes()`.
- Seven cards the simulator marks as incomplete drafts are left out of random
  decks (`DRAFT_CARDS` in [config.py](python/cfbot/config.py)).
- Observations do not include projectiles in flight, so a spell is seen only
  through its recorded play and its effect.
- The opponent's counted elixir ignores Elixir Collector income and ability
  costs.

## Layout

| Path | Contents |
|---|---|
| `src/` | Rust bridge: batched battles, featurizer, public play tracker, deck sampler |
| `python/cfbot/` | Model, PPO, league, decks, training, evaluation, agent, export |
| `tests/` | End-to-end tests on the CPU |
| `scripts/` | Environment smoke test and gradient-step profiler |
