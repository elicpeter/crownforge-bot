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
learned about 350 decisions per second on an M2 Max. This one learns about
2,900, and each decision covers nearly twice the game time. [Design](#design)
explains what changed and why.

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
after the simulator or anything under `src/` changes, then run the tests: they
compare the bot's public planes against the simulator's hidden state.

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
`--set key=value`. [config.py](python/cfbot/config.py) lists them all. Values
are read as JSON, so switches are `true`, `false` and `null`; `True`, `False`
and `None` are understood too.

One difference between devices comes from PyTorch, not from this code. On Apple
GPUs PyTorch 2.9 also sends gradient to the padding row of an embedding table;
on NVIDIA GPUs and the CPU that row stays at zero. Some of those rows are read
by real tokens: row 0 of the unit table by an ability button that holds no
unit, row 0 of the tower table by every unit that is not a tower, row 0 of the
card table by an empty hand slot and by a card with no second form. So on a Mac
those rows are learned and elsewhere they are not. A checkpoint moved from a Mac to an NVIDIA
GPU keeps the values it has and stops updating them; a run started on the
NVIDIA GPU never has them. Nothing breaks either way, but the two are not
exactly the same model, and a short run on a Mac is not an exact preview of
the long one.

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

On an M2 Max (32 GB) the `base` preset (3.8M parameters) learns about 2,900
decisions per second, counting collection and the update: about 8 s to collect
256 battles x 128 decisions and 10 s to train on them. That was measured in
the first iterations of a run and again sixty iterations in; busier boards
later in training are slower.

The network sets the pace, not the simulator. Stepping all 256 battles takes
about 5 ms of a 60 ms collection step. The rest is four small actor passes, one
for the learner and one for each league opponent, and on an Apple GPU a pass
over 21 rows costs almost half as much as one over 250. The update is
nine-tenths forward and backward passes. The code is arranged around that:

- **Collection.** The actor runs only on learner rows that have a choice; a row
  that can only wait is settled without it. The critic does not run during
  collection: it values the stored rollout afterwards in a few large passes.
  Scripted bots, rewards and bookkeeping stay on the CPU. Per step the
  observation goes to the device as two blocks and every sampled action comes
  back in one transfer.
- **Update.** Cross-attention projects the seven queries instead of the whole
  memory. The placement head runs only on rows whose stored action was a play.
  On Apple GPUs embedding tables take their gradient through a matrix product,
  because the lookup's own gradient is very slow there.

Each of these gives the same numbers as the plain way, to rounding (to
half-precision rounding where mixed precision is on).
[tests/test_equivalence.py](tests/test_equivalence.py) holds the collector, the
update and the network to that against frozen copies of the originals, on the
CPU and on an Apple GPU where there is one.

`scripts/profile_iteration.py` prints where one iteration's time goes;
`scripts/bench_env.py` times the simulator alone.

Two things to know on Apple GPUs. Kernels are compiled per tensor shape, so the
code pads to a few fixed sizes. And `minibatch` must fit in memory: going over
physical memory makes training hundreds of times slower instead of failing. The
base model keeps about 4 MB of activations per decision at full entity width,
so 2,048 is the limit on a 16 GB GPU. Half precision is not used there: it
measured slower than float32, several times slower when acting.

On an NVIDIA GPU mixed precision is used automatically, AdamW runs as fused
kernels, the finished rollout stays on the GPU (`rollout_on_device`, about
1 GB) and peak memory is logged as `gpu_mem_gb`. `--set compile=true` also
sends the two training passes through `torch.compile`. None of this has run on
an NVIDIA GPU yet, and nothing there has been timed.

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

## The CRForge simulator

`simulator=crforge` trains on [CRForge](https://github.com/voonhous/crforge), a
Java battle core held tick for tick to battles recorded in the game, instead
of the Rust simulator behind `sim`. The network, the collector, the league and
the evaluation are the same code; only the environment differs. A model is
tied to the simulator it was trained on, because the unit and hazard
vocabularies differ, and a checkpoint remembers which one that was.

Setup, once (CRForge needs Java 21, and builds its game tables from the game's
own files, which it does not ship; its `docs/game-tables.md` says where they
come from):

```sh
git clone https://github.com/voonhous/crforge ../crforge && cd ../crforge
export JAVA_HOME=$(/usr/libexec/java_home -v 21)       # macOS; any JDK 21
./gradlew :env:installDist :tables:dataVersionTables -Pcrforge.assetSource=... -Pcrforge.dataVersion=16.635.3
cd - && pip install 'jpype1>=1.5,<2'
scripts/build_crforge.sh ../crforge/env/build/install/env   # compiles java/cfbot/bridge
export CRFORGE_INSTALL=$PWD/../crforge/env/build/install/env CRFORGE_TABLES=~/.crforge/tables/16.635.3
python -m pytest -q tests/test_crforge.py
python -m cfbot.train --run runs/crf --set simulator=crforge --set budget_hours=1
```

The two folders can also be given as `crforge_install` and `crforge_tables`;
then the run's `config.json` holds them and `--resume` and `cfbot.evaluate`
need no environment variables. `JAVA_HOME` must name the JDK in every process
that uses the simulator. Rebuild the bridge class after pulling CRForge or
changing `java/`.

How it is put together:

- `java/cfbot/bridge/CfVector.java` owns one CRForge `LadderEnv` per battle and
  steps them on a thread pool. As on the other simulator, a step gives both
  seats' actions and then keeps waiting while neither seat has an open play
  (`max_skip`), so the collector sees the same kind of step. It also writes the
  observation in this bot's layout, the rewards and the shaping terms, into
  buffers Python maps once; a step is one call into the JVM.
- `python/cfbot/crforge.py` holds the card catalogue (the same tokens as the
  other simulator; 27 cards have another row name in the game's tables), the
  deck sampler (the same pool, mutation and playability rules) and the `Env`
  subclass. At start it plays every card once in a battle of its own and
  leaves out any the battle core refuses; on data 16.635.3 all 123 play.
- The JVM runs with the throughput collector, a fixed 6 GB heap and explicit
  collections off (`JVM_ARGS` in `crforge.py`): the first steps a batch about
  a fifth faster than the default collector, the rest take another 4% off the
  bridge call. `CFBOT_JVM_ARGS` replaces the options.

What the planes mean there, where it differs:

- The action space is the same 4,611 indices. CRForge has no second play
  option, so indices from 2,307 are never legal. The legal tiles of a card are
  the tiles it lands on unmoved (CRForge's mask rule 2).
- A play runs 20 ticks after it is given. `recent_plays` lists an own play at
  the tick it was given and an enemy play at the tick it ran.
- An enemy unit's health is unknown until the game would draw its health bar;
  it reads as full until then. Tower health is always exact.
- Hazards are area effects and projectiles in flight, which is more than the
  other simulator lists (it has no projectiles). The 16 rows are given out in
  order: area effects, then projectiles a card casts (Fireball, Rocket, a
  rolling Log, a Goblin Barrel; an Arrows volley of thirty arrows is one
  hazard at the middle of those still flying), then units' shots, and last
  the crown towers' own shots and arrows drawn only for show. A tower shoots
  most of the time, so most rows carry a tower shot.
- `hazard_overflow` counts only an area or a card's projectile left out, and
  should stay 0. Shots that found no row (a Hunter fires ten pellets at once)
  are counted by `Env.shots_dropped()`: 16 in 192,000 rows of random play.
  Runs made before this ordering (`crf-ctrl-s0`, `crf-ctrl-s1`,
  `crf-base6h-s0`) gave rows out in the order CRForge lists entities and
  dropped 165,000 to 230,000 hazards each, an Arrows volley filling the plane
  by itself.
- Tower ammunition, the damage stage of a ramping attack and the enchanted
  flag are not reported (always 0). Raged and slowed come from a unit's speed
  and its listed buffs.
- The counted enemy elixir is CRForge's own count of what a player can see.
- `trade` credits damage to a unit only if a card of its owner's deck deploys
  that unit, by the unit's row: a unit spawned later that shares a row with a
  deck card's body (a Witch's skeletons next to the Skeletons card) is counted
  too. Buildings, clones, units on a timer and units that destroy themselves
  are worth nothing, as before.
- `rules` and `level_jitter` belong to the other simulator and are refused.
  Every card and tower is level 11.
- `rejected` does not stay at 0: about two plays in a million are refused
  although the mask offered them. All seen so far are a Mirror given while an
  earlier play of the same side is still waiting to run, three cases CRForge
  declines to model and answers by sending nothing (no elixir is spent).
  `Env.refusals()` lists each refused play with its reason, so anything else
  shows up there.
- A play given in a match's last second never runs, because the match is
  decided first. It still counts as accepted. `Env.run_refused()` lists these.

Throughput on an M2 Max: a one-hour run of the base preset made 7.68M learner
decisions at 2,560 a second, against 8.30M at 2,739 on the other simulator
(7% fewer; that run also saw a busier machine, load 2.7 against 2.2). The
difference is in collecting: 9.4 s an iteration against 8.1 s. The bridge call
takes about 7.7 ms per step of 256 battles against 5.5 ms, and nearly all of
it is the battle core itself.

## The arena simulator (cr-rl-arena)

[cr-rl-arena](https://github.com/ilumn/cr-rl-arena) is the simulator of an Elo
tournament for agents: a Rust battle engine (`cr_sim`), and a server that pairs
connected agents and plays the matches itself. An agent brings its own
eight-card deck, sees its own hand and elixir and every unit and tower, and
answers each tick with a card and a tile or with nothing.

`simulator=arena` trains on that engine. [arena/](arena/) is a second Rust
extension, `cfbot_arena`. It does not depend on the simulator as a library: it
compiles the engine's source files from a checkout pinned to one commit, so a
battle in training is the battle the server plays at that commit. Build it,
and the stock simulator next to it, with

    scripts/setup_arena.sh                                               # macOS, Linux
    powershell -ExecutionPolicy Bypass -File scripts\setup_arena.ps1     # Windows

in an environment that already has PyTorch. It needs git and a Rust toolchain
(and on Windows the Visual Studio C++ build tools). `ARENA_REV` selects the
commit; use the one the tournament server runs, since a later commit may play
differently. On a machine without a
Crownforge checkout the root package cannot be built; the script then puts
`python/` on the environment's path instead, which is all this simulator needs.

    python -m cfbot.train --run runs/arena --set simulator=arena --set budget_hours=24

Everything else is as for the other simulators: the same planes, network,
league, gauntlet and tools. A checkpoint is tied to this simulator and to its
card list (`spec["rules_digest"]` is a digest of the card data).

### What the policy sees

Only what the server sends, which is less than the other simulators show:

- **Units and towers**: owner, kind, position, health, and whether a unit is
  still deploying. From those the bridge derives how long a unit has been out
  (time left to deploy, lifetime left for buildings and timed units) and its
  velocity over the last decision interval.
- **Not sent, so not in the planes**: spell areas, projectiles, shields, and
  every status effect (stun, slow, rage, freeze, invisibility). The hazard
  plane is always empty and the status columns of the entity plane are zero.
- **The opponent's cards** are inferred from the units that appear on its
  side ([arena/src/track.rs](arena/src/track.rs)): units that something
  standing or dying within three tiles of them produced (a hut, a Witch, a
  death spawn, a champion's summons) are not counted as plays; a Goblin Barrel
  or Graveyard is recognised by its units. A spell that leaves no unit
  (Fireball, Zap, Poison, The Log, ...) is never seen, and a spell that drops
  units next to a spawner of the same units is taken for the spawner's work.
- **The opponent's elixir** is counted from those inferred plays, so it is an
  estimate that can be wrong either way: too high after an unseen spell or a
  Mirror, too low while an Elixir Collector pays out.
- **Own state** is exact: hand, next card, the card cycle, what Mirror would
  replay, a card's evolution charge, and the champion's ability cooldown, all
  counted from the bot's own accepted moves.

The critic's training-only planes hold the opponent's true elixir, hand and
deck, as before.

### Actions

The same 4,611 flat actions. A play is a hand slot and one of the 576 tiles in
the player's own frame; both seats see themselves at the bottom, as on the
server, so nothing is flipped anywhere. The legal mask follows the engine's own
rule with one exception: troops and buildings on the fifteen own rows, plus
rows 15 to 20 of a lane whose enemy princess tower has fallen; spells, Miner
and Goblin Drill anywhere. The exception is the two river rows of such a
pocket: the engine takes a ground troop there, off the bridge, and the troop
then never leaves the water, so those tiles are offered only on the bridges
unless the unit flies, jumps the river or is a building.
The ability action fires the champion's ability when it is on the board, off
cooldown and paid for (and, for the Golden Knight's dash, with an enemy in
reach). Goblinstein's revive is not offered. The secondary play forms do not
exist here and are never legal.

In the tests every action the mask offers is accepted by the engine.

### The same rows in training and in the tournament

The bridge does not read the engine when it builds the policy's rows. It first
reduces the engine to a `View`, the contents of the server's `obs` message, and
everything else is computed from that. The tournament client builds the same
`View` from the server's JSON and runs the same code (`cfbot_arena.Featurizer`).
[tests/test_arena.py](tests/test_arena.py) plays whole battles twice, once in
the bridge and once in the stock simulator driven the way the server drives
it, and requires every plane of every decision to be bit-identical for both
seats.

### Decks

`deck_pool=archetypes` uses this simulator's own list
(`ARENA_ARCHETYPES` in [decks.py](python/cfbot/decks.py)): the familiar shapes
under the arena's card names, plus the two decks the arena's own material hands
a newcomer (the guide's example deck and the built-in default). Random decks
follow the same playability rules as elsewhere. `exclude_cards` names that
this simulator does not have are skipped.

A tournament entry plays one deck. Two tools for that:

- `focus_decks` and `focus_prob` give one seat of a battle one of the listed
  decks with that chance, whatever the other seat draws: more practice with
  the candidate decks against the usual spread of opponents. In a league
  battle the learner sits on that seat, so the practice goes to the network
  that is learning. Both can be set when a run is resumed.
- `python scripts/arena_pick_deck.py CHECKPOINT` plays the checkpoint on every
  candidate deck against one fixed list of opposing decks, and against the
  arena's example client on its own deck, and prints the table.

### Opponents

The league adds one scripted opponent on this simulator, `rush`: the arena's
example client at this bot's decision rate. Whenever it can afford a card it
plays one on a random tile of the six rows before the river. It is in the
gauntlet too, as a fourth fixed anchor of the rating scale.

### Playing in the tournament

    python -m cfbot.arena_client wss://HOST NAME --checkpoint runs/arena/checkpoint.pt \
        --deck knight,archers,giant,musketeer,skeletons,fireball,arrows,zap

The client joins the queue, answers every observation, and plays match after
match until stopped; `--log` keeps one JSON line per match. The policy decides
every `decision_ticks` ticks, as in training, and the client passes at once on
the other ticks. A decision takes about 10 ms on a laptop CPU against the
server's limit of 2 s. Ratings are stored under the name, and the server does
not protect names: anyone can join under yours.

A first check on an M2 Max, base preset, one hour, seed 0: 7.39M learner
decisions at 2,764 a second (the gradient step is the limit; the bridge steps
256 battles in under 2 ms), gauntlet rating +55 to +435, no rejected plays and
no crashes. On the arena's own server that checkpoint won 13 of 20 matches
against the arena's example client, which had beaten an untrained network 3 to
0, with no ignored moves. The scripted `rush` bot scored lower against the same
checkpoint in the gauntlet (0.08) than the real client did (0.35): the client
plays an all-troop deck, the gauntlet's decks hold spells it wastes. Check
against the real client before entering.

On an RTX 5070 Ti with a Ryzen 7 9800X3D (Windows 11, PyTorch 2.11, CUDA
12.8) the limit is different. The gradient step is cheap there and collecting
is not: each step of the batch makes four network calls (the learner and three
league opponents), and their fixed cost outweighs the arithmetic until the
batch is large. Measured decisions a second, two to three minutes each:

| preset | battles x steps | decisions/s | GPU memory |
|---|---|---|---|
| base | 256 x 128 (default) | 6,900 | 5.5 GB |
| base | 512 x 128 | 9,880 | 6.1 GB |
| base | 1024 x 128 | 11,970 | 7.6 GB |
| large | 256 x 128 | 5,600 | 9.2 GB |
| large | 512 x 128 | 7,640 | 10.2 GB |
| large | 1024 x 64 | 8,690 | 10.7 GB |
| large | 1536 x 64 | 9,310 | 12.1 GB |
| large | 2048 x 64 | 9,450 | 13.0 GB |
| large | 1536 x 64, `minibatch=1024` | 9,330 | 7.7 GB |

So more battles and shorter rollouts buy speed, and the large preset costs a
quarter of the throughput there, against two thirds on the M2 Max. Memory in
the gradient step grows with the widest board of the minibatch, about 1.1 GB
for the large preset for every 16 more units at `minibatch=2048`; a 16 GB card
is safe at 1024. `compile=true` does nothing on Windows: PyTorch's compiler
needs Triton there, the forward passes fall back, and the run was slower.

A detached run on Windows cannot be sent Ctrl-C. A file named `STOP` in the run
folder does the same: the run checkpoints after the iteration and exits, and
removes the file.

To try a checkpoint before entering, run the arena's own server locally and
connect the client and the arena's example client to it:

    pip install ./arena/cr-rl-arena/server
    CR_ARENA_BIND=127.0.0.1:8765 cr-arena &
    python arena/cr-rl-arena/server/examples/random_agent.py ws://127.0.0.1:8765 example &
    python -m cfbot.arena_client ws://127.0.0.1:8765 me --checkpoint ... --deck ... --matches 20

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
| `src/` | Rust bridge to Crownforge: batched battles, featurizer, cycle and elixir tracker, deck sampler |
| `arena/` | Rust bridge to cr-rl-arena: the same, computed from what the tournament server sends, and the client's featurizer |
| `python/cfbot/` | Model, PPO, league, decks, training, evaluation, agent, export |
| `tests/` | End-to-end tests, and the optimized collector, update and network against frozen originals |
| `scripts/` | Environment smoke test, simulator benchmark, iteration and gradient-step profilers, restart loop |
