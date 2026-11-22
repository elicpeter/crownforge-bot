# The arena simulator (cr-rl-arena)

[cr-rl-arena](https://github.com/ilumn/cr-rl-arena) is the simulator of an Elo
tournament for agents: a Rust battle engine (`cr_sim`), and a server that pairs
connected agents and plays the matches itself. An agent brings its own
eight-card deck, sees its own hand and elixir and every unit and tower, and
answers each tick with a card and a tile or with nothing.

`simulator=arena` trains on that engine. [arena/](../arena/) is a second Rust
extension, `cfbot_arena`. It does not depend on the simulator as a library: it
compiles the engine's source files from a checkout pinned to one commit, so a
battle in training is the battle the server plays at that commit. Build it,
and the stock simulator next to it, with

```sh
scripts/setup_arena.sh                                               # macOS, Linux
powershell -ExecutionPolicy Bypass -File scripts\setup_arena.ps1     # Windows
```

in an environment that already has PyTorch. It needs git and a Rust toolchain
(and on Windows the Visual Studio C++ build tools). `ARENA_REV` selects the
commit; use the one the tournament server runs, since a later commit may play
differently. On a machine without a Crownforge checkout the root package
cannot be built; the script then puts `python/` on the environment's path
instead, which is all this simulator needs.

```sh
python -m cfbot.train --run runs/arena --set simulator=arena --set budget_hours=24
```

Everything else is as for the other simulators: the same planes, network,
league, gauntlet and tools. A checkpoint is tied to this simulator and to its
card list (`spec["rules_digest"]` is a digest of the card data).

## What the policy sees

Only what the server sends, which is less than the other simulators show:

- **Units and towers**: owner, kind, position, health, and whether a unit is
  still deploying. From those the bridge derives how long a unit has been out
  (time left to deploy, lifetime left for buildings and timed units) and its
  velocity over the last decision interval.
- **Not sent, so not in the planes**: spell areas, projectiles, shields, and
  every status effect (stun, slow, rage, freeze, invisibility). The hazard
  plane is always empty and the status columns of the entity plane are zero.
- **The opponent's cards** are inferred from the units that appear on its
  side ([arena/src/track.rs](../arena/src/track.rs)): units that something
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

## Actions

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

## The same rows in training and in the tournament

The bridge does not read the engine when it builds the policy's rows. It first
reduces the engine to a `View`, the contents of the server's `obs` message, and
everything else is computed from that. The tournament client builds the same
`View` from the server's JSON and runs the same code (`cfbot_arena.Featurizer`).
[tests/test_arena.py](../tests/test_arena.py) plays whole battles twice, once in
the bridge and once in the stock simulator driven the way the server drives
it, and requires every plane of every decision to be bit-identical for both
seats.

## Decks

`deck_pool=archetypes` uses this simulator's own list
(`ARENA_ARCHETYPES` in [decks.py](../python/cfbot/decks.py)): the familiar shapes
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

## Opponents

The league adds one scripted opponent on this simulator, `rush`: the arena's
example client at this bot's decision rate. Whenever it can afford a card it
plays one on a random tile of the six rows before the river. It is in the
gauntlet too, alongside six deterministic opponents with fixed decks.

| Opponent | Playstyle |
|---|---|
| `hog-cycle` | Sends Hog Rider at the bridge, cycles cheap cards, and pulls building-targeting units with Cannon. |
| `golem-beatdown` | Banks elixir for Golem and builds support behind it, with earlier commitments in double elixir. |
| `bridge-pressure` | Sends Battle Ram at the bridge and punishes heavy deployments in the opposite lane. |
| `xbow-siege` | Sets up X-Bow in a clear lane and protects it with cheap defenders and Tesla. |
| `log-bait` | Pressures with Princess and Goblin Barrel, defends with Inferno Tower, and finishes weakened towers with Rocket. |
| `graveyard-control` | Builds Goblin Hut pressure and casts Graveyard when a friendly troop tanks or the target lane lacks splash defenders. |

The policies use the visible arena state and adapt placements and defenses to
the matchup. The playstyles follow [Supercell's archetype guide](https://supercell.com/en/games/clashroyale/blog/fun/archetype-deck-challenge/)
and [classic deck guide](https://supercell.com/en/games/clashroyale/blog/fun/list-of-classic-decks-in-the-challenge/).

```sh
python -m cfbot.evaluate CHECKPOINT --suite --pairs 32 --ema
python -m cfbot.evaluate CHECKPOINT --suite --deck hog_delivery_cycle --pairs 32 --ema
python -m cfbot.evaluate CHECKPOINT xbow-siege --pairs 64 --ema
```

By default the checkpoint plays a seeded mix of archetype and random decks. Use
`--deck` with a pool deck name or eight comma-separated arena card names to
hold its tournament deck fixed. Each pair plays twice with the seats swapped.
The suite reports each matchup separately. Existing runs begin logging the new
opponents at their next arena gauntlet, when the Elo fit also gains six anchors.

## Playing in the tournament

```sh
python -m cfbot.arena_client wss://HOST NAME --checkpoint runs/arena/checkpoint.pt \
    --deck knight,archers,giant,musketeer,skeletons,fireball,arrows,zap
```

The client joins the queue, answers every observation, and plays match after
match until stopped; `--log` keeps one JSON line per match. The policy decides
every `decision_ticks` ticks, as in training, and the client passes at once on
the other ticks. A decision takes about 10 ms on a laptop CPU against the
server's limit of 2 s. Ratings are stored under the name, and the server does
not protect names: anyone can join under yours.

To try a checkpoint before entering, run the arena's own server locally and
connect the client and the arena's example client to it:

```sh
pip install ./arena/cr-rl-arena/server
CR_ARENA_BIND=127.0.0.1:8765 cr-arena &
python arena/cr-rl-arena/server/examples/random_agent.py ws://127.0.0.1:8765 example &
python -m cfbot.arena_client ws://127.0.0.1:8765 me --checkpoint ... --deck ... --matches 20
```

The settings of the tournament entry, the steps to enter with them and how the
tournament itself is run are in [arena-how-to-enter.md](arena-how-to-enter.md).

## Measured training runs

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

## Looking ahead at play time

```sh
python -m cfbot.arena_client wss://HOST NAME --checkpoint ... --deck ... \
    --search scope=tile,cells=4,horizon=40,weight=20
```

With `--search` the client tries moves before it sends one (`cfbot/search.py`,
`arena/src/search.rs`). The bridge rebuilds an engine from the latest
observation, plays each candidate in it, runs the board `horizon` ticks on with
nobody playing another card, and the policy's public value head scores the
position that results. The move is drawn from `log pi + weight * value` over
the candidates. Training does not change; this is a setting of the client.

- `scope=tile`: the policy chooses when to play and which card, the look-ahead
  chooses among that card's `cells` most likely tiles.
- `scope=all`: waiting, every playable card's best tiles and the ability are
  compared.
- `depth=N` (with either scope): both players go on playing in each try. After
  the candidate the policy chooses moves for both sides, each from what that
  side would see, for `N` decisions, and the value head scores the position
  reached; `horizon` is then not used. `rollout_temperature` below 1 makes the
  imagined players less random, `samples` averages several tries per
  candidate, and `budget=SECONDS` stops a decision's look-ahead in time for a
  server that waits only so long. The setting the tournament entry uses, what
  was measured for it and the steps to enter with it are in
  [arena-how-to-enter.md](arena-how-to-enter.md).

The engine is nearly deterministic (its random numbers go to the deck shuffle
and to scatter such as Graveyard's), which is what makes a rollout worth
something. But the server does not show everything, and a rebuilt engine
starts without it: projectiles in flight, spells still falling, lingering
spell areas, stuns and slows, attack timers and targets, damaged shields, the
opponent's hand. The opponent plays nothing in a rollout. After one decision of
both players waiting, clock, elixir, hand and cycle are exact and the unit rows
are identical in 45% of positions. Every candidate is rolled out under the same
omissions, so the comparison is fairer than any one rollout is accurate.

Without `depth`, measured on an M2 Max with the iteration-3850 snapshot of the
long run, the look-ahead side on the eight focus decks, the same snapshot sampling on pool
decks, 512 games a row, two seeds:

| first side | seed 781 | seed 912 |
|---|---|---|
| plain policy, sampled | 0.674 | 0.677 |
| `scope=tile,cells=4,horizon=40,weight=20` | 0.735 | 0.732 |
| `scope=tile,cells=8,horizon=40,weight=60` | 0.721 | |
| `scope=tile,cells=4,horizon=20,weight=20` | | 0.721 |
| `scope=tile,cells=4,horizon=80,weight=20` | | 0.740 |
| `scope=tile,cells=4,horizon=40,weight=8` | | 0.741 |
| `scope=all,cells=2,horizon=40,weight=20` | 0.718 | |
| `scope=all,cells=2,horizon=40,weight=60` | 0.688 | |

Each row is +/- 0.044. Tile look-ahead over 3,072 games scores 0.732 against
0.676 for the plain policy over 1,024, about 47 Elo; horizon and weight make no
difference that these samples show. `scope=all` gains less and loses ground
with more weight: a rollout in which the opponent never answers is a poor judge
of when to commit. The opponent in all of this is the same network, so the gain
against other bots is not measured. On the CPU (four threads) the client then takes
37 ms a decision (worst seen 84 ms) for `tile` and 100 ms (161 ms) for `all`.

Other play-time settings, same setup, seed 781: temperature 0.8, 0.6 and 0.4
scored 0.680, 0.695 and 0.645, the policy's most likely tile with a sampled
card 0.670, and `--mode greedy` 0.474. Greedy play loses about 140 Elo, because
the policy gives waiting most of the probability on any one decision; nothing
else differs from sampling at temperature 1 by more than the noise.
