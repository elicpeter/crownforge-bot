# The CRForge simulator

`simulator=crforge` trains on [CRForge](https://github.com/voonhous/crforge), a
Java battle core held tick for tick to battles recorded in the game, instead
of the Rust simulator behind `sim`. The network, the collector, the league and
the evaluation are the same code; only the environment differs. A model is
tied to the simulator it was trained on, because the unit and hazard
vocabularies differ, and a checkpoint remembers which one that was.

## Setup

Once. CRForge needs Java 21, and builds its game tables from the game's own
files, which it does not ship; its `docs/game-tables.md` says where they come
from.

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

## How it is put together

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

## What the planes mean, where it differs

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

## Throughput

On an M2 Max, a one-hour run of the base preset made 7.68M learner
decisions at 2,560 a second, against 8.30M at 2,739 on the other simulator
(7% fewer; that run also saw a busier machine, load 2.7 against 2.2). The
difference is in collecting: 9.4 s an iteration against 8.1 s. The bridge call
takes about 7.7 ms per step of 256 battles against 5.5 ms, and nearly all of
it is the battle core itself.
