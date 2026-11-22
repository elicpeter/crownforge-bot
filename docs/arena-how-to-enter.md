# How to enter the arena tournament

Steps for a person or an agent with a fresh checkout of this repository to put
the bot on a cr-rl-arena server with the best deck, weights and play settings
found so far. What was measured for each choice is given next to it.

## What gets entered

| | |
|---|---|
| deck | `executioner,archers,dart_goblin,zap,goblins,royal_ghost,mighty_miner,elixir_golem` |
| weights | the snapshot with the highest iteration number in the GitHub release `arena-weights-2026-11-17` |
| play | sampled (the default), with look-ahead |
| look-ahead | `scope=all,cells=1,weight=40,depth=5,rollout_temperature=0.5,budget=0.6` |

Always take the newest snapshot, the `arena-itNNNNN.pt` with the highest
number, whatever it is when you read this. Do not pick one by its rating: the
newest ones have none in the release notes, because the run went on after the
last rating was measured, and a missing rating is no reason to step back to an
older file. Step 2 finds the file for you. When this was last updated
(2026-11-22) it was `arena-it24500.pt`, the run's last snapshot.

The deck must stay this one. From iteration 16,868 on the run trained on this
deck alone.

## How the tournament runs

The simulator's repository changed its server on 2026-11-22 (commit
`1d93c62`, "Add an organiser-run tournament"). The game itself did not change:
the `gym/` folder is identical to the commit the run trained on (`21615a82`),
so the weights, the deck and the build in step 1 are still right. What changed
is when games are played:

- Connecting enrols the bot. It then waits in a pool, and nothing starts by
  itself. `https://HOST/tournament` shows the pool: check that the name is
  there.
- The organiser draws a single-elimination bracket from whoever is connected
  at that moment, one game per pairing or a best of 3 or 5, and starts each
  game by hand. The wait for a game can be long. That is normal: leave the
  client running and do not restart it.
- A bot that is not connected when the bracket is drawn is not in it. Join
  early and stay.
- One lost pairing ends the tournament for the bot. A drawn game counts for
  nobody and is played again.
- A client that disconnects in a game forfeits that game. Between games a
  reconnect under the same name keeps the place in the bracket; the client
  does this by itself.
- Ten missed moves in a row void the game and disconnect the bot; the game is
  then run again once the client has rejoined.
- Games may be run at real-time speed for the audience. The server still
  waits up to its time limit for every move, so a slower game changes nothing
  for the bot.
- The pool page shows every bot's deck as soon as it joins, and tournament
  games are shown live.
- Names are not protected: any connection can join under any name. Agree the
  name with the organiser beforehand.
- Before the bracket is drawn, a person can pick a waiting bot from `/play`
  for an unrated game. The client plays it like any other.

Rehearsed on 2026-11-22 against that server commit on one machine: the client
joined, waited in the pool, and the stand-in organiser drew a best-of-3
bracket against the simulator's random player and ran it at real-time speed.
Two wins, 0 ignored moves, slowest decision 330 ms.

## Two things you must be given

They are not in this repository:

- the server address (`wss://...`)
- the name to play under (the rating and the place in the bracket are kept
  per name)

An organiser running this bot for the team has the address already and needs
only the team's name from the team. Anyone else: ask for both. Do not guess.

## 1. Build

Needs git, a Rust toolchain (https://rustup.rs) and Python 3.11 or later.

    git clone https://github.com/elicpeter/crownforge-bot && cd crownforge-bot
    python3 -m venv .venv && source .venv/bin/activate
    pip install torch
    scripts/setup_arena.sh

On Windows the last line is
`powershell -ExecutionPolicy Bypass -File scripts\setup_arena.ps1`. The script
fetches the simulator at the commit the run trained on (`21615a82`), builds the
bridge to it and ends with `arena bridge ready`. Leave it at that commit for
the tournament's `1d93c62`: the simulator code is the same in both. Only if
the tournament moves to a commit that changes `gym/`, set `ARENA_REV` to it
before running the script.

## 2. Get the weights

The repository and its release are private: `gh` must be logged in to an
account that can read them.

    SNAP=$(gh release view arena-weights-2026-11-17 --repo elicpeter/crownforge-bot --json assets \
        -q '[.assets[].name | select(test("^arena-it[0-9]+\\.pt$"))] | sort | last')
    echo "newest snapshot: $SNAP"
    mkdir -p weights/snapshots
    gh release download arena-weights-2026-11-17 --repo elicpeter/crownforge-bot \
        -p "$SNAP" -p arena-run-config.json -p SHA256SUMS -D weights/snapshots
    (cd weights/snapshots && grep -E "$SNAP|arena-run-config.json" SHA256SUMS | shasum -a 256 -c -)
    mv weights/snapshots/arena-run-config.json weights/config.json

The first command sets `SNAP` to the snapshot with the highest iteration number
(the names are padded to five digits, so they sort in order). The later steps
use `$SNAP`: run them in the same shell, or put the file name in by hand. If
you list the release yourself instead, the rule is the same: the highest
number.

Both files must say `OK` (on Linux without `shasum`, use `sha256sum -c -`).
The layout matters: a snapshot is a bare state dict, and the client reads the
model's shape from a file named `config.json` one folder above the folder the
snapshot is in.

    weights/config.json
    weights/snapshots/arena-itNNNNN.pt

## 3. Check it locally

A server on this machine and the simulator's own random player as the opponent:

    pip install ./arena/cr-rl-arena/server
    CR_ARENA_BIND=127.0.0.1:8765 cr-arena &
    python arena/cr-rl-arena/server/examples/random_agent.py ws://127.0.0.1:8765 example &
    python -m cfbot.arena_client ws://127.0.0.1:8765 check \
        --checkpoint weights/snapshots/$SNAP \
        --deck executioner,archers,dart_goblin,zap,goblins,royal_ghost,mighty_miner,elixir_golem \
        --search scope=all,cells=1,weight=40,depth=5,rollout_temperature=0.5,budget=0.6 \
        --matches 3

The server installed here is the one of commit `21615a82`, which pairs waiting
agents by itself. A server of commit `1d93c62` or later does that only when
started with `CR_AUTO_MATCH=1` in front of `cr-arena`; without it the client
prints `joined` and then waits, which is what it should do.

Stop the two background processes afterwards. Each match prints one line. What
it should look like (measured on 2026-11-20 with the snapshot of iteration
17700, on a laptop CPU):

    win [3, 0] rating 1216 (+16) match 1: 581 ticks, 6 plays, 0 ignored, 147.1 ms per decision, slowest so far 227 ms, look-ahead cut short 0 times

Go on only if all of these hold:

- three wins against the random player
- `0 ignored`
- `slowest so far` well under 1,000 ms
- no Python traceback above the match lines

## 4. Enter

    python -m cfbot.arena_client wss://HOST NAME \
        --checkpoint weights/snapshots/$SNAP \
        --deck executioner,archers,dart_goblin,zap,goblins,royal_ghost,mighty_miner,elixir_golem \
        --search scope=all,cells=1,weight=40,depth=5,rollout_temperature=0.5,budget=0.6 \
        --log matches.jsonl

The client prints `joined as NAME` and then waits in the pool until the
organiser starts a game; that can take a long time and prints nothing. It
plays every game it is given, goes back to the pool, and joins again by itself
if the connection drops. Check `https://HOST/tournament` for the name. Do not
pass `--matches` in the tournament, and do not stop the client until the
tournament is over: Ctrl-C in a game forfeits it, and a bot that is not
connected cannot be given its next game.

## Every setting, and which to leave alone

The command in step 4 is the whole entry. Nothing else needs setting. This
section says what each part does, so that you change something only for a
reason given here.

### The client's options

| option | use | why |
|---|---|---|
| `--checkpoint` | `weights/snapshots/$SNAP` | the newest snapshot, from step 2 |
| `--deck` | the eight cards above, in any order | the only deck the last 7,500 iterations trained on; play with other decks has not been measured since |
| `--search` | the look-ahead line above | worth about +0.1 of score against the bot's own copy, see below |
| `--device` | leave out (`cpu`) | fast enough, and what was tested; `cuda` only on a GPU nothing else uses |
| `--mode` | leave out (`sample`) | `greedy` scored the same in 1,024 games (0.497 against 0.499), and a bot that always plays the same move is easier to read |
| `--temperature` | leave out (1.0) | 0.8, 0.6 and 0.4 scored 0.497, 0.509 and 0.534 against 0.499 over 1,024 games each, without look-ahead: no gain shown; not measured together with look-ahead |
| `--ema` | leave out | does nothing with a release snapshot, which holds one set of weights |
| `--seed` | leave out | a new one each start |
| `--log FILE` | set it | one JSON line per match; the only record of how it went |
| `--matches N` | leave out, except for the local check | without it the client plays until stopped |

The client uses at most four CPU threads, by itself. Do not set thread
counts.

### The look-ahead's settings

Before each move the bot plays every candidate move forward in the simulator,
with both players going on to play from the policy, and adds what its value
head thinks of the positions reached to the policy's own preference.

| setting | use | what it is | what was measured |
|---|---|---|---|
| `scope=all` | keep | waiting, every playable card and the ability are compared | `scope=tile` (only where to put the card the policy chose) gained 0.02; `all` gained 0.09 on the same snapshot |
| `cells=1` | keep | per card, only its most likely tile | more tiles per card take longer and have not been measured with `scope=all` |
| `weight=40` | keep | how much the look-ahead counts against the policy's preference | 10: +0.04, 20: +0.06, 40: +0.11, 80: +0.02 to +0.07 |
| `depth=5` | keep; `3` if the machine is slow | decisions looked ahead (5 is 1.5 s of game time) | 3: +0.07 to +0.09, 5: +0.11, 7: no more than 5, at 40% more time per move |
| `rollout_temperature=0.5` | keep | how much chance there is in the imagined play | 0.5 a little better than 1.0 and 0.25 over 1,536 games, inside the error |
| `budget=0.6` | keep; lower if moves are late | seconds after which a move's look-ahead stops and is valued where it is | no effect on strength while moves take less; see the next section |
| `samples` | leave out (1) | lines played per candidate | `samples=2` gained no more than `weight=40` and takes three times as long |
| `horizon` | leave out | not used when `depth` is set | |

The gains are score against the same bot without look-ahead (0.5 is level,
512 to 1,536 games each, error ±0.025 to ±0.04), on the snapshots of
iterations 17050 and before. The full tables, and how the deck was chosen,
are in this repository's history: `docs/arena-tournament-entry.md` at commit
`5940064`.

Against opponents that are not its own copy the same setting gained between
+0.15 (the policy trained to beat it on the same deck) and nothing (older
snapshots it already beats nine times in ten). It lost nothing that the
matches could show.

## Fit the look-ahead to the machine

The setting is as strong on a slow machine as on a fast one as long as the
move arrives in time. A late move is not played at all, and ten late moves in
a row end the match. So the one thing to fit to the machine is time.

1. Find the server's time per move. The simulator's default is 2.0 s
   (`CR_MOVE_TIMEOUT`); ask whether the tournament changed it.
2. Run step 3 on the machine that will play, with nothing else heavy running.
   Read `ms per decision` and `slowest so far` from the last match line.
3. Pick the first row that fits:

| slowest decision in step 3 | use |
|---|---|
| under a third of the server's time (under 650 ms at 2.0 s) | the setting as it is |
| up to half of it | `budget=0.4` in place of `budget=0.6` |
| more than half, or `ignored` above 0 | `depth=3,budget=0.4` |
| still `ignored` above 0 | no `--search` at all (about 25 ms a move) |

The rows are a rule of thumb, not a measurement. Measured, CPU, four threads,
on an M2 Max: the setting took 145 to 250 ms a decision on average, and the
slowest decision of a check was between 230 and 620 ms. With other work
loading every core of the same machine one decision took 1,444 ms and the
look-ahead was cut short 16 times in three matches; two minutes later, with
less load, the slowest was 316 ms. `depth=3` took 151 ms on average where
`depth=5` took 247 ms.

The time cap is a target and not a promise: the first step of every line is
always played, so a decision can run past `budget`. Leave the distance to the
server's limit that the table leaves.

If the server gives less than 1.0 s per move, start from `depth=3,budget=0.3`
and check `ignored` in step 3 against a local server started with the same
limit (`CR_MOVE_TIMEOUT=1.0 CR_ARENA_BIND=127.0.0.1:8765 cr-arena`). This
case has not been run.

## Keep it running

- Run the client alone on its machine: no training, no builds, no video call.
- Plug the machine in and keep it awake for as long as the tournament runs.
  On macOS start the client under `caffeinate -i` (put it in front of
  `python`); a closed lid still puts a laptop to sleep. On Windows set sleep
  to "never" while plugged in.
- Use a wired or steady connection. A connection that drops in the middle of a
  match forfeits that match; the client then joins again by itself.
- Start it in a terminal that stays open (`tmux`, `screen`, or a window nobody
  closes). Stop it with Ctrl-C only between matches if you can: stopping in a
  match forfeits it.
- Use one client per name. Two clients under one name have not been tried.
- Do not restart it to change settings while it is winning. Every restart
  costs the match in progress.

## While it plays

Each match prints one line, as in step 3. Read these:

- `ignored` above 0: moves arrived too late or were refused. One or two in a
  long session are no worry. If the number grows from match to match, stop
  between matches and go one row down in the table above.
- `look-ahead cut short N times`: the time cap at work. A few per match are
  harmless. Dozens per match mean the machine is too slow or busy: free it,
  or go one row down.
- `slowest so far` near the server's limit: the same.
- A Python traceback: the client has stopped. Start it again with the same
  command; the rating is kept by the server under the name.
- Losses alone are no reason to change anything. The settings were chosen over
  thousands of games; a handful of tournament matches cannot show a better
  one.

Do not take `--search` away to save time unless moves are being missed even
at `depth=3`: against a copy of itself without look-ahead, the bot scored
0.763 with this setting and 0.648 without it (an earlier snapshot, 1,536
games).

## Not verified

- No match has been played against another team's bot or on the tournament's
  own server.
- The tournament's address and its move timeout are not known here. That it
  runs commit `1d93c62` is assumed from the simulator repository's newest
  commit on 2026-11-22, not confirmed by the organiser.
- A reconnect in the middle of a bracket and a walkover have not been
  rehearsed.
- Steps 1 to 3 were run as written on 2026-11-20 in a fresh clone on macOS
  (Python 3.13, CPU): the build ended with `arena bridge ready`, both
  checksums said `OK`, and the three matches were three wins with 0 ignored
  and a slowest decision of 251 ms. Steps 2 and 3 were run again after the
  release got `arena-it18800.pt`: step 2 found that file, and the result was
  three wins, 0 ignored, slowest decision 336 ms. The Windows script was not
  run again, and step 2 is written for a Unix shell.
- Steps 2 and 3 were run again on 2026-11-22 with `arena-it22400.pt`,
  `arena-it24350.pt` and `arena-it24500.pt`: three wins each, 0 ignored,
  slowest decision 316, 314 and 275 ms. A first try with `arena-it22400.pt` on a machine busy with other
  work gave a slowest decision of 1,444 ms.
- `arena-it24500.pt` has only been checked to load and to beat the random
  player. Matches that rate the snapshots after `arena-it17700.pt` were being
  played when this was written.
- A server limit other than 2.0 s per move, `--device cuda` for the client,
  and two clients under one name have not been tried.
- The time per move was measured on a laptop, not on the machine that will
  play.
- Step 4 has only been run against a server on the same machine.
