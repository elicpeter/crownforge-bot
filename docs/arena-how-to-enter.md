# How to enter the arena tournament

Steps for a person or an agent with a fresh checkout of this repository to put
the bot on a cr-rl-arena server with the best deck, weights and play settings
found so far. Why these were chosen, and what was measured, is in
[arena-tournament-entry.md](arena-tournament-entry.md).

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
older file. Step 2 finds the file for you. When this was written it was
`arena-it18800.pt`.

The deck must stay this one. From iteration 16,868 on the run trained on this
deck alone.

## Two things you must be given

They are not in this repository and nobody here knows them yet:

- the server address (`wss://...`)
- the name to play under (the rating is kept per name)

Ask for them. Do not guess.

## 1. Build

Needs git, a Rust toolchain (https://rustup.rs) and Python 3.11 or later.

    git clone https://github.com/elicpeter/crownforge-bot && cd crownforge-bot
    python3 -m venv .venv && source .venv/bin/activate
    pip install torch
    scripts/setup_arena.sh

On Windows the last line is
`powershell -ExecutionPolicy Bypass -File scripts\setup_arena.ps1`. The script
fetches the simulator at the commit the run trained on (`21615a82`), builds the
bridge to it and ends with `arena bridge ready`. If the tournament runs another
commit of the simulator, set `ARENA_REV` to it before running the script.

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

The client joins the queue, plays match after match and joins again if the
connection drops. Stop it with Ctrl-C. `--matches N` stops it after N matches.

## While it plays

- Run the client alone on its machine. The server waits 2 s for a move by
  default and gives up on a player after 10 missed moves in a row. With other
  heavy work on the same machine a decision took up to 1.4 s; alone it took at
  most 0.6 s.
- `--device cuda` makes the look-ahead faster where there is a GPU that
  nothing else is using. The CPU is fast enough without it.
- `ignored` above 0 means moves arrived too late or were refused. If it keeps
  growing, lower the time cap (`budget=0.4`). If that is not enough, lower
  `depth=3`. Both cost some strength; a missed move costs more.
- `look-ahead cut short N times` is the time cap at work. A few are harmless.
- A connection that drops in the middle of a match forfeits that match.
- Do not take `--search` away to save time unless moves are being missed even
  at `depth=3`: against a copy of itself without look-ahead, the bot scored
  0.763 with this setting and 0.648 without it (an earlier snapshot, 1,536
  games).

## Not verified

- No match has been played against another team's bot or on the tournament's
  own server.
- The tournament's address, its move timeout and the simulator commit it runs
  are not known here.
- Steps 1 to 3 were run as written on 2026-11-20 in a fresh clone on macOS
  (Python 3.13, CPU): the build ended with `arena bridge ready`, both
  checksums said `OK`, and the three matches were three wins with 0 ignored
  and a slowest decision of 251 ms. Steps 2 and 3 were run again after the
  release got `arena-it18800.pt`: step 2 found that file, and the result was
  three wins, 0 ignored, slowest decision 336 ms. The Windows script was not
  run again, and step 2 is written for a Unix shell.
- The newest snapshots have no rating. `arena-it18800.pt` has only been checked
  to load and to beat the random player; that it is stronger than
  `arena-it17700.pt` is expected from the run's trend, not measured.
- The time per move was measured on a laptop, not on the machine that will
  play.
- Step 4 has only been run against a server on the same machine.
