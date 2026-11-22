---
name: enter-arena-tournament
description: Put this repository's bot on a cr-rl-arena tournament server with the best deck, the newest weights and the measured look-ahead settings. Use when asked to enter, join, run, connect or restart the bot in the arena tournament, to check it before entering, or to change its play settings.
---

# Enter the arena tournament

The steps, the settings and what was measured are in one file. Read it in full
before running anything, and follow it as written:

    docs/arena-how-to-enter.md

Do not work from memory of an earlier read: the file changes when new weights
or new measurements arrive.

## Rules that the file also states, here so they are not missed

- The name to play under is `ezgg`, as the file says. The server address
  (`wss://...`) is not in the repository: ask the user for it and never guess
  it. Never connect to a tournament server under another name unless the user
  gives you one.
- Weights: always the snapshot with the highest iteration number in the GitHub
  release, whatever it is. A missing rating is no reason to take an older one.
- Deck and look-ahead settings: exactly as in the file. Change a setting only
  for a reason the file gives (moves arriving late on this machine).
- The tournament is a bracket the organiser starts by hand. After joining, the
  client waits in a pool, possibly for a long time, printing nothing. That is
  not a fault: do not restart it, and do not stop it until the tournament is
  over. Confirm on `https://HOST/tournament` that the name is in the pool.
- Run step 3 (the local check) on the machine that will play before step 4,
  and go on only if every one of its four conditions holds. Report the match
  lines to the user as printed.
- Entering is outward-facing: the bot's results count under the user's name
  from the first match. Confirm with the user before step 4 unless they have
  already told you to enter.
- While it plays, report `ignored`, `slowest so far` and `look-ahead cut short`
  from the match lines, and act on them only as the file says.
- Do not modify the simulator checkout in `arena/cr-rl-arena`.
