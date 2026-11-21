# Arena tournament entry

What the long arena run enters the cr-rl-arena tournament with: one deck, the
run's last snapshot, and a look-ahead at play time. Written on 2026-11-20
while the run was in its last hours; numbers that were still being measured
are marked as such.

Throughout, a score is the share of games won (a draw counts half) and comes
from seat-swapped pairs of games with sampled play. A match of 512 games
carries an error of about ±0.04, one of 1,024 games about ±0.03.

## The entry

| | |
|---|---|
| deck | `executioner, archers, dart_goblin, zap, goblins, royal_ghost, mighty_miner, elixir_golem` |
| weights | the last snapshot of the run on the training box (`runs/arena/snapshots/`), a bare state dict |
| play | sampled at temperature 1, with look-ahead |

```
python -m cfbot.arena_client wss://HOST NAME \
    --checkpoint runs/arena/snapshots/NNNNN.pt \
    --deck executioner,archers,dart_goblin,zap,goblins,royal_ghost,mighty_miner,elixir_golem \
    --search scope=all,cells=1,weight=40,depth=5,rollout_temperature=0.5,budget=0.6
```

The steps from an empty folder to a playing client, with the newest weights of
the GitHub release, are in [arena-how-to-enter.md](arena-how-to-enter.md).

A bare snapshot is loaded with the run's `config.json` two directories above
it. `--device cuda` makes the look-ahead faster where there is a GPU. Run the
client alone on its machine: see "Time per move" below.

The look-ahead setting above scored best over 1,536 games; two others were
within the error of it. It also gains against the newest counter policies,
which are different bots from the one it imagines, but it has not met another
team's bot (see "Not verified").

## How the deck was chosen

The run narrowed its focus decks in stages. The opposing side was never
narrowed: all eight decks of the last list stay in the pool, and the counter
policy that mirrors the focus decks still plays all eight.

**To eight** (iteration 6452): the top eight of deck tournament 2, see
[arena-deck-tournaments.md](arena-deck-tournaments.md).

**Eight to four** (iteration 11727). The iteration-10600 snapshot piloted each
deck against four opponents: itself on pool decks, and three counter policies
trained to beat the run (one on decks built to attack it, one on pool decks,
one on the focus decks). 4,096 games per deck.

| deck | average score | worst opponent |
|---|---|---|
| 1 | 0.734 | 0.507 |
| 5 | 0.730 | 0.525 |
| 2 | 0.696 | 0.476 |
| 7 | 0.688 | 0.481 |
| 6 | 0.687 | 0.463 |
| 8 | 0.668 | 0.460 |
| 3 | 0.638 | 0.397 |
| 4 | 0.608 | 0.348 |

Kept 1, 5, 2 and 7. Deck 7 over deck 6 was a tie within the error.

**Four to two** (iteration 16844). The iteration-16700 snapshot, the same
four opponents with newer counters, and itself on the four kept decks. 2,560
games per deck.

| deck | pool | counter a | counter b | counter c | kept decks | average |
|---|---|---|---|---|---|---|
| 1 | 0.847 | 0.715 | 0.826 | 0.574 | 0.545 | 0.701 |
| 5 | 0.799 | 0.746 | 0.822 | 0.547 | 0.558 | 0.694 |
| 2 | 0.764 | 0.652 | 0.793 | 0.504 | 0.492 | 0.641 |
| 7 | 0.730 | 0.625 | 0.746 | 0.424 | 0.382 | 0.581 |

**Two to one** (iteration label 16868). No single measurement separates decks
1 and 5 (0.734 against 0.730, then 0.701 against 0.694; a third, cut short,
gave 0.831 against 0.793 over three opponents). Deck 1 was ahead each time, on
three different snapshots, and by most against pool decks. The two share six
cards. Deck 1 was chosen so that the remaining hours went to one deck.

Deck numbers are positions in the list of eight:

1. executioner, archers, dart_goblin, zap, goblins, royal_ghost, mighty_miner, elixir_golem
2. archers, dart_goblin, elixir_golem, goblins, mighty_miner, musketeer, royal_ghost, zap
3. goblin_gang, dart_goblin, miner, mighty_miner, musketeer, royal_ghost, executioner, zap
4. goblin_gang, dart_goblin, guards, mighty_miner, musketeer, royal_ghost, executioner, zap
5. archers, goblin_barrel, elixir_golem, executioner, mighty_miner, musketeer, royal_ghost, zap
6. goblin_gang, elixir_golem, musketeer, royal_ghost, bats, zap, executioner, mighty_miner
7. archers, dart_goblin, elixir_golem, goblin_gang, mighty_miner, musketeer, royal_ghost, zap
8. bandit, dart_goblin, executioner, goblin_gang, mighty_miner, royal_ghost, three_musketeers, zap

`docs/arena-decks.json` still lists the eight: the Elo scale of the weights
release is measured on them.

## Strength since iteration 14450

The weights release rates snapshots up to iteration 14450 on the eight decks
(untrained network = 0). The later snapshots were rated on deck 1 alone: both
sides play deck 1, 1,024 games a match, each snapshot against the one before
it and the one two before it, eight matches, none crashed. The release's
ratings were kept fixed and the new ones fitted to these matches, so from
14450 on the scale measures play on deck 1.

| iteration | Elo |
|---|---|
| 14000 | +1495 (release) |
| 14450 | +1527 (release) |
| 15000 | +1553 |
| 16000 | +1596 |
| 17000 | +1639 |
| 17700 | +1669 |

| newer vs older | wins-draws-losses | score |
|---|---|---|
| 15000 vs 14000 | 572-0-452 | 0.559 |
| 15000 vs 14450 | 550-0-474 | 0.537 |
| 16000 vs 14450 | 638-0-386 | 0.623 |
| 16000 vs 15000 | 545-0-479 | 0.532 |
| 17000 vs 15000 | 639-1-384 | 0.625 |
| 17000 vs 16000 | 576-1-447 | 0.563 |
| 17700 vs 16000 | 612-0-412 | 0.598 |
| 17700 vs 17000 | 564-0-460 | 0.551 |

Every match is within 0.03 of what the ratings predict. Iterations 17000 and
17700 are the run's labels after a power loss on 2026-11-20: the run was
restarted from an earlier checkpoint with the weights of a later snapshot, so
those snapshots have about 106 iterations more training behind them than
their labels say.

## Look-ahead

Before it sends a move, the client tries moves out (`cfbot/search.py`,
`arena/src/search.rs`). With `depth` above zero both players go on playing in
each try: after the candidate move, the policy chooses moves for both sides,
each from what that side would see, for `depth` decisions, and the value head
scores the position reached. `scope=all` compares waiting, each hand card on
its most likely tile, and the champion's ability. The move is drawn from
`log pi + weight * value` over those candidates.

The opponent in a try holds the cards it has shown, filled up with a guess,
and is played by the same network. `rollout_temperature` below 1 makes both
imagined players less random. All candidates of one decision share their
random draws.

### Which kind of look-ahead

Iteration 14450 with look-ahead against itself without, both sides on the
four decks then in use, every deck pairing.

| look-ahead | games | score |
|---|---|---|
| none (control) | 1,024 | 0.499 |
| one own move, then nobody plays for 2 s; only the tile is chosen (`depth=0`, `scope=tile`) | 4,096 | 0.517 |
| both play on for 2.1 s; only the tile is chosen (`scope=tile,depth=7`) | 1,024 | 0.521 |
| both play on for 1.5 s; waiting and every card compared (`scope=all,cells=1,depth=5,weight=20`) | 1,024 | 0.586 |

The last row again on iteration 16700, decks 1 and 5: 0.578 over 512 games.

Other play-time settings without look-ahead made no difference on that
snapshot that 1,024 games each could show: greedy play 0.497, temperature 0.8,
0.6 and 0.4 scored 0.497, 0.509 and 0.534.

### Which setting

The snapshot labelled 17050 on deck 1, with look-ahead, against itself
without look-ahead on the eight decks. 512 games per setting, the same games
for every setting. All settings are `scope=all,cells=1`.

| setting | score | gain |
|---|---|---|
| no look-ahead | 0.637 | |
| `weight=10,depth=5` | 0.678 | +0.041 |
| `weight=20,depth=5` | 0.693 | +0.057 |
| `weight=80,depth=5,rollout_temperature=0.5` | 0.707 | +0.070 |
| `weight=20,depth=3` | 0.709 | +0.072 |
| `weight=20,depth=5,samples=2` | 0.739 | +0.103 |
| `weight=40,depth=5` | 0.758 | +0.121 |
| `weight=20,depth=7` | 0.760 | +0.123 |
| `weight=40,depth=5,rollout_temperature=0.5` | 0.761 | +0.124 |
| `weight=20,depth=5,rollout_temperature=0.5` | 0.766 | +0.129 |

Later additions on the same games: `weight=40,depth=7` 0.766,
`weight=40,depth=7,rollout_temperature=0.5` 0.767,
`weight=40,depth=5,rollout_temperature=0.25` 0.754,
`weight=20,depth=5,rollout_temperature=0.25` 0.736,
`weight=40,depth=3,rollout_temperature=0.5` 0.723 and `weight=80,depth=5`
0.660.

Several settings are level at about 0.76 and cannot be told apart at this
size, so five were played again on two more sets of games, each set with its
own baseline (0.637, 0.671 and 0.637 without look-ahead, 0.648 together).
1,536 games per setting:

| setting | the three sets | together | gain |
|---|---|---|---|
| `weight=40,depth=5,rollout_temperature=0.5` | 0.761 0.779 0.750 | 0.763 | +0.115 |
| `weight=40,depth=5` | 0.758 0.763 0.738 | 0.753 | +0.105 |
| `weight=20,depth=7` | 0.760 0.748 0.732 | 0.747 | +0.099 |
| `weight=20,depth=5,rollout_temperature=0.5` | 0.766 0.738 0.705 | 0.736 | +0.088 |
| `weight=20,depth=5` | 0.693 0.715 0.715 | 0.708 | +0.060 |

Each row is ±0.025. The first is the entry's setting: best together, and no
dearer per move than the others at `depth=5`. Looking further ahead did not
add to it (`depth=7` with `weight=40`: 0.766 against 0.758 on the same games)
and costs about 40% more time per move. More weight than 40 does worse.

### Against other opponents

The same snapshot on deck 1, without look-ahead and with the entry's setting,
against opponents that are not the bot the look-ahead imagines. Sets of 512
games, the same games in both cells of a row, none crashed. The counters are
policies trained from the run's weights to beat it: c on the eight focus
decks, a on four decks built to attack it, b on the pool decks. "Final" are
their last weights, which this snapshot had never played.

| opponent | games | without | with look-ahead | gain |
|---|---|---|---|---|
| counter c, final | 1,024 | 0.354 | 0.505 | +0.151 |
| counter c, the copy the run trained against | 512 | 0.494 | 0.664 | +0.170 |
| counter a, final | 1,024 | 0.658 | 0.716 | +0.058 |
| counter a, the copy the run trained against | 512 | 0.850 | 0.820 | -0.029 |
| counter b, final | 512 | 0.816 | 0.852 | +0.035 |
| iteration 12000 of the run | 1,024 | 0.890 | 0.896 | +0.007 |
| iteration 8000 of the run | 512 | 0.945 | 0.958 | +0.013 |

By set, for the rows of two sets: counter c final +0.154 and +0.147, counter a
final +0.034 and +0.082, iteration 12000 +0.002 and +0.012.

The look-ahead gains most where the snapshot is weakest, against the policy
specialised in the mirror, and next to nothing where the snapshot already
wins nine games in ten. In one row it scored lower, against the older counter
a; that difference and the gains against counter b and the two earlier
iterations are each inside the error of their match.

In these matches the look-ahead changed about 2% of the decisions.

### Time per move

The server gives 2.0 s per move by default (`CR_MOVE_TIMEOUT`) and
disconnects an agent that misses ten moves in a row. `budget` stops a
decision's look-ahead once another step would take it past that many seconds;
the decision then looks less far ahead instead of arriving late.

Measured on an M2 Max, CPU, four threads, `scope=all,cells=1,depth=5`:
247 ms a decision on average and 385 ms at worst over 346 decisions;
`depth=3` 151 ms and 199 ms. Without look-ahead the client takes about 25 ms.

Dress rehearsal on 2026-11-20: a local `cr_server`, this client with
look-ahead against this client without, same snapshot, deck 1.

| machine | budget | matches | ms per decision | slowest | budget cut in | moves the server ignored |
|---|---|---|---|---|---|---|
| busy with other jobs | 0.8 s, checked only after a step | 4 | 650 to 830 | 3,085 ms | 163 to 183 a match | 7 in one match |
| busy with other jobs | 0.6 s | 2 | 480 to 515 | 1,411 ms | 133 to 208 a match | 0 |
| quiet | 0.6 s | 2 | 269 to 282 | 574 ms | 0 and 1 | 0 |

All eight matches ran to a result recorded by the server. The first row is
why the budget now looks one step ahead. Eight games say nothing about which
client is stronger.

## Not verified

- **No match against another team's bot.** Every number here is the run
  against itself, earlier versions of itself, or counter policies started
  from its own weights.
- **Look-ahead against a bot of another kind.** The opponent a try imagines
  is the run's own network. The gain held against the counter policies, but
  those started from the run's weights too; a bot trained elsewhere may be
  read worse.
- **The final snapshot.** Ratings stop at the label 17700; the run went on
  after it.
- **The tournament's own settings**: its move timeout and the commit of the
  simulator it runs. The run trained on cr-rl-arena at commit `21615a82`.
- **Time per move on the machine that will play.** The figures above are
  from a Mac.
- The look-ahead guesses the opponent's unseen cards from the viewer's own
  deck, and does not see spells the opponent has played, as the tracker does
  not.
