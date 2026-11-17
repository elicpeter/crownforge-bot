# Arena deck tournaments

Three knockout tournaments chose the decks the long arena run trains on. All
were played with `scripts/arena_deck_tournament.py`: one snapshot of the run
pilots every deck on both seats, half the field is dropped each round and the
survivors play more games. A score is a deck's share of wins in its last round,
against the other survivors. The lists the run uses now are in
[arena-decks.json](arena-decks.json).

A score says how a deck does in this pilot's hands against decks this pilot
plays. The pilot has practised some decks far more than others, and has
practised against its own focus decks most of all, so the tables rank
candidates; they are not a verdict on a deck.

## Tournament 1: pilot iteration 2600, 3,041 decks

Final table of 32 decks. Its top 8 were the focus decks from iteration 3119 to
6452, and its top 100 went into the pool.

| # | score | games | deck |
|---|---|---|---|
| 1 | 0.602 | 128 | magic_archer, zap, executioner, three_musketeers, goblin_hut, goblin_gang, mighty_miner, archers |
| 2 | 0.590 | 128 | mighty_miner, magic_archer, witch, dart_goblin, executioner, goblins, battle_ram, goblin_barrel |
| 3 | 0.562 | 128 | witch, mighty_miner, zap, musketeer, executioner, royal_ghost, battle_ram, dart_goblin |
| 4 | 0.562 | 128 | goblin_gang, royal_ghost, skeleton_army, zap, magic_archer, ronin, three_musketeers, executioner |
| 5 | 0.562 | 128 | executioner, goblin_barrel, elixir_golem, wizard, three_musketeers, goblin_gang, royal_ghost, spear_goblins |
| 6 | 0.555 | 128 | three_musketeers, executioner, zap, royal_ghost, skeleton_army, dart_goblin, goblins, goblin_hut |
| 7 | 0.555 | 128 | goblins, mighty_miner, zap, three_musketeers, witch, elite_barbarians, executioner, archers |
| 8 | 0.555 | 128 | goblin_gang, dart_goblin, minion_horde, mighty_miner, musketeer, royal_ghost, executioner, zap |

## Tournament 2: pilot iteration 6000, about 8,900 decks

Four separate fields of 2,301 decks (the 100 best of tournament 1, the 34 pool
decks, 600 random, 200 built from well-scoring cards, 1,360 mutations), then a
final of all four fields' 48 finalists: 182 decks, 256 games each. The best
deck of tournament 1 came 178th. The top 8 are the focus decks since iteration
6452 and the top 48 were added to the pool.

| # | score | games | deck |
|---|---|---|---|
| 1 | 0.648 | 256 | executioner, archers, dart_goblin, zap, goblins, royal_ghost, mighty_miner, elixir_golem |
| 2 | 0.648 | 256 | archers, dart_goblin, elixir_golem, goblins, mighty_miner, musketeer, royal_ghost, zap |
| 3 | 0.635 | 256 | goblin_gang, dart_goblin, miner, mighty_miner, musketeer, royal_ghost, executioner, zap |
| 4 | 0.625 | 256 | goblin_gang, dart_goblin, guards, mighty_miner, musketeer, royal_ghost, executioner, zap |
| 5 | 0.615 | 256 | archers, goblin_barrel, elixir_golem, executioner, mighty_miner, musketeer, royal_ghost, zap |
| 6 | 0.606 | 256 | goblin_gang, elixir_golem, musketeer, royal_ghost, bats, zap, executioner, mighty_miner |
| 7 | 0.602 | 256 | archers, dart_goblin, elixir_golem, goblin_gang, mighty_miner, musketeer, royal_ghost, zap |
| 8 | 0.596 | 256 | bandit, dart_goblin, executioner, goblin_gang, mighty_miner, royal_ghost, three_musketeers, zap |

Checked before the change, iteration 6000 piloting, 1,024 games a line:

| | new 8 | tournament 1's 8 |
|---|---|---|
| against iteration 3100 on pool decks | 0.977 ± 0.010 | 0.946 ± 0.015 |
| against itself on pool decks | 0.803 ± 0.026 | 0.722 ± 0.031 |
| new 8 against tournament 1's 8 | 0.643 ± 0.030 | |

Iteration 3100 is from before the run had focus decks, so it had trained
against neither set.

## Tournament 3: pilot iteration 6000, mutations of tournament 2's best

Four fields of 2,285 decks, 2,000 of them mutations of tournament 2's top 48,
then a final of 181 decks at 512 games each.

| # | score | games | deck |
|---|---|---|---|
| 1 | 0.600 | 514 | archers, spear_goblins, elixir_golem, goblins, mighty_miner, musketeer, royal_ghost, zap |
| 2 | 0.595 | 512 | goblin_gang, royal_ghost, goblin_barrel, executioner, goblins, elixir_golem, mighty_miner, archers |
| 3 | 0.591 | 516 | goblin_gang, elixir_golem, musketeer, royal_ghost, goblins, zap, executioner, mighty_miner |
| 4 | 0.581 | 512 | skeletons, archers, dart_goblin, zap, goblins, royal_ghost, mighty_miner, elixir_golem |
| 5 | 0.577 | 514 | executioner, archers, minion_giant, zap, goblins, royal_ghost, mighty_miner, elixir_golem |
| 6 | 0.575 | 518 | executioner, archers, dart_goblin, zap, goblins, royal_ghost, mighty_miner, elixir_golem |
| 7 | 0.575 | 512 | executioner, archers, wizard, zap, goblins, royal_ghost, mighty_miner, elixir_golem |
| 8 | 0.569 | 514 | dart_goblin, executioner, goblin_gang, mighty_miner, goblins, musketeer, royal_ghost, zap |

Its 16 best and the 8 focus decks were then each played against the same 256
pool decks (512 games a deck, iteration 6000 on both sides):

| deck | score | ± | cards |
|---|---|---|---|
| tournament 3 #6 = focus 1 | 0.851 | 0.031 | archers, dart_goblin, elixir_golem, executioner, goblins, mighty_miner, royal_ghost, zap |
| tournament 3 #8 | 0.844 | 0.032 | dart_goblin, executioner, goblin_gang, goblins, mighty_miner, musketeer, royal_ghost, zap |
| focus 6 | 0.812 | 0.034 | bats, elixir_golem, executioner, goblin_gang, mighty_miner, musketeer, royal_ghost, zap |
| tournament 3 #16 | 0.807 | 0.036 | archers, dart_goblin, executioner, goblin_barrel, goblins, mighty_miner, royal_ghost, zap |
| tournament 3 #15 | 0.807 | 0.038 | archers, dart_goblin, elixir_golem, goblin_barrel, mighty_miner, musketeer, royal_ghost, zap |
| focus 5 | 0.803 | 0.035 | archers, elixir_golem, executioner, goblin_barrel, mighty_miner, musketeer, royal_ghost, zap |
| tournament 3 #3 | 0.802 | 0.035 | elixir_golem, executioner, goblin_gang, goblins, mighty_miner, musketeer, royal_ghost, zap |
| tournament 3 #2 | 0.802 | 0.037 | archers, elixir_golem, executioner, goblin_barrel, goblin_gang, goblins, mighty_miner, royal_ghost |
| tournament 3 #4 | 0.801 | 0.036 | archers, dart_goblin, elixir_golem, goblins, mighty_miner, royal_ghost, skeletons, zap |
| tournament 3 #5 | 0.797 | 0.037 | archers, elixir_golem, executioner, goblins, mighty_miner, minion_giant, royal_ghost, zap |
| tournament 3 #7 | 0.795 | 0.036 | archers, elixir_golem, executioner, goblins, mighty_miner, royal_ghost, wizard, zap |
| tournament 3 #11 | 0.795 | 0.036 | dart_goblin, executioner, goblin_barrel, goblin_gang, mighty_miner, musketeer, royal_ghost, zap |
| focus 3 | 0.794 | 0.038 | dart_goblin, executioner, goblin_gang, mighty_miner, miner, musketeer, royal_ghost, zap |
| tournament 3 #9 | 0.793 | 0.035 | archers, battle_ram, elixir_golem, executioner, goblin_barrel, goblins, mighty_miner, royal_ghost |
| tournament 3 #13 | 0.793 | 0.036 | archers, battle_ram, dart_goblin, elixir_golem, goblins, mighty_miner, royal_ghost, zap |
| focus 7 | 0.785 | 0.037 | archers, dart_goblin, elixir_golem, goblin_gang, mighty_miner, musketeer, royal_ghost, zap |
| focus 4 | 0.785 | 0.038 | dart_goblin, executioner, goblin_gang, guards, mighty_miner, musketeer, royal_ghost, zap |
| tournament 3 #14 | 0.783 | 0.039 | dart_goblin, elixir_golem, executioner, goblin_gang, mighty_miner, musketeer, royal_ghost, zap |
| tournament 3 #12 | 0.782 | 0.038 | archers, elixir_golem, goblins, knight, mighty_miner, musketeer, royal_ghost, zap |
| focus 2 | 0.778 | 0.037 | archers, dart_goblin, elixir_golem, goblins, mighty_miner, musketeer, royal_ghost, zap |
| focus 8 | 0.777 | 0.036 | bandit, dart_goblin, executioner, goblin_gang, mighty_miner, royal_ghost, three_musketeers, zap |
| tournament 3 #10 | 0.768 | 0.041 | archers, dart_goblin, executioner, goblins, mighty_miner, royal_ghost, spear_goblins, zap |
| tournament 3 #1 | 0.751 | 0.040 | archers, elixir_golem, goblins, mighty_miner, musketeer, royal_ghost, spear_goblins, zap |

The two best lines are a focus deck (0.851) and tournament 3's eighth (0.844);
the other 21 lie between 0.751 and 0.812, inside each other's error. The focus
decks were not changed after this tournament.
