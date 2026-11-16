"""Which deck should a checkpoint take to the arena tournament?

    python scripts/arena_pick_deck.py runs/main/checkpoint.pt
    python scripts/arena_pick_deck.py runs/main/checkpoint.pt --pairs 128 --decks my_decks.json

Every candidate deck is played by the checkpoint against the same list of
opposing decks (the archetypes and as many random playable decks), each pair
twice with the seats swapped, against two opponents:

* ``self``: the same checkpoint holding the opposing deck;
* ``rush``: the arena's example client, on the deck that client ships with.

Candidates are the archetype decks, the decks of ``--decks`` (a JSON list of
eight-name lists) and the run's most played self-play decks with a good score
(``deck_stats.json`` next to the checkpoint). The table is sorted by the score
against ``self``; ``+/-`` is a 95% interval over deck pairs. Scores within each
other's intervals are not told apart: raise ``--pairs`` for the leaders.

Results are also written next to the checkpoint as ``deck_pick.json``.
"""

import argparse
import json
import os
import random

from cfbot.config import pick_device
from cfbot.decks import ARENA_ARCHETYPES
from cfbot.env import make_env
from cfbot.evaluate import play_paired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--pairs", type=int, default=64, help="opposing decks per candidate")
    parser.add_argument("--decks", default=None, help="JSON file with more candidate decks")
    parser.add_argument("--from-stats", type=int, default=8, help="self-play decks to add from deck_stats.json")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--skip-rush", action="store_true")
    args = parser.parse_args()
    device = pick_device("auto")
    simulator = {"simulator": "arena", "crforge_install": None, "crforge_tables": None}
    run_dir = os.path.dirname(os.path.abspath(args.checkpoint))

    candidates = {name: deck for name, deck in ARENA_ARCHETYPES.items()}
    if args.decks:
        for index, deck in enumerate(json.load(open(args.decks))):
            candidates[f"file{index}"] = deck
    stats_path = os.path.join(run_dir, "deck_stats.json")
    if args.from_stats and os.path.exists(stats_path):
        seen = {tuple(sorted(deck)) for deck in candidates.values()}
        played = [entry for entry in json.load(open(stats_path)).get("decks", []) if entry["games"] >= 100]
        played.sort(key=lambda entry: -entry["score"])
        for entry in played:
            if tuple(sorted(entry["deck"])) not in seen and sum(name.startswith("stats") for name in candidates) < args.from_stats:
                candidates[f"stats{entry['score']:.2f}"] = entry["deck"]
                seen.add(tuple(sorted(entry["deck"])))

    probe = make_env(1, **simulator, workers=1)
    names = probe.spec["card_names"]
    opposing = list(ARENA_ARCHETYPES.values())
    opposing += [[names[token] for token in deck] for deck in probe.random_decks(len(opposing), args.seed)]
    rng = random.Random(args.seed)
    opposing = [rng.choice(opposing) for _ in range(args.pairs)]
    guide = ARENA_ARCHETYPES["guide_example"]

    out = {}
    out_path = os.path.join(run_dir, "deck_pick.json")
    for name, deck in candidates.items():
        row = {"deck": deck}
        result = play_paired(args.checkpoint, args.checkpoint, args.pairs, args.seed, "sample", device,
                             ema=args.ema, simulator=simulator, matchups=[(deck, other) for other in opposing])
        row["self"] = {key: result[key] for key in ("score", "ci95", "games", "crashes")}
        if not args.skip_rush:
            result = play_paired(args.checkpoint, "rush", args.pairs, args.seed, "sample", device,
                                 ema=args.ema, simulator=simulator, matchups=[(deck, guide)] * args.pairs)
            row["rush"] = {key: result[key] for key in ("score", "ci95", "games", "crashes")}
        out[name] = row
        rush = f"  rush {row['rush']['score']:.3f} +/- {row['rush']['ci95']:.3f}" if "rush" in row else ""
        print(f"{name:18s} self {row['self']['score']:.3f} +/- {row['self']['ci95']:.3f}{rush}", flush=True)
        with open(out_path, "w") as handle:
            json.dump(out, handle, indent=1)

    print("\nby score against self:")
    for name, row in sorted(out.items(), key=lambda item: -item[1]["self"]["score"]):
        rush = f"  rush {row['rush']['score']:.3f}" if "rush" in row else ""
        print(f"  {row['self']['score']:.3f} +/- {row['self']['ci95']:.3f}{rush}  {name}: {','.join(row['deck'])}")
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
