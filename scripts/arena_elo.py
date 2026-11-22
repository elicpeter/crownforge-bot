"""Elo for any set of bots, from matches between them on given decks.

    python scripts/arena_elo.py --bot main8000=runs/arena-box/snapshots/00160.pt \\
        --bot counter=runs/counter-c/checkpoint.pt --match counter main8000

    python scripts/arena_elo.py --bot a=A.pt --bot b=B.pt --bot c=C.pt --all --anchor a

A bot is ``NAME=PATH`` (a training checkpoint, or a bare snapshot whose run's
``config.json`` sits two directories up) or the bare name of a scripted
opponent such as ``lanerush``. ``--match A B`` plays one match; ``--all`` plays
every pair; ``--ladder`` plays each bot against the one and two places before
it in the order given, which is enough when the order is oldest to newest.

Both sides play the same decks: every ordered pair of ``--decks`` (the eight
focus decks by default), each as two seat-swapped games, so neither side has
the better decks. ``--decks`` takes a JSON file holding a list of decks, or an
object with ``focus_decks``.

Every result is added to ``--out`` (a JSON file) as soon as its match ends, and
a match already in that file is not played again, so a table grows over
several calls and several days. ``--fit-only`` plays nothing and prints the
ratings of what the file holds. Bots that share no chain of matches cannot be
put on one scale; the script says so instead of printing a number.

The ratings are the maximum-likelihood fit of the usual Elo curve to all the
results at once, ``--anchor`` (default: the first bot in the file) set to 0.
Under the ratings comes each match's score next to the score the ratings
predict. Where those differ by more than the error of one match, one number
does not describe that bot: it does better or worse against a particular
opponent than its rating says.
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

DEFAULT_DECKS = "docs/arena-decks.json"


def fit(results: dict, anchor: str | None) -> tuple[dict[str, float], list[str]]:
    """Ratings of the anchor's connected group, and the bots outside it."""
    games = [(r["first"], r["second"], r["wins"] + r["draws"] / 2, r["wins"] + r["draws"] + r["losses"])
             for r in results.values()]
    names = list(dict.fromkeys(name for g in games for name in g[:2]))
    if not names:
        return {}, []
    anchor = anchor if anchor in names else names[0]
    group, grew = {anchor}, True
    while grew:
        grew = False
        for a, b, _, _ in games:
            if (a in group) != (b in group):
                group |= {a, b}
                grew = True
    games = [g for g in games if g[0] in group]
    rating = {name: 0.0 for name in names if name in group}
    for _ in range(20000):
        grad = {name: 0.0 for name in rating}
        for a, b, points, n in games:
            # A clean sweep has no finite rating: count half a game the other way.
            points = min(max(points, 0.5), n - 0.5)
            expect = n / (1 + 10 ** ((rating[b] - rating[a]) / 400))
            grad[a] += points - expect
            grad[b] -= points - expect
        for name in rating:
            rating[name] += 0.05 * grad[name]
    base = rating[anchor]
    return {name: value - base for name, value in rating.items()}, [n for n in names if n not in group]


def report(results: dict, anchor: str | None) -> None:
    rating, outside = fit(results, anchor)
    if not rating:
        print("no matches yet")
        return
    print(f"\nElo ({len(results)} matches, {sum(r['wins'] + r['draws'] + r['losses'] for r in results.values())} games)")
    for name, value in sorted(rating.items(), key=lambda item: item[1]):
        print(f"  {name:24s} {value:+7.0f}")
    for name in outside:
        print(f"  {name:24s} no chain of matches to the anchor")
    print("\nmatch, score seen, score the ratings predict")
    for r in results.values():
        if r["first"] not in rating:
            continue
        n = r["wins"] + r["draws"] + r["losses"]
        seen = (r["wins"] + r["draws"] / 2) / n
        expect = 1 / (1 + 10 ** ((rating[r["second"]] - rating[r["first"]]) / 400))
        flag = "   <-- differs by more than the match's error" if abs(seen - expect) > max(r["ci95"], 0.02) else ""
        print(f"  {r['first']} vs {r['second']}: {r['wins']}-{r['draws']}-{r['losses']}  {seen:.3f}  {expect:.3f}{flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bot", action="append", default=[], metavar="NAME=PATH",
                        help="a checkpoint or snapshot under a name, or a scripted opponent's name")
    parser.add_argument("--match", action="append", nargs=2, default=[], metavar=("FIRST", "SECOND"))
    parser.add_argument("--all", action="store_true", help="every pair of bots")
    parser.add_argument("--ladder", action="store_true", help="each bot against the one and two before it")
    parser.add_argument("--decks", default=DEFAULT_DECKS)
    parser.add_argument("--pairs", type=int, default=512, help="deck pairs a match; two games each")
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument("--mode", default="sample", choices=["sample", "greedy"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="runs/arena_elo.json")
    parser.add_argument("--anchor", default=None, help="the bot whose rating is 0")
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()

    results = {}
    if os.path.exists(args.out):
        with open(args.out) as handle:
            results = json.load(handle)
    if args.fit_only:
        report(results, args.anchor)
        return

    bots, order = {}, []
    for spec in args.bot:
        name, _, path = spec.partition("=")
        path = path or name  # a scripted opponent goes by its own name
        if "=" in spec and not os.path.exists(path):
            parser.error(f"{name}: no file at {path}")
        bots[name] = path
        order.append(name)
    matches = [tuple(m) for m in args.match]
    if args.all:
        matches += [(order[j], order[i]) for i in range(len(order)) for j in range(i + 1, len(order))]
    if args.ladder:
        matches += [(order[i], order[i - step]) for i in range(len(order)) for step in (1, 2) if i - step >= 0]
    matches = list(dict.fromkeys(matches))
    for match in matches:
        for name in match:
            if name not in bots:
                parser.error(f"--match names {name}, which no --bot defines")
    if not matches:
        parser.error("nothing to play: give --match, --all or --ladder")

    from cfbot.config import pick_device
    from cfbot.evaluate import play_paired

    with open(args.decks) as handle:
        decks = json.load(handle)
    decks = decks["focus_decks"] if isinstance(decks, dict) else decks
    square = [(a, b) for a in decks for b in decks]
    matchups = (square * (args.pairs // len(square) + 1))[: args.pairs]
    device = pick_device(args.device)
    setting = {"decks": os.path.abspath(args.decks), "pairs": args.pairs, "seed": args.seed, "mode": args.mode}

    for first, second in matches:
        key = f"{first} vs {second}"
        done = results.get(key) or results.get(f"{second} vs {first}")
        if done is not None:
            if any(done.get(k) != v for k, v in setting.items()):
                print(f"{key}: already in {args.out} under other settings; kept as it is")
            else:
                print(f"{key}: already played")
            continue
        start = time.time()
        r = play_paired(bots[first], bots[second], args.pairs, args.seed, args.mode, device, matchups=matchups)
        score = min(max(r["score"], 1e-3), 1 - 1e-3)
        results[key] = {"first": first, "second": second, "first_path": bots[first], "second_path": bots[second],
                        **{k: r[k] for k in ("score", "ci95", "wins", "draws", "losses", "crashes")}, **setting}
        print(f"{key}: {r['wins']}W {r['draws']}D {r['losses']}L score {r['score']:.3f} +/- {r['ci95']:.3f} "
              f"= {400 * math.log10(score / (1 - score)):+.0f} Elo ({time.time() - start:.0f}s, {r['crashes']} crashed)",
              flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out + ".tmp", "w") as handle:
            json.dump(results, handle, indent=1)
        os.replace(args.out + ".tmp", args.out)
    report(results, args.anchor)


if __name__ == "__main__":
    main()
