"""A knockout among thousands of decks, all piloted by one checkpoint.

    python scripts/arena_deck_tournament.py runs/arena/snapshots/00052.pt
    python scripts/arena_deck_tournament.py CHECKPOINT --random 2000 --top 1000 --mutants 1000

The field: the training pool, the run's most played self-play decks, random
playable decks, decks drawn from the best-scoring cards of the run's self-play
statistics (``deck_stats.json`` next to the checkpoint), and one- or two-card
mutations of the best-scoring decks. Every round each surviving deck plays
``--games`` games against other survivors drawn at random, the checkpoint
playing both seats, and the better half goes on; later rounds give each deck
more games, since the survivors are closer in strength. When ``--final`` decks are
left they play ``--final-games`` each, and that table is the result.

What it measures is how well *this checkpoint* does with a deck against the
same checkpoint with other decks. A deck the network has had little practice
with scores below its worth, and a deck that wins through a simulator bug
scores as high as one that wins fairly. Read the top of the table as
candidates for focus decks, not as a verdict.

Results are written next to the checkpoint as ``deck_tournament.json`` after
every round (``--out`` renames it), with a copy per stage, so a stopped
tournament still leaves its standings and the early rounds' wider tables stay.
``--decks`` enters given decks, an earlier tournament's best for one, and
mutates them too; ``--mutant-any`` sets how often a mutation reaches for any
card instead of a well-scoring one, which is how a card the run's statistics
do not favour gets tried inside a strong deck.
"""

import argparse
import json
import os
import random
import time
from collections import deque

import numpy as np
import torch

from cfbot.agent import Agent
from cfbot.config import pick_device
from cfbot.decks import ARENA_POOL
from cfbot.env import POLICY_KEYS, card_ids, make_env, trim_entities
from cfbot.league import load_policy

SIMULATOR = {"simulator": "arena", "crforge_install": None, "crforge_tables": None}
BUCKET = 128


def playable(deck, kind, champion, cost) -> bool:
    spells = sum(kind[c] == "spell" for c in deck)
    buildings = sum(kind[c] == "building" for c in deck)
    troops = sum(kind[c] == "troop" for c in deck)
    total = sum(cost[c] for c in deck)
    return (len(set(deck)) == 8 and sum(champion[c] for c in deck) <= 1 and 1 <= spells <= 3
            and buildings <= 2 and troops >= 4 and 21 <= total <= 37)


def play_round(env, agent, decks, matches, device, seed, label=""):
    """Play ``matches`` (pairs of deck indices) and return each one's result
    for its first deck: 1 win, 0.5 draw, 0 loss, nan if it did not finish.
    Every battle slot keeps one match waiting behind the one it is playing, so
    no slot ever idles on a game nobody asked for."""
    n = env.num_envs
    results = np.full(len(matches), np.nan)
    waiting = deque(range(len(matches)))
    slots = [deque() for _ in range(n)]

    def feed(slot):
        if waiting:
            match = waiting.popleft()
            a, b = matches[match]
            env.queue_match(slot, decks[a], decks[b], (seed * 1_000_003 + match) % (1 << 62))
            slots[slot].append(match)

    for slot in range(n):
        feed(slot)
        feed(slot)
    obs = env.reset()
    left = len(matches)
    started = time.time()
    steps = 0
    while left and steps < 200_000:
        choice = env.last_choice.nonzero().squeeze(-1)
        actions = torch.zeros(env.rows, dtype=torch.int64)
        if len(choice):
            # Rows are padded to a multiple of BUCKET: a GPU backend compiles its
            # kernels per batch shape, and a new shape every step is very slow.
            pad = (-len(choice)) % BUCKET
            index = torch.cat([choice, choice[:1].expand(pad)]).to(device)
            rows = trim_entities({key: obs[key].index_select(0, index) for key in POLICY_KEYS})
            # On CUDA, the precision the policy was trained and collected in.
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                actions[choice] = agent.act(rows).cpu()[: len(choice)]
        obs, info = env.step(actions)
        steps += 1
        for slot in np.nonzero((info["outcome"] >= 0) & info["queued"])[0]:
            if not slots[slot]:
                continue
            match = slots[slot].popleft()
            outcome = int(info["outcome"][slot])
            if outcome in (0, 1, 2):
                results[match] = (1.0, 0.0, 0.5)[outcome]
            left -= 1
            feed(slot)
    rate = (len(matches) - left) / max(1e-9, time.time() - started)
    print(f"  {label}{len(matches) - left} games in {time.time() - started:.0f} s ({rate:.1f} games/s)", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--random", type=int, default=1000, help="random playable decks")
    parser.add_argument("--top", type=int, default=1000, help="decks drawn from the best-scoring cards")
    parser.add_argument("--mutants", type=int, default=1000, help="mutations of the best-scoring decks")
    parser.add_argument("--top-cards", type=int, default=40, help="how many of the best cards the --top decks draw from")
    parser.add_argument("--games", type=int, default=8, help="games per deck in a knockout round")
    parser.add_argument("--final", type=int, default=32, help="decks in the final table")
    parser.add_argument("--final-games", type=int, default=128)
    parser.add_argument("--envs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--decks", default=None, metavar="FILE",
                        help="JSON list of decks (or of entries with a 'deck') to enter as 'given' and to mutate")
    parser.add_argument("--mutant-any", type=float, default=0.3,
                        help="share of a mutation's new cards drawn from all cards, not the best-scoring ones")
    parser.add_argument("--mutate-given", action="store_true",
                        help="mutate only the decks of --decks, not the self-play favourites or the pool")
    parser.add_argument("--only-given", action="store_true",
                        help="the field is the --decks file alone: no pool, self-play favourites or generated decks")
    parser.add_argument("--out", default="deck_tournament.json", help="result file name, next to the checkpoint")
    args = parser.parse_args()
    device = pick_device("auto")
    rng = random.Random(args.seed)
    run_dir = os.path.dirname(os.path.abspath(args.checkpoint))

    env = make_env(args.envs, **SIMULATOR, seed=args.seed, device=device)
    spec = env.spec
    names = spec["card_names"]
    table = np.asarray(spec["card_table"])
    cards = [name for token, name in enumerate(names) if name and spec["card_supported"][token]]
    token = {name: index for index, name in enumerate(names) if name}
    kind = {c: "troop" if table[token[c], 2] else "building" if table[token[c], 3] else "spell" for c in cards}
    champion = {c: bool(table[token[c], 6]) for c in cards}
    cost = {c: int(round(table[token[c], 1] * 10)) for c in cards}
    ok = lambda deck: playable(deck, kind, champion, cost)  # noqa: E731

    field: dict[tuple, dict] = {}

    def add(deck, source):
        key = tuple(sorted(deck))
        if key not in field and len(set(deck)) == 8:
            field[key] = {"deck": list(deck), "source": source}

    given = []
    if args.decks:
        with open(args.decks) as handle:
            given = [entry["deck"] if isinstance(entry, dict) else entry for entry in json.load(handle)]
        for deck in given:
            add(deck, "given")
    if args.only_given:
        args.random = args.top = args.mutants = 0
    for name, deck in ({} if args.only_given else ARENA_POOL).items():
        add(deck, f"pool:{name}")
    stats_path = os.path.join(run_dir, "deck_stats.json")
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {"cards": {}, "decks": []}
    if args.only_given:
        stats = {"cards": {}, "decks": []}
    seen = [entry for entry in stats["decks"] if entry["games"] >= 100]
    seen.sort(key=lambda entry: -entry["score"])
    for entry in seen:
        if all(card in token for card in entry["deck"]):
            add(entry["deck"], "selfplay")
    # Random decks under the sampler's own rules.
    for deck in env.random_decks(args.random, args.seed):
        add([names[t] for t in deck], "random")
    # Decks of the cards that score best in self-play, better cards more often.
    scored = sorted(((c, v["score"]) for c, v in stats["cards"].items() if c in cost and v["games"] >= 500),
                    key=lambda item: -item[1])[: args.top_cards]
    made = tries = 0
    while scored and made < args.top and tries < args.top * 400:
        tries += 1
        pool, weights = zip(*[(c, max(0.02, s - 0.45)) for c, s in scored])
        deck = []
        while len(deck) < 8:
            card = rng.choices(pool, weights)[0]
            if card not in deck:
                deck.append(card)
        if ok(deck) and tuple(sorted(deck)) not in field:
            add(deck, "top-cards")
            made += 1
    # One or two cards of a strong deck replaced by a good card.
    bases = given + [entry["deck"] for entry in seen[:16]] + list(ARENA_POOL.values())
    if args.mutate_given and given:
        bases = given
    good = [c for c, _ in scored] or cards
    made = tries = 0
    while made < args.mutants and tries < args.mutants * 400:
        tries += 1
        deck = list(rng.choice(bases))
        for _ in range(rng.choice((1, 1, 2))):
            card = rng.choice(cards if rng.random() < args.mutant_any else good)
            if card not in deck:
                deck[rng.randrange(8)] = card
        if ok(deck) and tuple(sorted(deck)) not in field:
            add(deck, "mutant")
            made += 1

    entries = list(field.values())
    for entry in entries:
        entry["tokens"] = card_ids(spec, entry["deck"])
        entry["rounds"] = []
    sources = {}
    for entry in entries:
        sources[entry["source"].split(":")[0]] = sources.get(entry["source"].split(":")[0], 0) + 1
    print(f"{len(entries)} decks: {sources}", flush=True)

    model, _ = load_policy(args.checkpoint, spec, device, ema=args.ema)
    agent = Agent(model, mode="sample", seed=args.seed)
    decks = [entry["tokens"] for entry in entries]
    alive = list(range(len(entries)))
    out_path = os.path.join(run_dir, args.out)

    def save(stage):
        ranked = sorted(alive, key=lambda i: -entries[i]["rounds"][-1][0]) if entries[alive[0]]["rounds"] else alive
        result = {"checkpoint": args.checkpoint, "stage": stage, "field": len(entries), "sources": sources,
                  "standing": [{"deck": entries[i]["deck"], "source": entries[i]["source"],
                                "rounds": entries[i]["rounds"]} for i in ranked]}
        # The latest standing, and one file per stage: a later stage holds fewer decks.
        for path in (out_path, out_path.replace(".json", f".{stage.replace(' ', '_')}.json")):
            with open(path, "w") as handle:
                json.dump(result, handle, indent=1)

    def round_of(games, label):
        # Each deck meets `games` opponents drawn from the survivors: games/2 random pairings.
        matches = []
        for _ in range(max(1, games // 2)):
            order = alive[:]
            rng.shuffle(order)
            if len(order) % 2:
                order.append(rng.choice(order[:-1]))
            matches += [(order[i], order[i + 1]) for i in range(0, len(order), 2)]
            matches += [(order[i + 1], order[i]) for i in range(0, len(order), 2)]
        results = play_round(env, agent, decks, matches, device, rng.randrange(1 << 30), label)
        score = {i: [0.0, 0] for i in alive}
        for (a, b), result in zip(matches, results):
            if not np.isnan(result):
                score[a][0] += result
                score[a][1] += 1
                score[b][0] += 1 - result
                score[b][1] += 1
        for i in alive:
            entries[i]["rounds"].append((round(score[i][0] / max(1, score[i][1]), 4), score[i][1]))

    number = 0
    games = args.games
    while len(alive) > args.final:
        number += 1
        print(f"round {number}: {len(alive)} decks, {games} games each", flush=True)
        round_of(games, "")
        # Fewer decks, closer in strength: each needs more games to be told apart.
        games = min(64, 2 * round(games * 0.75))
        alive.sort(key=lambda i: -entries[i]["rounds"][-1][0])
        alive = alive[: max(args.final, len(alive) // 2)]
        save(f"after round {number}")
        best = entries[alive[0]]
        print(f"  best {best['rounds'][-1][0]:.3f} {best['source']}: {','.join(best['deck'])}", flush=True)
    print(f"final: {len(alive)} decks, {args.final_games} games each", flush=True)
    round_of(args.final_games, "")
    alive.sort(key=lambda i: -entries[i]["rounds"][-1][0])
    save("final")
    for place, i in enumerate(alive, 1):
        entry = entries[i]
        score, games = entry["rounds"][-1]
        print(f"{place:3d}. {score:.3f} ({games} games) {entry['source']:22s} {','.join(entry['deck'])}")
    print(f"written to {out_path}")


if __name__ == "__main__":
    main()
