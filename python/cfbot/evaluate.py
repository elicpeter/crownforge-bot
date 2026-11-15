"""Paired evaluation and the rating gauntlet.

    python -m cfbot.evaluate runs/main/checkpoint.pt random --pairs 200
    python -m cfbot.evaluate A.pt B.pt --mode greedy
    python -m cfbot.evaluate --ratings runs/main

Each side is a checkpoint path or a scripted bot: ``random``, ``spend8``,
``spend5``. A comparison plays a fixed list of deck pairs. Every pair is played
twice from the same seed with the seats swapped, so the first side holds the
same deck in both games; one game is counted per battle slot, so short games
are not over-represented; and a simulator crash is reported, not scored.

During training the same routine runs as a gauntlet against fixed anchors
(the scripted bots and frozen snapshots at doubling iterations), always on the
same deck pairs and seeds. ``--ratings`` fits one rating scale over every
gauntlet a run has recorded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from .agent import Agent
from .baselines import RandomBot, SpendBot
from .config import DRAFT_CARDS, pick_device
from .decks import ARCHETYPES
from .env import Env, card_ids, select_rows
from .league import load_policy

SCRIPTED = {
    "random": lambda seed: RandomBot(0.2, seed),
    "spend8": lambda seed: SpendBot(8.0, seed),
    "spend5": lambda seed: SpendBot(5.0, seed),
}
GAUNTLET_SEED = 20261112
# A queued game that has not finished after this many steps is abandoned.
MAX_STEPS = 6000


class Player:
    """One side of a comparison: a scripted bot's name, a checkpoint path, or
    a loaded policy."""

    def __init__(self, who, spec: dict, device, mode: str, seed: int, ema: bool = False):
        self.bot = self.agent = None
        if isinstance(who, str) and who in SCRIPTED:
            self.bot = SCRIPTED[who](seed)
        elif isinstance(who, str):
            model, _ = load_policy(who, spec, torch.device(device), ema=ema)
            self.agent = Agent(model, mode=mode, seed=seed)
        else:
            self.agent = Agent(who, mode=mode, seed=seed)

    def act(self, obs) -> torch.Tensor:
        return self.bot.act(obs) if self.bot is not None else self.agent.act(obs)


def deck_pairs(env: Env, count: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """A fixed list of matchups: the archetypes and as many random playable
    decks, drawn from ``seed`` alone."""
    decks = [card_ids(env.spec, deck) for deck in ARCHETYPES.values()]
    decks += env.random_decks(len(decks), seed)
    rng = random.Random(seed)
    return [(rng.choice(decks), rng.choice(decks)) for _ in range(count)]


@torch.no_grad()
def play_paired(
    first,
    second,
    pairs: int = 100,
    seed: int = 1234,
    mode: str = "sample",
    device="cpu",
    rules: str | None = None,
    ema: bool = False,
    max_envs: int = 256,
) -> dict:
    """Score ``first`` against ``second`` over ``pairs`` deck pairs, two
    seat-swapped games each."""
    device = torch.device(device)
    scores: list[tuple[float, float]] = []
    wins = draws = losses = crashes = unplayed = 0
    lengths: list[int] = []
    crash_decks: list = []
    matchups = None
    players = None
    for start in range(0, pairs, max_envs // 2):
        n = min(max_envs // 2, pairs - start)
        env = Env(2 * n, seed=seed + start, exclude_cards=DRAFT_CARDS, rules=rules, device=device)
        if matchups is None:
            matchups = deck_pairs(env, pairs, seed)
            players = [
                Player(first, env.spec, device, mode, seed, ema),
                Player(second, env.spec, device, mode, seed + 1, ema),
            ]
        for k, (ours, theirs) in enumerate(matchups[start : start + n]):
            game_seed = (seed * 1_000_003 + start + k) % (1 << 62)
            # Battle 2k: the first side is Blue. Battle 2k + 1: it is Red.
            env.queue_match(2 * k, ours, theirs, game_seed)
            env.queue_match(2 * k + 1, theirs, ours, game_seed)
        obs = env.reset()
        side = torch.arange(2 * n) % 2
        rows = [(torch.arange(2 * n) * 2 + side).to(device), (torch.arange(2 * n) * 2 + 1 - side).to(device)]
        result = np.full(2 * n, np.nan)
        pending = np.ones(2 * n, dtype=bool)
        for _ in range(MAX_STEPS):
            actions = torch.zeros(env.rows, dtype=torch.int64, device=device)
            for player, player_rows in zip(players, rows):
                actions[player_rows] = player.act(select_rows(obs, player_rows))
            obs, info = env.step(actions)
            for index in np.nonzero((info["outcome"] >= 0) & info["queued"] & pending)[0]:
                pending[index] = False
                outcome = int(info["outcome"][index])
                if outcome == 3:
                    crashes += 1
                    continue
                lengths.append(int(info["episode_ticks"][index]))
                if outcome == 2:
                    result[index] = 0.5
                    draws += 1
                elif outcome == int(side[index]):
                    result[index] = 1.0
                    wins += 1
                else:
                    result[index] = 0.0
                    losses += 1
            if not pending.any():
                break
        unplayed += int(pending.sum())
        crash_decks += env.crashes()
        scores += [(result[2 * k], result[2 * k + 1]) for k in range(n)]

    per_pair = np.array([np.nanmean(pair) for pair in scores if not np.all(np.isnan(pair))])
    games = wins + draws + losses
    score = float(per_pair.mean()) if len(per_pair) else 0.5
    # The pair, not the game, is the independent unit.
    spread = float(per_pair.std(ddof=1)) if len(per_pair) > 1 else 0.5
    return {
        "wins": wins, "draws": draws, "losses": losses, "games": games,
        "crashes": crashes, "unplayed": unplayed, "crash_decks": crash_decks,
        "score": score, "ci95": 1.96 * spread / math.sqrt(max(1, len(per_pair))),
        "mean_seconds": float(np.mean(lengths)) * 0.05 if lengths else 0.0,
    }


# ------------------------------------------------------------------ gauntlet


def run_gauntlet(model, league, config, device, iteration: int) -> dict:
    """Play the current policy against every anchor and append the result to
    the run's ``gauntlet.jsonl``. Returns numbers for the training log."""
    opponents: dict[str, object] = {name: name for name in SCRIPTED}
    for snapshot in league.anchors()[-config.gauntlet_anchors :]:
        if snapshot["iteration"] < iteration:
            opponents[f"anchor{snapshot['iteration']}"] = league.path(snapshot)
    results = {}
    for name, who in opponents.items():
        outcome = play_paired(model, who, config.gauntlet_pairs, GAUNTLET_SEED, "sample", device,
                              config.rules)
        results[name] = {key: outcome[key] for key in ("score", "ci95", "games", "crashes")}
        for blue, red in outcome["crash_decks"][:5]:
            print(f"  simulator crash in gauntlet: {blue} vs {red}", flush=True)
    path = os.path.join(config.run_dir, "gauntlet.jsonl")
    with open(path, "a") as handle:
        handle.write(json.dumps({"iteration": iteration, "time": time.time(), "results": results}) + "\n")
    ratings = fit_ratings(read_gauntlet(config.run_dir))
    out = {"gauntlet_elo": round(ratings.get(f"it{iteration}", 0.0), 1)}
    out.update({f"g_{name}": round(entry["score"], 3) for name, entry in results.items()})
    return out


def read_gauntlet(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "gauntlet.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fit_ratings(records: list[dict]) -> dict[str, float]:
    """Bradley-Terry ratings on the Elo scale over all gauntlet results, with
    the ``random`` bot at 0. Anchors named ``anchorN`` are the same player as
    the checkpoint ``itN`` was when it was frozen, so the scale is connected
    across the run."""
    won: dict[tuple[str, str], float] = {}
    names: set[str] = {"random"}
    for record in records:
        me = f"it{record['iteration']}"
        for opponent, entry in record["results"].items():
            other = f"it{opponent[6:]}" if opponent.startswith("anchor") else opponent
            names.update((me, other))
            won[me, other] = won.get((me, other), 0.0) + entry["score"] * entry["games"]
            won[other, me] = won.get((other, me), 0.0) + (1 - entry["score"]) * entry["games"]
    order = sorted(names)
    # Half a win each way against the reference keeps every rating finite.
    for name in order:
        if name != "random":
            won[name, "random"] = won.get((name, "random"), 0.0) + 0.5
            won["random", name] = won.get(("random", name), 0.0) + 0.5
    strength = {name: 1.0 for name in order}
    for _ in range(300):
        updated = {}
        for name in order:
            wins = sum(won.get((name, other), 0.0) for other in order)
            weight = sum(
                (won.get((name, other), 0.0) + won.get((other, name), 0.0))
                / (strength[name] + strength[other])
                for other in order
                if other != name
            )
            updated[name] = wins / weight if weight > 0 else strength[name]
        strength = updated
    return {name: 400.0 * math.log10(strength[name] / strength["random"]) for name in order}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("first", nargs="?")
    parser.add_argument("second", nargs="?")
    parser.add_argument("--ratings", metavar="RUN_DIR", help="print gauntlet ratings of a run")
    parser.add_argument("--pairs", type=int, default=100, help="deck pairs; two games each")
    parser.add_argument("--mode", default="sample", choices=["sample", "greedy"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ema", action="store_true", help="use averaged weights where a checkpoint has them")
    parser.add_argument("--rules", default=None, help="'complete' or a rules document path")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.ratings:
        ratings = fit_ratings(read_gauntlet(args.ratings))
        for name, rating in sorted(ratings.items(), key=lambda item: item[1]):
            print(f"{name:>12s} {rating:8.1f}")
        return
    if not (args.first and args.second):
        parser.error("give two sides, or --ratings RUN_DIR")
    result = play_paired(args.first, args.second, args.pairs, args.seed, args.mode,
                         pick_device(args.device), args.rules, args.ema)
    print(
        f"{args.first} vs {args.second}: {result['wins']}W {result['draws']}D {result['losses']}L, "
        f"score {result['score']:.3f} +/- {result['ci95']:.3f} over {args.pairs} deck pairs"
        f" ({result['crashes']} crashed, {result['unplayed']} unplayed)"
    )
    for blue, red in result["crash_decks"]:
        print("simulator crash:", blue, "vs", red)


if __name__ == "__main__":
    main()
