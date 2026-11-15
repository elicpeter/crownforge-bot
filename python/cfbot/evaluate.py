"""Head-to-head evaluation.

    python -m cfbot.evaluate runs/main/checkpoint.pt random --games 400
    python -m cfbot.evaluate runs/main/checkpoint.pt runs/main/snapshots/00010.pt
    python -m cfbot.evaluate A.pt B.pt --mode greedy --mirror

Each side is a checkpoint path or a scripted bot: ``random``, ``spend8``,
``spend5``. Sides alternate between Blue and Red across battles.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from .agent import Agent
from .baselines import RandomBot, SpendBot
from .config import DRAFT_CARDS, pick_device
from .decks import resolve_pool
from .env import Env, select_rows

SCRIPTED = {"random": lambda: RandomBot(0.2), "spend8": lambda: SpendBot(8.0), "spend5": lambda: SpendBot(5.0)}


class Player:
    def __init__(self, name: str, env: Env, rows: torch.Tensor, mode: str, seed: int):
        self.rows = rows
        if name in SCRIPTED:
            self.bot, self.agent = SCRIPTED[name](), None
        else:
            self.bot = None
            self.agent = Agent.load(name, env.spec, env.device, mode=mode, seed=seed)

    def act(self, obs) -> torch.Tensor:
        sub = select_rows(obs, self.rows)
        return self.bot.act(sub) if self.bot is not None else self.agent.act(sub)


def play(first, second, games, envs, mode, seed, mirror, device, pool_prob=0.5) -> dict:
    env = Env(envs, seed=seed, mirror_decks=mirror, exclude_cards=DRAFT_CARDS, device=device,
              deck_pool=resolve_pool("archetypes"), pool_prob=pool_prob)
    obs = env.reset()
    # The first player is Blue in even battles and Red in odd ones.
    side = torch.arange(envs) % 2
    first_rows = (torch.arange(envs) * 2 + side).to(device)
    second_rows = (torch.arange(envs) * 2 + 1 - side).to(device)
    players = [Player(first, env, first_rows, mode, seed), Player(second, env, second_rows, mode, seed + 1)]
    wins = draws = losses = 0
    while wins + draws + losses < games:
        actions = torch.zeros(env.rows, dtype=torch.int64, device=device)
        for player in players:
            actions[player.rows] = player.act(obs)
        obs, info = env.step(actions)
        for index in np.nonzero(info["outcome"] >= 0)[0]:
            outcome = int(info["outcome"][index])
            if outcome >= 2:
                draws += 1
            elif outcome == int(side[index]):
                wins += 1
            else:
                losses += 1
    total = wins + draws + losses
    score = (wins + 0.5 * draws) / total
    margin = 1.96 * math.sqrt(max(score * (1 - score), 1e-9) / total)
    return {"wins": wins, "draws": draws, "losses": losses, "score": score, "ci95": margin}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--mode", default="sample", choices=["sample", "greedy"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mirror", action="store_true", help="both sides use the same deck")
    parser.add_argument("--pool-prob", type=float, default=0.5,
                        help="share of decks drawn from the archetype list; the rest are random")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = play(args.first, args.second, args.games, args.envs, args.mode, args.seed, args.mirror,
                  pick_device(args.device), args.pool_prob)
    print(
        f"{args.first} vs {args.second}: {result['wins']}W {result['draws']}D {result['losses']}L, "
        f"score {result['score']:.3f} +/- {result['ci95']:.3f}"
    )


if __name__ == "__main__":
    main()
