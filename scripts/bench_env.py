"""Model-free stepping benchmark: the simulator and bridge alone.

    python scripts/bench_env.py                       # 256 battles, default workers
    python scripts/bench_env.py --workers 4,6,8,10,12 # sweep the thread pool

Every row is driven by a scripted policy that plays a random legal action with
a fixed probability, drawn outside the timed region, so the timed call is
exactly ``VecEnv.step``: parallel battle stepping, featurizing, and gathering
the observation arrays for Python. ``--play`` sets how busy the boards get.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch

from cfbot.config import DRAFT_CARDS
from cfbot.decks import resolve_pool
from cfbot.env import Env


def scripted_actions(legal: torch.Tensor, play: float, generator: torch.Generator) -> np.ndarray:
    weights = legal.float()
    weights[:, 0] = 0
    has_play = weights.sum(-1) > 0
    weights[~has_play, 0] = 1
    choice = torch.multinomial(weights, 1, generator=generator).squeeze(-1)
    draw = torch.rand(choice.shape[0], generator=generator)
    return torch.where(has_play & (draw < play), choice, torch.zeros_like(choice)).numpy()


def run(num_envs: int, workers: int, steps: int, warmup: int, play: float, seed: int) -> dict:
    env = Env(num_envs, workers=workers, seed=seed, exclude_cards=DRAFT_CARDS,
              deck_pool=resolve_pool("archetypes"), pool_prob=0.10)
    env.set_mix(0.10, 0.20)
    env.reset(seed=seed)
    generator = torch.Generator().manual_seed(seed)
    raw = env.env  # the Rust object: time it without the torch wrapper
    obs = env.last_cpu
    times, widths, ticks, choice = [], [], [], []
    wrap = []
    for step in range(warmup + steps):
        actions = scripted_actions(obs["legal"], play, generator)
        start = time.perf_counter()
        out = raw.step(actions)
        middle = time.perf_counter()
        obs = {key: torch.from_numpy(out[key]) for key in ("legal", "entities", "entity_mask")}
        end = time.perf_counter()
        if step >= warmup:
            times.append(middle - start)
            wrap.append(end - middle)
            widths.append(out["entities"].shape[1])
            ticks.append(float(out["ticks"].mean()))
            choice.append(float(obs["legal"][:, 1:].bool().any(-1).float().mean()))
    times_ms = np.array(times) * 1e3
    return {
        "workers": workers,
        "mean_ms": float(times_ms.mean()),
        "median_ms": float(np.median(times_ms)),
        "p90_ms": float(np.percentile(times_ms, 90)),
        "rows_per_s": num_envs * 2 / float(np.mean(times)),
        "entity_width": statistics.mean(widths),
        "ticks_per_step": statistics.mean(ticks),
        "choice_frac": statistics.mean(choice),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--workers", default="0", help="comma-separated pool sizes; 0 = all cores")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=400, help="steps before timing, to mix game phases")
    parser.add_argument("--play", type=float, default=0.15, help="chance a row with a choice plays")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    print(f"{args.envs} battles ({args.envs * 2} rows), {args.steps} timed steps after {args.warmup} warm-up, "
          f"play {args.play}")
    for workers in [int(w) for w in args.workers.split(",")]:
        for _ in range(args.repeat):
            r = run(args.envs, workers, args.steps, args.warmup, args.play, args.seed)
            print(f"  workers {r['workers']:2d}: step {r['mean_ms']:6.2f} ms mean, {r['median_ms']:6.2f} median, "
                  f"{r['p90_ms']:6.2f} p90 | {r['rows_per_s']:8.0f} rows/s | width {r['entity_width']:.0f} "
                  f"ticks {r['ticks_per_step']:.1f} choice {r['choice_frac']:.2f}", flush=True)


if __name__ == "__main__":
    main()
