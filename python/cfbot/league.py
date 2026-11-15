"""Opponent pool for self-play.

The learner plays three kinds of opponents:

* itself (both sides learn),
* frozen past snapshots, chosen by prioritized fictitious self-play: snapshots
  the learner still loses to are sampled more often, which stops it from
  forgetting how to beat old strategies,
* scripted baselines, mostly early on.

Opponent ids used by the collector: 0 is the learner itself, ``1..K`` are
snapshot slots, and the rest are baselines.
"""

from __future__ import annotations

import json
import os
import random

import torch

from .baselines import make_baselines
from .config import ModelConfig, TrainConfig
from .model import Policy


class League:
    def __init__(self, config: TrainConfig, spec: dict, device: torch.device):
        self.config = config
        self.spec = spec
        self.device = device
        self.dir = os.path.join(config.run_dir, "snapshots")
        os.makedirs(self.dir, exist_ok=True)
        # Each snapshot: id, iteration, path, elo, and decayed game/score counts.
        self.snapshots: list[dict] = []
        self.slot_modules: list[Policy | None] = [None] * config.league_slots
        self.slot_snapshot: list[int | None] = [None] * config.league_slots
        self.baselines = make_baselines()
        self.baseline_stats = [{"games": 0.0, "score": 0.0} for _ in self.baselines]
        self.elo = 0.0
        self.rng = random.Random(config.seed + 17)

    # ------------------------------------------------------------- opponents

    @property
    def num_slots(self) -> int:
        return self.config.league_slots

    def is_baseline(self, opponent: int) -> bool:
        return opponent > self.num_slots

    def baseline(self, opponent: int):
        return self.baselines[opponent - self.num_slots - 1]

    def baseline_frac(self, iteration: int) -> float:
        c = self.config
        t = min(1.0, iteration / max(1, c.baseline_anneal_iters))
        return c.baseline_frac + (c.baseline_frac_final - c.baseline_frac) * t

    def layout(self, iteration: int, num_envs: int) -> list[int]:
        """Opponent id for every battle.

        Battles are split into fixed blocks: scripted baselines, snapshot slots,
        and self-play for the rest. Block sizes change only when the baseline
        share anneals or the first snapshot appears, so batch shapes stay
        constant between those moments.
        """
        c = self.config
        loaded = [k for k in range(self.num_slots) if self.slot_modules[k] is not None]
        per_baseline = round(num_envs * self.baseline_frac(iteration) / len(self.baselines))
        league_share = max(0.0, 1.0 - c.self_play_frac - self.baseline_frac(iteration))
        per_slot = int(num_envs * league_share / max(1, self.num_slots)) if loaded else 0
        layout: list[int] = []
        for index in range(len(self.baselines)):
            layout += [self.num_slots + 1 + index] * per_baseline
        for slot in loaded:
            layout += [slot + 1] * per_slot
        layout = layout[:num_envs]
        return layout + [0] * (num_envs - len(layout))

    # ------------------------------------------------------------- snapshots

    def win_rate(self, snapshot: dict) -> float:
        return (snapshot["score"] + 1.0) / (snapshot["games"] + 2.0)

    def add_snapshot(self, model: Policy, iteration: int) -> None:
        index = len(self.snapshots)
        path = os.path.join(self.dir, f"{index:05d}.pt")
        tmp = path + ".tmp"
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, tmp)
        os.replace(tmp, path)
        self.snapshots.append(
            {"id": index, "iteration": iteration, "path": path, "elo": self.elo,
             "games": 0.0, "score": 0.0}
        )

    def refresh_slots(self) -> list[int]:
        """Load a newly sampled snapshot into every slot. Returns the slots that
        changed, so games in progress against them can be ignored."""
        if not self.snapshots:
            return []
        power = self.config.pfsp_power
        weights = [(1.0 - self.win_rate(s)) ** power + 0.05 for s in self.snapshots]
        weights[-1] *= 2.0  # keep pressure from the most recent self
        changed = []
        for slot in range(self.num_slots):
            pick = self.rng.choices(range(len(self.snapshots)), weights)[0]
            if self.slot_snapshot[slot] == pick:
                continue
            if self.slot_modules[slot] is None:
                module = Policy(self.spec, self.config.model).to(self.device)
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
                self.slot_modules[slot] = module
            state = torch.load(self.snapshots[pick]["path"], map_location=self.device)
            self.slot_modules[slot].load_state_dict(state)
            self.slot_snapshot[slot] = pick
            changed.append(slot + 1)
        return changed

    # --------------------------------------------------------------- results

    def record(self, opponent: int, result: float) -> None:
        """Record one finished game. ``result`` is +1, 0 or -1 for the learner."""
        score = (result + 1.0) / 2.0
        if self.is_baseline(opponent):
            stats = self.baseline_stats[opponent - self.num_slots - 1]
            stats["games"] = stats["games"] * 0.995 + 1.0
            stats["score"] = stats["score"] * 0.995 + score
            return
        snapshot_id = self.slot_snapshot[opponent - 1]
        if snapshot_id is None:
            return
        snapshot = self.snapshots[snapshot_id]
        snapshot["games"] = snapshot["games"] * 0.995 + 1.0
        snapshot["score"] = snapshot["score"] * 0.995 + score
        expected = 1.0 / (1.0 + 10.0 ** ((snapshot["elo"] - self.elo) / 400.0))
        self.elo += 4.0 * (score - expected)

    def summary(self) -> dict:
        out = {"elo": round(self.elo, 1), "snapshots": len(self.snapshots)}
        for bot, stats in zip(self.baselines, self.baseline_stats):
            if stats["games"] >= 5:
                key = f"win_{bot.name}{getattr(bot, 'threshold', '')}"
                out[key] = round(stats["score"] / stats["games"], 3)
        return out

    # ----------------------------------------------------------- persistence

    def state_dict(self) -> dict:
        return {
            "snapshots": self.snapshots,
            "slot_snapshot": self.slot_snapshot,
            "baseline_stats": self.baseline_stats,
            "elo": self.elo,
        }

    def load_state_dict(self, state: dict) -> None:
        self.snapshots = [s for s in state["snapshots"] if os.path.exists(s["path"])]
        self.baseline_stats = state["baseline_stats"]
        self.elo = state["elo"]
        self.slot_snapshot = [None] * self.num_slots
        self.refresh_slots()

    def save_index(self) -> None:
        path = os.path.join(self.config.run_dir, "league.json")
        with open(path + ".tmp", "w") as handle:
            json.dump(self.state_dict(), handle, indent=1)
        os.replace(path + ".tmp", path)


def load_policy(path: str, spec: dict, device: torch.device) -> tuple[Policy, dict]:
    """Load a training checkpoint or a bare snapshot for play or evaluation."""
    data = torch.load(path, map_location=device, weights_only=False)
    if "model" in data:
        config = TrainConfig.from_dict(data["config"]).model
        weights = data["model"]
    else:
        # Bare snapshot: the run's config sits two directories up.
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(path)))
        with open(os.path.join(run_dir, "config.json")) as handle:
            config = ModelConfig(**json.load(handle)["model"])
        weights = data
    model = Policy(spec, config).to(device)
    model.load_state_dict(weights)
    model.eval()
    return model, data if "model" in data else {}
