"""Opponent pool for self-play.

The learner plays three kinds of opponents:

* itself (both sides learn),
* frozen past snapshots, chosen by prioritized fictitious self-play: snapshots
  the learner still loses to are sampled more often, which stops it from
  forgetting how to beat old strategies,
* scripted baselines, mostly early on.

Opponent ids used by the collector: 0 is the learner itself, ``1..K`` are
snapshot slots, then come the baselines, and after them any frozen policies
from outside the run (``outside_opponents``).

``lineage_elo`` rates the learner against frozen copies of its own past
rating. It shows whether the learner keeps beating its history; it cannot tell
progress from a cycle and says nothing about other opponents. The gauntlet in
``evaluate.py`` is the strength measurement.
"""

from __future__ import annotations

import json
import os
import random

import torch

from .baselines import baseline_label, make_baselines
from .config import ModelConfig, TrainConfig
from .model import Policy

# The newest snapshots are never thinned.
KEEP_RECENT = 50


class League:
    def __init__(self, config: TrainConfig, spec: dict, device: torch.device):
        self.config = config
        self.spec = spec
        self.device = device
        self.dir = os.path.join(config.run_dir, "snapshots")
        os.makedirs(self.dir, exist_ok=True)
        # Each snapshot: id, iteration, file (inside the snapshot directory),
        # elo, decayed game/score counts, and whether it is a rating anchor.
        self.snapshots: list[dict] = []
        self.next_id = 0
        self.next_anchor = 0
        self.slot_modules: list[Policy | None] = [None] * config.league_slots
        self.slot_snapshot: list[int | None] = [None] * config.league_slots
        self.baselines = make_baselines(spec.get("simulator"), spec)
        self.baseline_stats = [{"games": 0.0, "score": 0.0} for _ in self.baselines]
        self.fixed_stats = {"games": 0.0, "score": 0.0}
        self.elo = 0.0
        self.rng = random.Random(config.seed + 17)
        if config.fixed_opponent:
            module, _ = load_policy(config.fixed_opponent, spec, device)
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            self.slot_modules = [module] + [None] * (config.league_slots - 1)
        # Frozen policies from outside the run: name, path, decks, the loaded
        # network, the file's modification time and decayed results.
        self.outside: list[dict] = []
        if config.outside_opponents and config.outside_frac > 0 and not self.fixed:
            for entry in config.outside_opponents:
                module = Policy(spec, config.model).to(device)
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
                self.outside.append({"name": entry["name"], "path": entry["path"], "decks": entry["decks"],
                                     "learner_focus": entry.get("learner_focus"), "module": module,
                                     "mtime": None, "games": 0.0, "score": 0.0})
            self.refresh_outside(strict=True)

    # ------------------------------------------------------------- opponents

    @property
    def num_slots(self) -> int:
        return self.config.league_slots

    @property
    def fixed(self) -> bool:
        return bool(self.config.fixed_opponent)

    def is_baseline(self, opponent: int) -> bool:
        return self.num_slots < opponent <= self.num_slots + len(self.baselines)

    def baseline(self, opponent: int):
        return self.baselines[opponent - self.num_slots - 1]

    def is_outside(self, opponent: int) -> bool:
        return opponent > self.num_slots + len(self.baselines)

    def outside_of(self, opponent: int) -> dict:
        return self.outside[opponent - self.num_slots - len(self.baselines) - 1]

    def module(self, opponent: int) -> Policy:
        """The frozen network behind a snapshot slot or an outside opponent."""
        if self.is_outside(opponent):
            return self.outside_of(opponent)["module"]
        return self.slot_modules[opponent - 1]

    def refresh_outside(self, strict: bool = False) -> list[int]:
        """Load every outside opponent whose file is new or has changed.
        Returns the opponent ids that changed, so games in progress against
        them can be ignored. A file that cannot be read keeps the weights
        already loaded (it may be half copied), unless ``strict``."""
        changed = []
        for index, entry in enumerate(self.outside):
            try:
                mtime = os.path.getmtime(entry["path"])
                if mtime == entry["mtime"]:
                    continue
                data = torch.load(entry["path"], map_location=self.device, weights_only=False)
                entry["module"].load_state_dict(data["model"] if "model" in data else data)
            except Exception as error:
                if strict:
                    raise
                print(f"league: outside opponent {entry['name']} not reloaded from {entry['path']}: {error}",
                      flush=True)
                continue
            if entry["mtime"] is not None:
                print(f"league: outside opponent {entry['name']} reloaded from {entry['path']}", flush=True)
            entry["mtime"] = mtime
            changed.append(self.num_slots + len(self.baselines) + 1 + index)
        return changed

    def baseline_frac(self, progress: float) -> float:
        c = self.config
        t = min(1.0, progress / max(1e-9, c.baseline_anneal_frac))
        return c.baseline_frac + (c.baseline_frac_final - c.baseline_frac) * t

    def layout(self, progress: float, num_envs: int) -> list[int]:
        """Opponent id for every battle.

        Battles are split into fixed blocks: scripted baselines, snapshot slots,
        and self-play for the rest. Block sizes change only when the baseline
        share anneals or the first snapshot appears, so batch shapes stay
        constant between those moments.
        """
        if self.fixed:
            return [1] * num_envs
        c = self.config
        loaded = [k for k in range(self.num_slots) if self.slot_modules[k] is not None]
        share = self.baseline_frac(progress)
        per_baseline = round(num_envs * share / len(self.baselines))
        league_share = max(0.0, 1.0 - c.self_play_frac - share)
        per_slot = int(num_envs * league_share / max(1, self.num_slots)) if loaded else 0
        layout: list[int] = []
        for index in range(len(self.baselines)):
            layout += [self.num_slots + 1 + index] * per_baseline
        for slot in loaded:
            layout += [slot + 1] * per_slot
        if self.outside:
            per_outside = int(num_envs * c.outside_frac / len(self.outside))
            for index in range(len(self.outside)):
                layout += [self.num_slots + len(self.baselines) + 1 + index] * per_outside
        layout = layout[:num_envs]
        return layout + [0] * (num_envs - len(layout))

    # ------------------------------------------------------------- snapshots

    def path(self, snapshot: dict) -> str:
        return os.path.join(self.dir, snapshot["file"])

    def win_rate(self, snapshot: dict) -> float:
        return (snapshot["score"] + 1.0) / (snapshot["games"] + 2.0)

    def anchors(self) -> list[dict]:
        return [s for s in self.snapshots if s["anchor"]]

    def add_snapshot(self, model: Policy, iteration: int) -> None:
        if self.fixed:
            return
        index = self.next_id
        self.next_id += 1
        name = f"{index:05d}.pt"
        path = os.path.join(self.dir, name)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path + ".tmp")
        os.replace(path + ".tmp", path)
        # Anchors sit at doubling iterations. They are never thinned, so the
        # gauntlet always has the same reference points.
        anchor = iteration >= self.next_anchor
        if anchor:
            self.next_anchor = max(2 * iteration, self.config.snapshot_iters)
        self.snapshots.append(
            {"id": index, "iteration": iteration, "file": name, "elo": self.elo,
             "games": 0.0, "score": 0.0, "anchor": anchor}
        )
        self.thin()

    def thin(self) -> None:
        """Keep disk use bounded: drop old snapshots so the remainder stays
        spread over the run. Anchors, loaded and recent snapshots are kept."""
        limit = self.config.max_snapshots
        while sum(not s["anchor"] for s in self.snapshots) > limit:
            loaded = {i for i in self.slot_snapshot if i is not None}
            old = self.snapshots[:-KEEP_RECENT]
            candidates = [
                position
                for position, snapshot in enumerate(old)
                if not snapshot["anchor"] and position not in loaded and 0 < position < len(old) - 1
            ]
            if not candidates:
                return
            # Remove the snapshot whose neighbours are closest together.
            position = min(
                candidates,
                key=lambda p: self.snapshots[p + 1]["iteration"] - self.snapshots[p - 1]["iteration"],
            )
            removed = self.snapshots.pop(position)
            self.slot_snapshot = [
                None if i is None else (i - 1 if i > position else i) for i in self.slot_snapshot
            ]
            try:
                os.remove(self.path(removed))
            except OSError:
                pass

    def refresh_slots(self) -> list[int]:
        """Load newly sampled snapshots into the slots. Returns the slots that
        changed, so games in progress against them can be ignored.

        The first slot draws from the newest few snapshots. The others draw
        without repeating a snapshot: mostly by how often the learner still
        loses, sometimes uniformly."""
        if not self.snapshots or self.fixed:
            return []
        c = self.config
        count = len(self.snapshots)
        weights = [(1.0 - self.win_rate(s)) ** c.pfsp_power + 1e-3 for s in self.snapshots]
        taken: set[int] = set()
        changed = []
        for slot in range(self.num_slots):
            free = [i for i in range(count) if i not in taken] or list(range(count))
            if slot == 0:
                recent = [i for i in free if i >= count - c.league_recent] or free
                pick = self.rng.choice(recent)
            elif self.rng.random() < c.league_explore:
                pick = self.rng.choice(free)
            else:
                pick = self.rng.choices(free, [weights[i] for i in free])[0]
            taken.add(pick)
            if self.slot_snapshot[slot] == pick:
                continue
            if self.slot_modules[slot] is None:
                module = Policy(self.spec, self.config.model).to(self.device)
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
                self.slot_modules[slot] = module
            state = torch.load(self.path(self.snapshots[pick]), map_location=self.device)
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
        if self.is_outside(opponent):
            stats = self.outside_of(opponent)
            stats["games"] = stats["games"] * 0.995 + 1.0
            stats["score"] = stats["score"] * 0.995 + score
            return
        if self.fixed:
            self.fixed_stats["games"] = self.fixed_stats["games"] * 0.995 + 1.0
            self.fixed_stats["score"] = self.fixed_stats["score"] * 0.995 + score
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
        out = {"lineage_elo": round(self.elo, 1), "snapshots": len(self.snapshots)}
        for bot, stats in zip(self.baselines, self.baseline_stats):
            if stats["games"] >= 5:
                key = f"win_{baseline_label(bot)}"
                out[key] = round(stats["score"] / stats["games"], 3)
        if self.fixed and self.fixed_stats["games"] >= 5:
            out["win_fixed"] = round(self.fixed_stats["score"] / self.fixed_stats["games"], 3)
        for entry in self.outside:
            if entry["games"] >= 5:
                out[f"win_{entry['name']}"] = round(entry["score"] / entry["games"], 3)
        return out

    # ----------------------------------------------------------- persistence

    def state_dict(self) -> dict:
        return {
            "snapshots": self.snapshots,
            "next_id": self.next_id,
            "next_anchor": self.next_anchor,
            "slot_snapshot": self.slot_snapshot,
            "baseline_stats": self.baseline_stats,
            "baseline_names": [baseline_label(bot) for bot in self.baselines],
            "fixed_stats": self.fixed_stats,
            "outside_stats": {entry["name"]: {"games": entry["games"], "score": entry["score"]}
                              for entry in self.outside},
            "elo": self.elo,
        }

    def load_state_dict(self, state: dict) -> None:
        snapshots = []
        for snapshot in state["snapshots"]:
            snapshot = dict(snapshot)
            # Files are named relative to the run's snapshot directory, so a
            # run can be moved or resumed from anywhere.
            snapshot.setdefault("file", os.path.basename(snapshot.pop("path", "")))
            snapshot.pop("path", None)
            snapshot.setdefault("anchor", False)
            if os.path.exists(self.path(snapshot)):
                snapshots.append(snapshot)
            else:
                print(f"league: snapshot {snapshot['file']} is missing from {self.dir}; dropped")
        self.snapshots = snapshots
        self.next_id = max(
            int(state.get("next_id", 0)), 1 + max((s["id"] for s in snapshots), default=-1)
        )
        self.next_anchor = state.get("next_anchor", 0)
        # A checkpoint from before the scripted opponents changed keeps the
        # counts of the ones that are still here and starts the others afresh.
        names = state.get("baseline_names") or ["random", "spend8.0", "spend5.0", "rush"]
        old = dict(zip(names, state["baseline_stats"]))
        self.baseline_stats = [old.get(baseline_label(bot), {"games": 0.0, "score": 0.0}) for bot in self.baselines]
        self.fixed_stats = state.get("fixed_stats", self.fixed_stats)
        for entry in self.outside:
            entry.update(state.get("outside_stats", {}).get(entry["name"], {}))
        self.elo = state["elo"]
        self.slot_snapshot = [None] * self.num_slots
        self.refresh_slots()

    def save_index(self) -> None:
        path = os.path.join(self.config.run_dir, "league.json")
        with open(path + ".tmp", "w") as handle:
            json.dump(self.state_dict(), handle, indent=1)
        os.replace(path + ".tmp", path)


def rules_label(spec: dict) -> dict:
    """What a checkpoint was trained on. Stored as a label, never enforced."""
    return {key: spec[key] for key in ("rules_id", "rules_digest", "base_level", "bot_schema")}


def load_policy(
    path: str, spec: dict, device: torch.device, ema: bool = False
) -> tuple[Policy, dict]:
    """Load a training checkpoint or a bare snapshot for play or evaluation.

    The model always takes its level and timing from the live environment's
    ``spec``; a checkpoint trained under other rules still loads, with a
    warning. ``ema`` loads the averaged actor weights when the checkpoint has
    them."""
    data = torch.load(path, map_location=device, weights_only=False)
    if "model" in data:
        config = TrainConfig.from_dict(data["config"]).model
        weights = dict(data["model"])
        if ema and data.get("ema"):
            weights.update(data["ema"])
        trained_on = data.get("rules")
        if trained_on and trained_on != rules_label(spec):
            print(f"warning: {path} was trained on {trained_on}, now running {rules_label(spec)}")
    else:
        # Bare snapshot: the run's config sits two directories up.
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(path)))
        with open(os.path.join(run_dir, "config.json")) as handle:
            fields = {f for f in ModelConfig.__dataclass_fields__}
            config = ModelConfig(
                **{k: v for k, v in json.load(handle)["model"].items() if k in fields}
            )
        weights = data
    model = Policy(spec, config).to(device)
    model.load_state_dict(weights)
    model.eval()
    return model, data if "model" in data else {}
