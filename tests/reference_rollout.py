"""Frozen copy of the collector as it was before the throughput work
(commit a09ef80 plus the value_target knob). The tests compare the optimized
collector against it; do not edit it to make a test pass."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from cfbot import actions as A
from cfbot.config import TrainConfig
from cfbot.decks import DeckStats
from cfbot.env import ENTITY_KEYS, Env, observed_towers, select_rows, tower_potential
from cfbot.league import League
from reference_model import Policy


def opponent_rows(values: torch.Tensor) -> torch.Tensor:
    """Per-row values of the other player in the same battle."""
    return values.view(-1, 2).flip(1).reshape(-1)


class Collector:
    """Steps the environment batch with the learner and its league opponents.

    Each battle has one opponent id (0 means the learner plays both sides),
    fixed by the league's block layout, and for league games the side the
    learner controls, redrawn whenever that battle finishes.
    """

    def __init__(
        self,
        config: TrainConfig,
        env: Env,
        model: Policy,
        league: League,
        deck_stats: DeckStats | None = None,
        seed: int = 0,
    ):
        self.config = config
        self.env = env
        self.model = model
        self.league = league
        self.deck_stats = deck_stats
        self.device = env.device
        self.obs = env.reset()
        self.obs_cpu = env.last_cpu
        self.rng = np.random.default_rng(seed + 5)
        self.flip_rng = torch.Generator().manual_seed(seed + 11)
        self.opponent = np.zeros(env.num_envs, dtype=np.int64)
        self.learner_side = self.rng.integers(2, size=env.num_envs)
        # Games whose opponent changed mid-game do not count for ratings.
        self.tainted = np.zeros(env.num_envs, dtype=bool)
        self.relayout(0.0)

    def relayout(self, progress: float) -> None:
        """Apply the league's block layout for this point of the run. Battles
        whose opponent changes keep playing, but that game is not rated."""
        layout = np.asarray(self.league.layout(progress, self.env.num_envs), dtype=np.int64)
        self.tainted |= layout != self.opponent
        self.opponent = layout

    def taint(self, opponents: list[int]) -> None:
        if opponents:
            self.tainted |= np.isin(self.opponent, opponents)

    def _controllers(self) -> torch.Tensor:
        """Controller id per row: 0 for the learner, otherwise the opponent id."""
        control = np.repeat(self.opponent, 2)
        rows = np.arange(self.env.num_envs) * 2 + self.learner_side
        control[rows] = 0
        return torch.from_numpy(control)

    @torch.no_grad()
    def collect(self, iteration: int, shaping: float) -> tuple[dict, dict]:
        config, env, model = self.config, self.env, self.model
        steps, rows, device = config.rollout_steps, env.rows, self.device
        model.eval()
        amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.amp and device.type == "cuda")

        obs_steps: list[dict[str, torch.Tensor]] = []
        ints = torch.zeros(2, steps, rows, dtype=torch.int64)
        floats = torch.zeros(4, steps, rows)
        flags = torch.zeros(4, steps, rows, dtype=torch.bool)
        stats = defaultdict(float)
        lengths = []

        for t in range(steps):
            control = self._controllers()
            actions = torch.zeros(rows, dtype=torch.int64, device=device)

            learner_cpu = (control == 0).nonzero().squeeze(-1)
            learner_rows = learner_cpu.to(device)
            everyone = learner_rows.numel() == rows
            flipped = None
            if config.flip_augment:
                # Mirror the view of a random half of the learner's rows. The
                # stored observation, cell and log-probability are the ones the
                # policy really used; only the cell sent to the simulator is
                # mirrored back, so every transition is a real one.
                flipped = (control == 0) & (torch.rand(rows, generator=self.flip_rng) < 0.5)
                self.obs_cpu = A.flip_rows(self.obs_cpu, flipped)
                self.obs = {key: value.to(device) for key, value in self.obs_cpu.items()}
            with amp:
                out = model.act(self.obs if everyone else select_rows(self.obs, learner_rows))
            if flipped is None:
                actions[learner_rows] = out["flat"]
            else:
                mirror = flipped[learner_cpu].to(device)
                real_cell = torch.where(mirror, A.flip_cell(out["cell"]), out["cell"])
                actions[learner_rows] = A.to_flat(out["type"], real_cell)
            ints[0, t, learner_cpu] = out["type"].cpu()
            ints[1, t, learner_cpu] = out["cell"].cpu()
            floats[0, t, learner_cpu] = out["logp"].float().cpu()
            floats[1, t, learner_cpu] = out["value"].float().cpu()

            for opponent in control.unique().tolist():
                if opponent == 0:
                    continue
                idx = (control == opponent).nonzero().squeeze(-1).to(device)
                sub = select_rows(self.obs, idx)
                if self.league.is_baseline(opponent):
                    actions[idx] = self.league.baseline(opponent).act(sub)
                else:
                    with amp:
                        reply = self.league.slot_modules[opponent - 1].act(sub, with_value=False)
                    actions[idx] = reply["flat"]

            potential_before = tower_potential(observed_towers(self.obs)).cpu()
            choice = self.obs_cpu["legal"][:, 1:].bool().any(-1)
            stats["forced_wait"] += float((~choice).float().mean())
            obs_steps.append(self.obs_cpu)
            learner = control == 0
            flags[1, t] = learner
            flags[2, t] = choice

            self.obs, info = env.step(actions)
            self.obs_cpu = env.last_cpu
            done = torch.from_numpy(info["done"])
            # A truncated episode (a simulator crash or a time limit) did not
            # reach an outcome: it carries no shaping and no terminal value.
            truncated = torch.from_numpy(info["truncated"])
            flags[3, t] = truncated
            # A step can cover several decision intervals; discount by game time.
            discount = config.gamma ** (torch.from_numpy(info["ticks"]) / config.decision_ticks)
            potential_after = tower_potential(torch.from_numpy(info["final_towers"]))
            potential_after = potential_after.masked_fill(done, 0.0)
            trade = torch.from_numpy(info["trade"])
            leak = torch.from_numpy(info["leak"])
            shaped = (
                config.tower_coef * (discount * potential_after - potential_before)
                + config.trade_coef * (trade - opponent_rows(trade))
                - config.leak_coef * (leak - opponent_rows(leak))
            )
            shaped = shaped.masked_fill(truncated, 0.0)
            floats[2, t] = torch.from_numpy(info["reward"]) + shaping * shaped
            floats[3, t] = discount
            flags[0, t] = done

            played = torch.from_numpy(info["verdict"] == 1)
            stats["plays"] += float(played.sum())
            stats["learner_plays"] += float((played & learner).sum())
            stats["learner_choices"] += float((choice & learner).sum())
            stats["rejected"] += float((info["verdict"] >= 2).sum())
            stats["overflow"] += float(info["entity_overflow"].sum())
            stats["hazard_overflow"] += float(info["hazard_overflow"].sum())
            stats["ticks"] += float(info["ticks"].mean())
            stats["trade"] += float(info["trade"].mean())
            stats["leak"] += float(info["leak"].mean())
            for index in np.nonzero(info["outcome"] >= 0)[0]:
                outcome = int(info["outcome"][index])
                stats["episodes"] += 1
                stats["draws"] += outcome == 2
                stats["crashes"] += outcome == 3
                lengths.append(int(info["episode_ticks"][index]))
                opponent = int(self.opponent[index])
                if opponent != 0 and outcome != 3 and not self.tainted[index]:
                    result = float(info["reward"][index * 2 + self.learner_side[index]])
                    self.league.record(opponent, result)
                if opponent == 0 and outcome != 3 and self.deck_stats is not None:
                    # Self-play only: one policy on both sides, so the result
                    # says something about the decks.
                    self.deck_stats.record(info["decks"][index], float(info["reward"][index * 2]))
                self.learner_side[index] = self.rng.integers(2)
                self.tainted[index] = False

        # Value of the state after the last step, for bootstrapping.
        with amp:
            last = model.value(self.obs).float().cpu()

        width = max(step["entities"].shape[1] for step in obs_steps)
        obs = {}
        for key in obs_steps[0]:
            parts = [step[key] for step in obs_steps]
            if key in ENTITY_KEYS:
                lead = (0, 0) if parts[0].dim() == 3 else ()
                parts = [F.pad(p, lead + (0, width - p.shape[1])) for p in parts]
            obs[key] = torch.stack(parts)

        batch = {
            "obs": obs,
            "type": ints[0],
            "cell": ints[1],
            "logp": floats[0],
            "value": floats[1],
            "reward": floats[2],
            "discount": floats[3],
            "done": flags[0],
            # A truncated step has no valid target, so nothing learns from it.
            "train": flags[1] & ~flags[3],
            "choice": flags[2],
        }
        batch["advantage"], batch["return"] = gae(
            batch["reward"], batch["value"], batch["done"], flags[3], batch["discount"], last,
            config.gae_lambda,
        )
        for key in ("forced_wait", "ticks", "trade", "leak"):
            stats[key] /= steps
        # Share of the learner's real decisions that were a play. Near zero
        # means the policy has collapsed to waiting.
        stats["play_rate"] = stats["learner_plays"] / max(1.0, stats["learner_choices"])
        stats["deck_retries"] = env.deck_retries()
        stats["mean_seconds"] = float(np.mean(lengths)) * env.spec["tick_ms"] / 1000 if lengths else 0.0
        stats["train_frac"] = float(flags[1].float().mean())
        stats["entities"] = width
        return batch, dict(stats)


def gae(reward, value, done, truncated, discount, last_value, lam: float):
    """Generalized advantage estimation over ``[T, N]`` tensors with a per-step
    discount. A finished episode contributes no bootstrap value. A truncated
    step contributes nothing at all: its advantage is zero and it passes no
    error back to earlier steps."""
    steps = reward.shape[0]
    advantage = torch.zeros_like(reward)
    running = torch.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(steps)):
        alive = (~done[t]).float() * discount[t]
        delta = reward[t] + alive * next_value - value[t]
        running = (delta + lam * alive * running).masked_fill(truncated[t], 0.0)
        advantage[t] = running
        next_value = value[t]
    return advantage, advantage + value
