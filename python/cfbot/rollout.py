"""Experience collection against the league, and advantage estimation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from . import actions as A
from .config import TrainConfig
from .decks import DeckStats
from .env import (
    CRITIC_KEYS, ENTITY_KEYS, POLICY_KEYS, Env, entity_width, gather_rows, observed_towers,
    tower_potential,
)
from .league import League
from .model import Policy
from .ppo import autocast

# The learner's actor pass runs only on rows that have a choice. Their number
# changes every step, and Apple GPUs compile kernels per tensor shape, so the
# pass is padded to a multiple of this many rows (never beyond the learner's
# row count, which is what the pass used to take).
ROW_BUCKET = 64
# Rows of the stored rollout the critic values in one pass.
VALUE_ROWS = 4096


def opponent_rows(values: torch.Tensor) -> torch.Tensor:
    """Per-row values of the other player in the same battle."""
    return values.view(-1, 2).flip(1).reshape(-1)


def padded(index: torch.Tensor, multiple: int, limit: int | None = None) -> torch.Tensor:
    """``index`` with its first entry repeated up to a multiple of ``multiple``,
    or up to ``limit`` entries if that is fewer (and no fewer than it has)."""
    size = -(-len(index) // multiple) * multiple
    if limit is not None:
        size = max(len(index), min(size, limit))
    extra = size - len(index)
    return torch.cat([index, index[:1].expand(extra)]) if extra else index


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
        self.obs_cpu, self.choice, self.entities = env.last_cpu, env.last_choice, env.last_entity_count
        self.rng = np.random.default_rng(seed + 5)
        self.flip_rng = torch.Generator().manual_seed(seed + 11)
        self.opponent = np.zeros(env.num_envs, dtype=np.int64)
        self.learner_side = self.rng.integers(2, size=env.num_envs)
        focus = getattr(env, "focus_seat", None)
        if focus is not None:
            # Arena only: a battle that holds a focus deck has the learner on that seat.
            self.learner_side = np.where(focus >= 0, focus, self.learner_side).astype(np.int64)
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
        """Play ``rollout_steps`` decisions in every battle.

        Each step makes one round trip to the device: the observation goes up
        as two blocks, the rows that need a network (learner rows with a
        choice, league-slot rows) are gathered there, the actors run, and
        every sampled action comes back in one transfer. Scripted bots,
        rewards and bookkeeping stay on the CPU. The critic does not run
        here; it values the stored rollout afterwards in a few large passes,
        which gives the same numbers.
        """
        config, env, model = self.config, self.env, self.model
        steps, rows, device = config.rollout_steps, env.rows, self.device
        model.eval()
        amp = autocast(config, device)
        # With the rollout on the device, each step's planes stay where they
        # were uploaded for acting and are stacked there, so nothing is
        # uploaded twice.
        on_device = config.rollout_on_device
        if on_device is None:
            on_device = device.type == "cuda"

        obs_steps: list[dict[str, torch.Tensor]] = []
        counts = torch.zeros(steps, rows, dtype=torch.int64)
        ints = torch.zeros(2, steps, rows, dtype=torch.int64)
        floats = torch.zeros(4, steps, rows)
        flags = torch.zeros(4, steps, rows, dtype=torch.bool)
        stats = defaultdict(float)
        lengths = []

        for t in range(steps):
            control = self._controllers()
            learner_cpu = (control == 0).nonzero().squeeze(-1)
            flipped = None
            if config.flip_augment:
                # Mirror the view of a random half of the learner's rows. The
                # stored observation, cell and log-probability are the ones the
                # policy really used; only the cell sent to the simulator is
                # mirrored back, so every transition is a real one.
                flipped = (control == 0) & (torch.rand(rows, generator=self.flip_rng) < 0.5)
                self.obs_cpu = A.flip_rows(self.obs_cpu, flipped)
                self.obs = env.upload(self.obs_cpu, packed=True)
            obs_cpu = self.obs_cpu
            # Per row: whether anything but waiting is legal, and how many
            # entities it holds. Mirroring changes neither.
            choice, entities = self.choice, self.entities
            counts[t] = entities

            # A learner row that can only wait takes action 0 with probability
            # one under any weights, so the actor skips it. The noise is still
            # drawn for every learner row, in the order a full pass would draw
            # it, so the random stream does not depend on which rows are run.
            open_rows = choice[learner_cpu]
            acting = learner_cpu[open_rows]
            learner_noise = (
                torch.rand((len(learner_cpu), A.NUM_TYPES), device=device),
                torch.rand((len(learner_cpu), A.CELLS), device=device),
            )
            groups = []  # (rows, the same padded for the pass, network, noise)
            if len(acting):
                groups.append((acting, padded(acting, ROW_BUCKET, len(learner_cpu)), model, None))
            bots = []
            for opponent in control.unique().tolist():
                if opponent == 0:
                    continue
                idx = (control == opponent).nonzero().squeeze(-1)
                if self.league.is_baseline(opponent):
                    bots.append((idx, self.league.baseline(opponent)))
                    continue
                noise = (
                    torch.rand((len(idx), A.NUM_TYPES), device=device),
                    torch.rand((len(idx), A.CELLS), device=device),
                )
                groups.append((idx, idx, self.league.slot_modules[opponent - 1], noise))

            actions = torch.zeros(rows, dtype=torch.int64)
            if groups:
                # One upload: the rows to gather, then where the learner's
                # acting rows sit in its noise.
                index = [run for _, run, _, _ in groups]
                gathered = sum(len(run) for run in index)
                if len(acting):
                    index.append(padded(open_rows.nonzero().squeeze(-1), ROW_BUCKET, len(learner_cpu)))
                index = torch.cat(index).to(device)
                picked = index[:gathered]
                sub = env.take(self.obs, picked, POLICY_KEYS)
                pieces = []
                start = 0
                for cpu_rows, run, module, noise in groups:
                    part = {key: value[start : start + len(run)] for key, value in sub.items()}
                    start += len(run)
                    if module is model:
                        # Only the rows that act set the entity width.
                        width = entity_width(int(entities[cpu_rows].max()), part["entity_mask"].shape[1])
                        for key in ENTITY_KEYS:
                            part[key] = part[key][:, :width]
                        where = index[gathered:]
                        noise = tuple(draw.index_select(0, where) for draw in learner_noise)
                    with amp:
                        out = module.act(part, with_value=False, noise=noise)
                    pieces.append(out["flat"][: len(cpu_rows)].float())
                    if module is model:
                        pieces.append(out["logp"][: len(cpu_rows)].float())
                # One download for everything the networks decided this step.
                # Action indices are small integers, exact in float32.
                packed = torch.cat(pieces).cpu()
                start = 0
                for cpu_rows, _, module, _ in groups:
                    count = len(cpu_rows)
                    flat = packed[start : start + count].long()
                    start += count
                    if module is model:
                        # The policy chose in the frame it was shown; the
                        # stored type and cell are in that frame.
                        kind, cell = A.from_flat(flat)
                        ints[0, t, cpu_rows] = kind
                        ints[1, t, cpu_rows] = cell
                        floats[0, t, cpu_rows] = packed[start : start + count]
                        start += count
                        if flipped is not None:
                            real = torch.where(flipped[cpu_rows], A.flip_cell(cell), cell)
                            flat = A.to_flat(kind, real)
                    actions[cpu_rows] = flat
            # Scripted bots read two planes and sample on the CPU. They run
            # after the networks, in id order, as they always have: on a CPU
            # device they share the networks' random generator.
            for idx, bot in bots:
                actions[idx] = bot.act({"legal": obs_cpu["legal"][idx], "global": obs_cpu["global"][idx]})

            potential_before = tower_potential(observed_towers(obs_cpu))
            stats["forced_wait"] += float((~choice).float().mean())
            obs_steps.append(self.obs if on_device else obs_cpu)
            learner = control == 0
            flags[1, t] = learner
            flags[2, t] = choice

            self.obs, info = env.step(actions)
            self.obs_cpu, self.choice, self.entities = env.last_cpu, env.last_choice, env.last_entity_count
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
                # The episode that has just started may hold a focus deck on one
                # seat (arena only): the learner takes that seat.
                focus = getattr(self.env, "focus_seat", None)
                if focus is not None and focus[index] >= 0:
                    self.learner_side[index] = int(focus[index])
                self.tainted[index] = False

        obs = self._stack(obs_steps)
        del obs_steps
        floats[1] = self._values(obs, counts, flags[1], amp)
        # Value of the state after the last step, for bootstrapping.
        with amp:
            last = model.value(self.obs).float().cpu()

        batch = {
            "obs": obs,
            # Entities per row, so a minibatch can find its width on the CPU.
            "entity_count": counts,
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
        stats["entities"] = obs["entities"].shape[2]
        return batch, dict(stats)

    @staticmethod
    def _stack(obs_steps: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Stack per-step observations into ``[T, rows, ...]``, padding the
        entity axis to the widest step."""
        width = max(step["entities"].shape[1] for step in obs_steps)
        obs = {}
        for key in obs_steps[0]:
            parts = [step[key] for step in obs_steps]
            if key in ENTITY_KEYS:
                lead = (0, 0) if parts[0].dim() == 3 else ()
                parts = [p if p.shape[1] == width else F.pad(p, lead + (0, width - p.shape[1]))
                         for p in parts]
            obs[key] = torch.stack(parts)
        return obs

    def _values(self, obs: dict[str, torch.Tensor], counts: torch.Tensor, learner: torch.Tensor, amp):
        """Critic value of every stored row the learner controlled, ``[T, rows]``
        on the CPU. Other rows keep zero; nothing reads them."""
        flat = {key: obs[key].flatten(0, 1) for key in CRITIC_KEYS}
        index = learner.flatten().nonzero().squeeze(-1)
        out = []
        for start in range(0, len(index), VALUE_ROWS):
            # Consecutive steps, so one pass sees boards of similar size. The
            # last of several passes is filled up to the common shape.
            pick = index[start : start + VALUE_ROWS]
            if len(index) > VALUE_ROWS:
                pick = padded(pick, VALUE_ROWS)
            chunk = gather_rows(flat, counts.flatten(), pick, CRITIC_KEYS, self.device)
            with amp:
                out.append(self.model.value(chunk).float())
        values = torch.zeros(learner.numel())
        values[index] = torch.cat(out)[: len(index)].cpu()
        return values.view_as(learner)


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
