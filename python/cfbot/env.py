"""Torch-facing wrapper around the Rust batch environment."""

from __future__ import annotations

import numpy as np
import torch

from . import _env

# Public inputs of the policy (simulator schema 2 plus bot-owned planes).
POLICY_KEYS = (
    "entities", "entity_mask", "global", "hand", "own_deck", "recent_plays", "play_mask",
    "revealed_deck", "reveal_mask", "hazards", "hazard_mask", "extras", "legal",
)
# Hidden opponent state for the training-only critic.
PRIVATE_KEYS = ("priv_global", "priv_hand", "priv_deck")
OBS_KEYS = POLICY_KEYS + PRIVATE_KEYS
# Keys whose second axis is the per-step entity width and must be padded
# when steps are stacked.
ENTITY_KEYS = ("entities", "entity_mask")


def card_ids(spec: dict, names) -> list[int]:
    """Translate card names to vocabulary ids."""
    lookup = {name: index for index, name in enumerate(spec["card_names"]) if name}
    missing = [name for name in names if name not in lookup]
    if missing:
        raise ValueError(f"unknown cards: {missing}")
    return [lookup[name] for name in names]


class Env:
    """Batch of battles. Rows are ``env * 2 + side``; both players of battle
    ``e`` are rows ``2e`` and ``2e + 1``."""

    def __init__(
        self,
        num_envs: int,
        workers: int = 0,
        decision_ticks: int = 6,
        seed: int = 0,
        max_entities: int = 128,
        deck_pool: list[list[str]] | None = None,
        pool_prob: float = 1.0,
        mirror_decks: bool = False,
        max_skip: int = 8,
        exclude_cards=(),
        device: torch.device | str = "cpu",
    ):
        probe = _env.VecEnv(num_envs=1, workers=1)
        spec = probe.spec()
        self.env = _env.VecEnv(
            num_envs=num_envs,
            workers=workers,
            decision_ticks=decision_ticks,
            seed=seed,
            max_entities=max_entities,
            deck_pool=[card_ids(spec, deck) for deck in deck_pool] if deck_pool else None,
            pool_prob=pool_prob,
            mirror_decks=mirror_decks,
            max_skip=max_skip,
            exclude_cards=card_ids(spec, exclude_cards),
        )
        self.spec = self.env.spec()
        self.num_envs = num_envs
        self.rows = num_envs * 2
        self.device = torch.device(device)

    def _obs(self, raw: dict) -> dict[str, torch.Tensor]:
        # `last_cpu` shares memory with the fresh numpy arrays of this step, so
        # rollout storage can keep it without another copy.
        self.last_cpu = {key: torch.from_numpy(raw[key]) for key in OBS_KEYS}
        return {key: value.to(self.device) for key, value in self.last_cpu.items()}

    def reset(self, seed: int | None = None) -> dict[str, torch.Tensor]:
        return self._obs(self.env.reset(seed))

    def step(self, actions: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
        """Returns the next observation (on the device) and step info (numpy)."""
        raw = self.env.step(actions.detach().to("cpu", torch.int64).numpy())
        info = {
            key: raw[key]
            for key in (
                "reward", "trade", "leak", "ticks", "done", "truncated", "verdict",
                "final_towers", "outcome", "episode_decisions", "episode_ticks",
                "entity_overflow",
            )
        }
        return self._obs(raw), info

    def set_pool_prob(self, pool_prob: float) -> None:
        self.env.set_pool_prob(pool_prob)

    def crashes(self):
        return self.env.crashes()

    def deck_retries(self) -> int:
        return self.env.deck_retries()


def select_rows(obs: dict[str, torch.Tensor], rows: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value[rows] for key, value in obs.items()}


def tower_potential(towers: torch.Tensor) -> torch.Tensor:
    """Tower-health advantage in [-1, 1] from ``[own k, l, r, enemy k, l, r]``."""
    weights = towers.new_tensor([0.5, 0.25, 0.25, -0.5, -0.25, -0.25])
    return (towers * weights).sum(-1)


def observed_towers(obs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Tower health ``[own k, l, r, enemy k, l, r]`` read from entity rows."""
    ent = obs["entities"]
    kind, friendly, x, hp = ent[..., 0], ent[..., 1] > 0, ent[..., 4], ent[..., 6]
    lanes = [kind == 1, (kind == 2) & (x < 0.5), (kind == 2) & (x >= 0.5)]
    out = [(hp * (lane & side)).sum(-1) for side in (friendly, ~friendly) for lane in lanes]
    return torch.stack(out, dim=-1)
