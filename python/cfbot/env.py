"""Torch-facing wrapper around the Rust batch environment."""

from __future__ import annotations

import numpy as np
import torch

from . import _env

# Public inputs of the policy (simulator schema 2 plus bot-owned planes).
POLICY_KEYS = (
    "entities", "entity_extras", "entity_mask", "global", "abilities", "hand", "own_deck",
    "deck_extras", "recent_plays", "play_mask", "revealed_deck", "reveal_extras", "reveal_mask",
    "hazards", "hazard_mask", "extras", "legal",
)
# Hidden opponent state for the training-only critic.
PRIVATE_KEYS = ("priv_global", "priv_hand", "priv_deck")
OBS_KEYS = POLICY_KEYS + PRIVATE_KEYS
# What the critic reads: everything but the legal-action mask.
CRITIC_KEYS = tuple(key for key in OBS_KEYS if key != "legal")
# Keys whose second axis is the per-step entity width and must be padded
# when steps are stacked.
ENTITY_KEYS = ("entities", "entity_extras", "entity_mask")
# Step outputs that describe the transition rather than the next observation.
INFO_KEYS = (
    "reward", "trade", "leak", "ticks", "done", "truncated", "verdict", "final_towers", "outcome",
    "episode_decisions", "episode_ticks", "entity_overflow", "hazard_overflow", "decks", "queued",
)


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
        rules: str | None = None,
        level_jitter: tuple[int, int] | None = None,
        device: torch.device | str = "cpu",
    ):
        probe = _env.VecEnv(num_envs=1, workers=1, rules=rules)
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
            rules=rules,
            level_jitter=tuple(level_jitter) if level_jitter else None,
        )
        self.spec = self.env.spec()
        self.num_envs = num_envs
        self.rows = num_envs * 2
        self.device = torch.device(device)

    def _obs(self, raw: dict) -> dict[str, torch.Tensor]:
        # `last_cpu` shares memory with the fresh numpy arrays of this step, so
        # rollout storage can keep it without another copy.
        self.last_cpu = {key: torch.from_numpy(raw[key]) for key in OBS_KEYS}
        # Per row: whether anything but waiting is legal, and how many entity
        # rows are valid. Both follow from the planes; the bridge has them for
        # free.
        if "choice" in raw:
            self.last_choice = torch.from_numpy(raw["choice"])
            self.last_entity_count = torch.from_numpy(raw["entity_count"])
        else:  # a bridge built before it exported them
            self.last_choice = self.last_cpu["legal"][:, 1:].bool().any(-1)
            self.last_entity_count = self.last_cpu["entity_mask"].sum(-1)
        return self.upload(self.last_cpu, packed=True)

    def upload(self, obs: dict[str, torch.Tensor], packed: bool = False) -> dict[str, torch.Tensor]:
        """Copy CPU planes to the device.

        ``packed`` sends them as two blocks, one per dtype, and returns views
        into those: a transfer costs about the same whatever its size, and one
        step's observation is twenty small planes. The views are fine to
        gather rows from; a caller that will reshape whole planes should ask
        for separate copies. On CUDA the copies are queued without waiting for
        each one; the first use waits for them."""
        device = self.device
        if device.type == "cpu":
            return dict(obs)
        lazy = device.type == "cuda"
        if not packed:
            return {key: value.to(device, non_blocking=lazy) for key, value in obs.items()}
        out = {}
        lead = next(iter(obs.values())).shape[0]
        for dtype in {value.dtype for value in obs.values()}:
            keys = [key for key, value in obs.items() if value.dtype == dtype]
            block = torch.cat([obs[key].reshape(lead, -1) for key in keys], dim=1)
            block = block.to(device, non_blocking=lazy)
            start = 0
            for key in keys:
                size = obs[key][0].numel()
                out[key] = block[:, start : start + size].view(obs[key].shape)
                start += size
        return out

    def reset(self, seed: int | None = None) -> dict[str, torch.Tensor]:
        return self._obs(self.env.reset(seed))

    def step(self, actions: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
        """Returns the next observation (on the device) and step info (numpy)."""
        raw = self.env.step(actions.detach().to("cpu", torch.int64).numpy())
        return self._obs(raw), {key: raw[key] for key in INFO_KEYS}

    def set_mix(self, pool_prob: float, mutant_prob: float = 0.0) -> None:
        """Deck sources for new episodes: chance of a pool deck, chance of a
        mutated pool deck, random otherwise."""
        self.env.set_mix(pool_prob, mutant_prob)

    def set_pool(self, decks: list[list[str]], weights: list[float] | None = None) -> None:
        self.env.set_pool([card_ids(self.spec, deck) for deck in decks], weights)

    def queue_match(self, env: int, blue: list[int], red: list[int], seed: int | None = None) -> None:
        """Make battle ``env``'s next episode this exact match (decks as card
        tokens). The step output flags its result with ``queued``."""
        self.env.queue_match(env, list(blue), list(red), seed)

    def random_decks(self, count: int, seed: int) -> list[list[int]]:
        """Random decks under the sampler's playability rules, as card tokens."""
        return self.env.random_decks(count, seed)

    def crashes(self):
        return self.env.crashes()

    def deck_retries(self) -> int:
        return self.env.deck_retries()


def select_rows(obs: dict[str, torch.Tensor], rows: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value[rows] for key, value in obs.items()}


def entity_width(longest: int, limit: int) -> int:
    """Entity columns needed for rows of up to ``longest`` entities: the next
    multiple of 16, at most ``limit``."""
    return min(limit, max(16, -(-longest // 16) * 16))


def gather_rows(
    obs: dict[str, torch.Tensor], counts: torch.Tensor, pick: torch.Tensor, keys, device: torch.device
) -> dict[str, torch.Tensor]:
    """Rows ``pick`` of a stored rollout, on ``device``, with the entity axis
    cut to the longest of those rows in steps of 16.

    ``obs`` holds the rollout flattened to one row per decision, on the CPU or
    already on the device. ``counts`` holds every row's entity count on the
    CPU, so finding the width never waits for the device."""
    home = obs["entity_mask"].device
    width = entity_width(int(counts[pick].max()), obs["entity_mask"].shape[1])
    where = pick.to(home)
    rows = {}
    for key in keys:
        value = obs[key]
        if key in ENTITY_KEYS:
            value = value[:, :width]
        rows[key] = value.index_select(0, where).to(device)
    return rows


def trim_entities(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Cut the entity axis to the longest row present, in steps of 16."""
    mask = obs["entity_mask"]
    longest = int(mask.sum(-1).max()) if mask.numel() else 0
    width = entity_width(longest, mask.shape[-1])
    if width == mask.shape[-1]:
        return obs
    out = dict(obs)
    for key in ENTITY_KEYS:
        out[key] = obs[key][:, :width]
    return out


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
