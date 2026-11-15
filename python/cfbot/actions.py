"""Factorized action space.

The simulator's flat action grid has 4,611 entries:

    0                      wait
    1 + slot*576 + cell    play hand slot (primary form)
    2305 + ability         use own ability button 0 or 1
    2307 + slot*576 + cell play hand slot (secondary form)

with ``cell = y*18 + x`` in the acting player's frame. The policy chooses an
action *type* first, then a *cell* for play types:

    type 0        wait
    type 1..4     play slot 0..3, primary form
    type 5..8     play slot 0..3, secondary form
    type 9..10    ability 0..1
"""

from __future__ import annotations

import torch

WIDTH = 18
HEIGHT = 32
CELLS = WIDTH * HEIGHT
HAND_SLOTS = 4
PLAY_ACTION_COUNT = 1 + HAND_SLOTS * CELLS  # 2305
SECONDARY_PLAY_START = PLAY_ACTION_COUNT + 2  # 2307
NUM_ACTIONS = SECONDARY_PLAY_START + HAND_SLOTS * CELLS  # 4611
NUM_TYPES = 11
PLAY_TYPES = 8


def split_legal(legal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a flat legal mask ``[..., 4611]`` into type and cell masks.

    Returns ``type_mask [..., 11]`` and ``cell_mask [..., 8, 576]`` (bool).
    Waiting is always allowed so a row can never be left without an action.
    """
    legal = legal.bool()
    lead = legal.shape[:-1]
    primary = legal[..., 1:PLAY_ACTION_COUNT].reshape(*lead, HAND_SLOTS, CELLS)
    secondary = legal[..., SECONDARY_PLAY_START:].reshape(*lead, HAND_SLOTS, CELLS)
    cell_mask = torch.cat([primary, secondary], dim=-2)
    wait = torch.ones_like(legal[..., :1])
    abilities = legal[..., PLAY_ACTION_COUNT:SECONDARY_PLAY_START]
    type_mask = torch.cat([wait, cell_mask.any(-1), abilities], dim=-1)
    return type_mask, cell_mask


def to_flat(action_type: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Convert (type, cell) to the simulator's flat action index."""
    slot = (action_type - 1) % HAND_SLOTS
    primary = 1 + slot * CELLS + cell
    secondary = SECONDARY_PLAY_START + slot * CELLS + cell
    ability = PLAY_ACTION_COUNT + (action_type - 9)
    flat = torch.where(action_type <= 4, primary, secondary)
    flat = torch.where(action_type >= 9, ability, flat)
    return torch.where(action_type == 0, torch.zeros_like(flat), flat)


def from_flat(flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`to_flat`. Cell is 0 for wait and ability actions."""
    is_primary = (flat >= 1) & (flat < PLAY_ACTION_COUNT)
    is_ability = (flat >= PLAY_ACTION_COUNT) & (flat < SECONDARY_PLAY_START)
    is_secondary = flat >= SECONDARY_PLAY_START
    packed = torch.where(is_secondary, flat - SECONDARY_PLAY_START, flat - 1).clamp(min=0)
    slot = packed // CELLS
    cell = packed % CELLS
    action_type = torch.zeros_like(flat)
    action_type = torch.where(is_primary, 1 + slot, action_type)
    action_type = torch.where(is_secondary, 5 + slot, action_type)
    action_type = torch.where(is_ability, 9 + (flat - PLAY_ACTION_COUNT), action_type)
    cell = torch.where(is_primary | is_secondary, cell, torch.zeros_like(cell))
    return action_type, cell


def flip_cell(cell: torch.Tensor) -> torch.Tensor:
    """Mirror a cell index left-to-right."""
    return cell - cell % WIDTH + (WIDTH - 1 - cell % WIDTH)


def flip_legal(legal: torch.Tensor) -> torch.Tensor:
    """Mirror a flat legal mask left-to-right."""
    lead = legal.shape[:-1]

    def flip(block: torch.Tensor) -> torch.Tensor:
        return block.reshape(*lead, HAND_SLOTS, HEIGHT, WIDTH).flip(-1).reshape(*lead, -1)

    return torch.cat(
        [
            legal[..., :1],
            flip(legal[..., 1:PLAY_ACTION_COUNT]),
            legal[..., PLAY_ACTION_COUNT:SECONDARY_PLAY_START],
            flip(legal[..., SECONDARY_PLAY_START:]),
        ],
        dim=-1,
    )


def flip_observation(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Mirror an observation left-to-right. Coordinates are in [0, 1]; padded
    rows stay zero. The simulator is not exactly symmetric in x (formation
    offsets, even-footprint buildings, tie breaks), so a mirrored view is only
    ever shown to the policy while it acts; transitions are never relabelled."""
    out = dict(obs)
    for key, column, mask_key in (
        ("entities", 4, "entity_mask"),
        ("recent_plays", 4, "play_mask"),
        ("hazards", 2, "hazard_mask"),
    ):
        values = obs[key].clone()
        present = obs[mask_key].bool()
        if key == "recent_plays":
            present = present & (values[..., 3] > 0)
        values[..., column] = torch.where(present, 1 - values[..., column], values[..., column])
        out[key] = values
    # Velocity along x changes sign with the mirror.
    extras = obs["entity_extras"].clone()
    extras[..., 3] = -extras[..., 3]
    out["entity_extras"] = extras
    out["legal"] = flip_legal(obs["legal"])
    return out


def flip_rows(obs: dict[str, torch.Tensor], rows: torch.Tensor) -> dict[str, torch.Tensor]:
    """Mirror only the rows where ``rows`` is true."""
    flipped = flip_observation(obs)
    out = dict(obs)
    for key in ("entities", "recent_plays", "hazards", "entity_extras", "legal"):
        mask = rows.view(-1, *([1] * (obs[key].dim() - 1)))
        out[key] = torch.where(mask, flipped[key], obs[key])
    return out
