"""Look-ahead at play time (arena simulator only).

The policy proposes a few moves, the bridge plays each of them out from what
the player sees (``VecEnv.search`` in training-style batches, the same code
behind ``Featurizer.search`` for the tournament client), and the policy's own
public value head scores the positions that result. The move is then drawn from

    log pi(a) + weight * Q(a)        over the candidates,

where ``Q`` is the value of the position ``horizon`` ticks after the move with
nobody playing another card in between. ``weight = 0`` is the plain policy
restricted to its candidates; a large weight plays whatever looks best.

Two ways of choosing candidates:

``scope="tile"``
    The policy chooses when to play and which card, as it always does; the
    look-ahead only chooses among that card's ``cells`` most likely tiles.
``scope="all"``
    Waiting, every playable card's ``cells`` most likely tiles and the ability
    are all compared. A wait is scored as waiting for the whole horizon, which
    is harsher than the one decision a wait really commits to, so this scope
    leans towards playing; ``wait_floor`` scores a wait at no less than the
    value of the current position.

What a rollout does not know is listed in ``arena/src/search.rs``.
"""

from __future__ import annotations

import numpy as np
import torch

from . import actions as A
from .env import POLICY_KEYS

# Rows the value head sees per call are padded to a multiple of this, so MPS
# and CUDA meet only a few batch shapes.
BUCKET = 64


class Searcher:
    """A policy with look-ahead. ``act(obs, rollouts)`` returns flat actions.

    ``rollouts(picked, candidates)`` plays the candidates out: ``picked`` are
    indices into the rows of ``obs`` and ``candidates`` an int64 array
    ``[len(picked), k]`` of flat action indices (-1 for none); it returns the
    bridge's dict of leaf planes, ``[len(picked) * k, ...]``.
    """

    def __init__(self, model, scope: str = "tile", cells: int = 4, horizon: int = 40, weight: float = 20.0,
                 temperature: float = 1.0, greedy: bool = False, wait_floor: bool = True, seed: int = 0):
        if scope not in ("tile", "all"):
            raise ValueError("scope must be 'tile' or 'all'")
        self.model = model.eval()
        self.scope, self.cells, self.horizon, self.weight = scope, cells, horizon, weight
        self.temperature, self.greedy, self.wait_floor = temperature, greedy, wait_floor
        self.generator = torch.Generator().manual_seed(seed)
        self.decisions = 0
        self.changed = 0

    # -- drawing -----------------------------------------------------------

    def _draw(self, logits: torch.Tensor) -> torch.Tensor:
        """One index per row of ``logits`` (``-inf`` where not allowed)."""
        if self.greedy:
            return logits.argmax(-1)
        uniform = torch.rand(logits.shape, generator=self.generator).clamp(1e-10, 1 - 1e-7)
        return (logits.cpu() / self.temperature - torch.log(-torch.log(uniform))).argmax(-1).to(logits.device)

    # -- candidates --------------------------------------------------------

    def _candidates(self, logp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Flat action indices ``[n, k]`` (-1 for none) and their log prior."""
        n = logp.shape[0]
        plays = logp[:, 1 : A.PLAY_ACTION_COUNT].reshape(n, A.HAND_SLOTS, A.CELLS)
        cell_logp, cell = plays.topk(self.cells, dim=-1)  # [n, 4, cells]
        flat = 1 + torch.arange(A.HAND_SLOTS, device=logp.device).view(1, -1, 1) * A.CELLS + cell
        if self.scope == "tile":
            # The policy's own choice of what to do; the look-ahead picks the tile.
            kinds = torch.cat(
                [logp[:, :1], plays.logsumexp(-1), logp[:, A.PLAY_ACTION_COUNT : A.PLAY_ACTION_COUNT + 1]], dim=1
            )
            kind = self._draw(kinds)  # 0 wait, 1-4 a slot, 5 the ability
            slot = (kind - 1).clamp(0, A.HAND_SLOTS - 1)
            rows = torch.arange(n, device=logp.device)
            index, prior = flat[rows, slot], cell_logp[rows, slot]
            fixed = torch.where(kind == 0, torch.zeros_like(kind), torch.full_like(kind, A.PLAY_ACTION_COUNT))
            play = (kind >= 1) & (kind <= A.HAND_SLOTS)
            index = torch.where(play.unsqueeze(-1), index, torch.full_like(index, -1))
            index[:, 0] = torch.where(play, index[:, 0], fixed)
            prior = torch.where(play.unsqueeze(-1), prior, torch.full_like(prior, float("-inf")))
            prior[:, 0] = torch.where(play, prior[:, 0], torch.zeros_like(prior[:, 0]))
        else:
            index = torch.cat(
                [torch.zeros(n, 1, dtype=flat.dtype, device=logp.device), flat.reshape(n, -1),
                 torch.full((n, 1), A.PLAY_ACTION_COUNT, dtype=flat.dtype, device=logp.device)], dim=1)
            prior = torch.cat(
                [logp[:, :1], cell_logp.reshape(n, -1), logp[:, A.PLAY_ACTION_COUNT : A.PLAY_ACTION_COUNT + 1]], dim=1)
        index = torch.where(torch.isfinite(prior), index, torch.full_like(index, -1))
        return index, prior

    # -- values ------------------------------------------------------------

    def _values(self, leaves: dict) -> torch.Tensor:
        device = next(self.model.parameters()).device
        count = len(leaves["ok"])
        padded = count if device.type == "cpu" else -(-count // BUCKET) * BUCKET
        obs = {}
        for key in POLICY_KEYS:
            plane = torch.from_numpy(leaves[key])
            if padded > count:
                plane = torch.cat([plane, plane[:1].expand(padded - count, *plane.shape[1:])])
            obs[key] = plane.to(device)
        enc = self.model.encode(obs)
        value = self.model.expected(self.model.value_public(enc["trunk"]).float())[:count]
        ended = torch.from_numpy(leaves["ended"]).to(device)
        result = torch.from_numpy(leaves["result"]).to(device)
        value = torch.where(ended, result, value)
        return torch.where(torch.from_numpy(leaves["ok"]).to(device), value, torch.full_like(value, float("-inf")))

    # -- acting ------------------------------------------------------------

    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], rollouts) -> torch.Tensor:
        public = {key: obs[key] for key in POLICY_KEYS}
        logp, root = self.model.full_log_probs(public)
        index, prior = self._candidates(logp)
        first = index[:, 0].clamp(min=0)
        # Rows with something to compare: more than one candidate.
        picked = torch.nonzero((index >= 0).sum(-1) > 1).squeeze(-1)
        if picked.numel() == 0:
            return first
        cand = index[picked]
        leaves = rollouts(picked.cpu().numpy(), cand.cpu().numpy().astype(np.int64))
        q = self._values(leaves).view(cand.shape)
        if self.scope == "all" and self.wait_floor:
            q[:, 0] = torch.maximum(q[:, 0], root[picked])
        q = torch.where(cand >= 0, q, torch.full_like(q, float("-inf")))
        score = prior[picked] + self.weight * q
        # A candidate the rollout could not play keeps the policy from choosing it at all.
        score = torch.where(torch.isfinite(score), score, torch.full_like(score, float("-inf")))
        dead = ~torch.isfinite(score).any(-1)
        choice = self._draw(score)
        chosen = cand.gather(1, choice.unsqueeze(-1)).squeeze(-1)
        chosen = torch.where(dead, first[picked], chosen)
        self.decisions += int(picked.numel())
        self.changed += int((chosen != self._plain(prior[picked], cand)).sum())
        out = first.clone()
        out[picked] = chosen
        return out

    def _plain(self, prior: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """The policy's most likely candidate, to count how often look-ahead disagrees."""
        return cand.gather(1, prior.argmax(-1, keepdim=True)).squeeze(-1)
