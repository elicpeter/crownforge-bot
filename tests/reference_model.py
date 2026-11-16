"""Frozen copy of the network as it was before the throughput work
(commit a09ef80 plus the board_kernel and value_target knobs). The tests load
the same weights into this and into the current network and compare outputs
and gradients; do not edit it to make a test pass."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cfbot import actions as A
from cfbot.config import ModelConfig

BOARD_H, BOARD_W = A.HEIGHT // 2, A.WIDTH // 2
BOARD_CELLS = BOARD_H * BOARD_W
N_QUERIES = 1 + 4 + 2  # global, hand cards, ability buttons
# Logit for illegal actions. Finite on purpose: with -inf, the entropy term
# 0 * -inf produces NaN gradients.
ILLEGAL = -1e9


def fourier(xy: torch.Tensor, bands: int) -> torch.Tensor:
    """Sinusoidal features of coordinates in [-1, 1]."""
    freqs = (2.0 ** torch.arange(bands, device=xy.device, dtype=xy.dtype)) * (math.pi / 2)
    angles = xy.unsqueeze(-1) * freqs
    return torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)


def mlp(inp: int, hidden: int, out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inp, hidden), nn.GELU(), nn.Linear(hidden, out))


class Attention(nn.Module):
    """Multi-head attention from queries to a masked memory."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, queries, memory, mask=None):
        n, q, d = queries.shape
        s = memory.shape[1]
        head = d // self.heads
        qs = self.q(queries).view(n, q, self.heads, head).transpose(1, 2)
        k, v = self.kv(memory).view(n, s, 2, self.heads, head).permute(2, 0, 3, 1, 4)
        attn_mask = None if mask is None else mask[:, None, None, :]
        out = F.scaled_dot_product_attention(qs, k, v, attn_mask=attn_mask)
        return self.proj(out.transpose(1, 2).reshape(n, q, d))


class SelfBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ratio: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = mlp(dim, dim * ratio, dim)

    def forward(self, x, mask):
        y = self.norm1(x)
        x = x + self.attn(y, y, mask)
        return x + self.mlp(self.norm2(x))


class QueryBlock(nn.Module):
    """Queries read the memory, talk to each other, then update."""

    def __init__(self, dim: int, heads: int, ratio: int):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.cross = Attention(dim, heads)
        self.norm_s = nn.LayerNorm(dim)
        self.self_attn = Attention(dim, heads)
        self.norm_m = nn.LayerNorm(dim)
        self.mlp = mlp(dim, dim * ratio, dim)

    def forward(self, queries, memory, mask):
        queries = queries + self.cross(self.norm_q(queries), memory, mask)
        y = self.norm_s(queries)
        queries = queries + self.self_attn(y, y)
        return queries + self.mlp(self.norm_m(queries))


class NextBlock(nn.Module):
    """ConvNeXt-style block on a channels-last board ``[N, H, W, C]``. Built
    from a depthwise convolution and linear layers only, because dense
    convolution gradients are very slow on Apple GPUs."""

    def __init__(self, channels: int, ratio: int = 3, kernel: int = 3):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel, padding=kernel // 2, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.fc1 = nn.Linear(channels, channels * ratio)
        self.fc2 = nn.Linear(channels * ratio, channels)
        self.scale = nn.Parameter(torch.full((channels,), 0.1))

    def forward(self, x):
        y = self.depthwise(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return x + self.scale * self.fc2(F.gelu(self.fc1(self.norm(y))))


class Embedder(nn.Module):
    """Turns raw schema-2 planes into token embeddings. The actor and the
    critic each own one, so they share no weights."""

    GLOBAL_FEATURES = 15
    ENTITY_FEATURES = 26

    def __init__(self, spec: dict, dim: int, d_card: int, bands: int):
        super().__init__()
        self.bands = bands
        four = 4 * bands
        self.base_level = float(spec["base_level"])
        self.ticks_per_second = 1000.0 / spec["tick_ms"]
        self.regular = float(spec["regular_ticks"])
        self.total = float(spec["regular_ticks"] + spec["overtime_ticks"])
        self.double = float(spec["double_elixir_tick"])
        self.triple = float(spec["triple_elixir_tick"])
        self.max_elixir = float(spec["max_elixir"])

        self.register_buffer("card_table", torch.as_tensor(spec["card_table"]).float().clone())
        self.register_buffer("unit_table", torch.as_tensor(spec["unit_table"]).float().clone())
        self.card_emb = nn.Embedding(spec["n_cards"], d_card, padding_idx=0)
        self.card_stats = nn.Linear(spec["card_f"], d_card)
        self.unit_emb = nn.Embedding(spec["n_units"], dim, padding_idx=0)
        self.unit_stats = nn.Linear(spec["unit_f"], dim)
        self.kind_emb = nn.Embedding(5, dim, padding_idx=0)
        self.tower_emb = nn.Embedding(5, dim, padding_idx=0)
        self.side_emb = nn.Embedding(2, dim)
        self.hazard_emb = nn.Embedding(spec["n_hazards"], dim, padding_idx=0)
        self.play_rank = nn.Parameter(torch.randn(spec["play_rows"], dim) * 0.02)
        self.type_emb = nn.Parameter(torch.randn(6, dim) * 0.02)
        self.slot_emb = nn.Parameter(torch.randn(4, dim) * 0.02)
        self.in_hand_emb = nn.Parameter(torch.randn(2, dim) * 0.02)

        self.entity_feat = mlp(self.ENTITY_FEATURES + four, dim, dim)
        self.hazard_feat = nn.Linear(3 + four, dim)
        self.global_feat = mlp(self.GLOBAL_FEATURES, dim, dim)
        self.hand_card = nn.Linear(d_card * 2, dim)
        self.hand_feat = nn.Linear(4, dim)
        self.deck_card = nn.Linear(d_card, dim)
        self.deck_feat = nn.Linear(spec["deck_x_f"], dim)
        self.ability_feat = nn.Linear(spec["ability_f"] - 1, dim)
        self.reveal_card = nn.Linear(d_card, dim)
        self.reveal_feat = nn.Linear(2 + spec["reveal_x_f"], dim)
        self.play_card = nn.Linear(d_card, dim)
        self.play_feat = nn.Linear(4 + four, dim)
        self.norms = nn.ModuleDict({name: nn.LayerNorm(dim) for name in
                                    ("entity", "hazard", "glob", "hand", "deck", "reveal", "play",
                                     "ability")})

    def cards(self, tokens: torch.Tensor) -> torch.Tensor:
        ids = tokens.long()
        return self.card_emb(ids) + self.card_stats(self.card_table[ids])

    def now_tick(self, obs) -> torch.Tensor:
        return obs["global"][:, 0] * self.ticks_per_second

    def forward(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        glob = obs["global"]
        tick = self.now_tick(obs)
        elixir = glob[:, 1]
        extras = obs["extras"]
        global_features = torch.stack(
            [
                tick / self.regular,
                tick / self.total,
                (tick < self.double).float(),
                ((tick >= self.double) & (tick < self.triple)).float(),
                (tick >= self.triple).float(),
                (tick >= self.regular).float(),
                elixir / self.max_elixir,
                (elixir >= self.max_elixir - 1e-3).float(),
                glob[:, 2] / 2.0,
                extras[:, 0] / self.max_elixir,
                extras[:, 1],
                extras[:, 2],
                extras[:, 3],
                extras[:, 4],
                extras[:, 5],
            ],
            dim=-1,
        )
        out = {"glob": self.norms["glob"](self.global_feat(global_features))}

        ent = obs["entities"]
        xy = ent[..., 4:6] * 2 - 1
        feats = torch.cat(
            [
                ent[..., 1:2],
                (ent[..., 3:4] - self.base_level) / 5.0,
                xy,
                ent[..., 6:8],
                ent[..., 9:10] / 10.0,
                ent[..., 10:19],
                ent[..., 19:20] / 4.0,
                ent[..., 20:24],
                obs["entity_extras"],
                fourier(xy, self.bands),
            ],
            dim=-1,
        )
        unit = ent[..., 2].long()
        out["entity"] = self.norms["entity"](
            self.unit_emb(unit)
            + self.unit_stats(self.unit_table[unit])
            + self.kind_emb(ent[..., 0].long())
            + self.tower_emb(ent[..., 8].long())
            + self.side_emb(ent[..., 1].long())
            + self.entity_feat(feats)
            + self.type_emb[0]
        )
        out["entity_mask"] = obs["entity_mask"].bool()
        out["entity_xy"] = ent[..., 4:6]
        out["entity_side"] = ent[..., 1]

        ability = obs["abilities"]
        button = ability[..., 0].long()
        out["ability"] = self.norms["ability"](
            self.unit_emb(button)
            + self.unit_stats(self.unit_table[button])
            + self.ability_feat(ability[..., 1:])
        )

        haz = obs["hazards"]
        hxy = haz[..., 2:4] * 2 - 1
        out["hazard"] = self.norms["hazard"](
            self.hazard_emb(haz[..., 0].long())
            + self.side_emb(haz[..., 1].long())
            + self.hazard_feat(torch.cat([hxy, haz[..., 4:5] * 4, fourier(hxy, self.bands)], dim=-1))
            + self.type_emb[1]
        )
        out["hazard_mask"] = obs["hazard_mask"].bool()
        out["hazard_xy"] = haz[..., 2:4]

        hand = obs["hand"]
        hand_features = torch.stack(
            [
                (hand[..., 0] > 0).float(),
                (hand[..., 2] - self.base_level) / 5.0 * (hand[..., 1] > 0),
                hand[..., 3] / 10.0,
                ((hand[..., 1] > 0) & (elixir.unsqueeze(-1) >= hand[..., 3])).float(),
            ],
            dim=-1,
        )
        out["hand"] = self.norms["hand"](
            self.hand_card(torch.cat([self.cards(hand[..., 0]), self.cards(hand[..., 1])], dim=-1))
            + self.hand_feat(hand_features)
            + self.slot_emb
        )

        deck = obs["own_deck"]
        in_hand = (deck[..., 0].unsqueeze(-1) == hand[..., 0].unsqueeze(-2)).any(-1)
        out["deck"] = self.norms["deck"](
            self.deck_card(self.cards(deck[..., 0]))
            + self.deck_feat(obs["deck_extras"])
            + self.in_hand_emb[in_hand.long()]
            + self.type_emb[2]
        )

        reveal = obs["revealed_deck"]
        age = ((tick.unsqueeze(-1) - reveal[..., 1]) / 600.0).clamp(0, 3)
        out["reveal"] = self.norms["reveal"](
            self.reveal_card(self.cards(reveal[..., 0]))
            + self.reveal_feat(
                torch.cat(
                    [torch.stack([age, reveal[..., 2] / 10.0], dim=-1), obs["reveal_extras"]], dim=-1
                )
            )
            + self.type_emb[3]
        )
        out["reveal_mask"] = obs["reveal_mask"].bool()

        play = obs["recent_plays"]
        pxy = (play[..., 4:6] * 2 - 1) * play[..., 3:4]
        age = ((tick.unsqueeze(-1) - play[..., 6]) / 200.0).clamp(0, 3)
        play_features = torch.cat(
            [play[..., 3:4], pxy, age.unsqueeze(-1), fourier(pxy, self.bands)], dim=-1
        )
        # Plays are stored oldest first; rank 0 is the newest one.
        play_mask = obs["play_mask"].bool()
        slots = torch.arange(play.shape[1], device=play.device)
        rank = (play_mask.sum(-1, keepdim=True) - 1 - slots).clamp(0, play.shape[1] - 1)
        out["play"] = self.norms["play"](
            self.play_card(self.cards(play[..., 1]))
            + self.side_emb(play[..., 0].long())
            + self.play_feat(play_features)
            + self.play_rank[rank]
            + self.type_emb[4]
        )
        out["play_mask"] = play_mask
        return out


def pooled(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean and max over a token axis."""
    weight = mask.unsqueeze(-1).to(tokens.dtype)
    mean = (tokens * weight).sum(1) / weight.sum(1).clamp(min=1)
    peak = tokens.masked_fill(~mask.unsqueeze(-1), -1e4).amax(1)
    return torch.cat([mean, peak * mask.any(1, keepdim=True)], dim=-1)


class Critic(nn.Module):
    """Outcome prediction from the full state, including the opponent's hand."""

    def __init__(self, spec: dict, config: ModelConfig):
        super().__init__()
        d, dc, hidden = config.d_critic, config.d_card, config.critic_hidden
        self.embed = Embedder(spec, d, dc, config.fourier_bands)
        self.block = SelfBlock(d, max(1, d // 32), config.mlp_ratio)
        self.private = mlp(dc * 5 + spec["priv_global_f"], d, d)
        self.max_elixir = float(spec["max_elixir"])
        self.head = nn.Sequential(
            nn.Linear(d * 9, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, config.value_bins),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        tok = self.embed(obs)
        entities = self.block(tok["entity"], tok["entity_mask"])
        priv = obs["priv_global"]
        priv_features = torch.stack(
            [priv[:, 0] / self.max_elixir, priv[:, 1] / 100.0, priv[:, 2] / 100.0], dim=-1
        )
        private = self.private(
            torch.cat(
                [
                    self.embed.cards(obs["priv_hand"]).flatten(-2),
                    self.embed.cards(obs["priv_deck"]).mean(1),
                    priv_features,
                ],
                dim=-1,
            )
        )
        features = torch.cat(
            [
                tok["glob"],
                pooled(entities, tok["entity_mask"]),
                pooled(tok["hazard"], tok["hazard_mask"]),
                tok["hand"].mean(1),
                tok["deck"].mean(1),
                pooled(tok["play"], tok["play_mask"])[:, : tok["play"].shape[-1]],
                private,
            ],
            dim=-1,
        )
        return self.head(features).float()


class Policy(nn.Module):
    def __init__(self, spec: dict, config: ModelConfig):
        super().__init__()
        if config.board_kernel % 2 != 1:
            raise ValueError("board_kernel must be odd")
        if config.value_target not in ("two_hot", "hl_gauss"):
            raise ValueError("value_target must be 'two_hot' or 'hl_gauss'")
        self.config = config
        d, c, dt = config.d_model, config.d_board, config.d_trunk
        self.embed = Embedder(spec, d, config.d_card, config.fourier_bands)
        self.entity_blocks = nn.ModuleList(
            [SelfBlock(d, config.n_heads, config.mlp_ratio) for _ in range(config.entity_layers)]
        )

        # Board image.
        self.to_board = nn.Linear(d, c)
        self.board_base = nn.Parameter(torch.randn(1, BOARD_H, BOARD_W, c) * 0.02)
        self.board_global = nn.Linear(d, c)
        self.board_norm = nn.LayerNorm(c)
        # Units per cell and side, added after the norm, which would otherwise
        # hide how many units share a cell.
        self.board_count = nn.Linear(2, c)
        self.board_blocks = nn.Sequential(
            *[NextBlock(c, kernel=config.board_kernel) for _ in range(config.board_blocks)]
        )
        self.board_out = nn.LayerNorm(c)
        self.cell_token = nn.Linear(c, d)
        self.cell_pos = nn.Parameter(torch.randn(BOARD_CELLS, d) * 0.02)

        # Queries.
        self.ability_query = nn.Parameter(torch.randn(2, d) * 0.02)
        self.query_blocks = nn.ModuleList(
            [QueryBlock(d, config.n_heads, config.mlp_ratio) for _ in range(config.query_layers)]
        )
        self.query_norm = nn.LayerNorm(d)

        # Action heads.
        self.option_emb = nn.Parameter(torch.randn(2, d) * 0.02)
        self.type_head = mlp(d * 2, d, 1)
        self.type_bias = nn.Parameter(torch.zeros(A.NUM_TYPES))
        with torch.no_grad():
            self.type_bias[0] = config.wait_bias
        self.place_cond = mlp(d * 2, d, c * 2)
        self.place_block = NextBlock(c)
        self.place_out = nn.Linear(c, 4)
        # Low-rank per-cell term: an exact-tile prior for each card.
        self.cell_table = nn.Parameter(torch.randn(A.CELLS, 32) * 0.02)
        self.cell_query = nn.Linear(d * 2, 32)

        # Trunk for the public value and auxiliary heads.
        self.trunk = nn.Sequential(nn.Linear(d * 2 + c * 2, dt), nn.LayerNorm(dt), nn.GELU())
        bins = config.value_bins
        self.register_buffer("support", torch.linspace(-config.value_range, config.value_range, bins))
        self.value_public = mlp(dt, dt, bins)
        self.aux_hand = nn.Linear(dt, spec["n_cards"])
        self.aux_elixir = nn.Linear(dt, 1)
        self.max_elixir = float(spec["max_elixir"])

        self.critic = Critic(spec, config)

        for layer in (self.type_head[-1], self.place_out, self.cell_query):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.value_public[-1].weight)
        nn.init.zeros_(self.value_public[-1].bias)

    def actor_parameters(self):
        return [p for name, p in self.named_parameters() if not name.startswith("critic.")]

    # ---------------------------------------------------------------- encoder

    @staticmethod
    def _cell_index(xy: torch.Tensor) -> torch.Tensor:
        col = (xy[..., 0] * BOARD_W).long().clamp(0, BOARD_W - 1)
        row = (xy[..., 1] * BOARD_H).long().clamp(0, BOARD_H - 1)
        return row * BOARD_W + col

    def encode(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tok = self.embed(obs)
        entities, entity_mask = tok["entity"], tok["entity_mask"]
        for block in self.entity_blocks:
            entities = block(entities, entity_mask)
        n = entities.shape[0]

        # Scatter units and hazards onto the board.
        things = torch.cat([entities, tok["hazard"]], dim=1)
        thing_mask = torch.cat([entity_mask, tok["hazard_mask"]], dim=1)
        cells = self.to_board(things) * thing_mask.unsqueeze(-1).to(things.dtype)
        entity_cell = self._cell_index(tok["entity_xy"])
        index = torch.cat([entity_cell, self._cell_index(tok["hazard_xy"])], dim=1)
        index = index.unsqueeze(-1).expand(-1, -1, cells.shape[-1])
        board = torch.zeros(n, BOARD_CELLS, cells.shape[-1], dtype=cells.dtype, device=cells.device)
        board = board.scatter_add(1, index, cells).view(n, BOARD_H, BOARD_W, -1)
        board = board + self.board_base.to(board.dtype) + self.board_global(tok["glob"])[:, None, None, :]
        side = tok["entity_side"].to(cells.dtype) * entity_mask.to(cells.dtype)
        both = torch.stack([side, entity_mask.to(cells.dtype) - side], dim=-1)
        counts = torch.zeros(n, BOARD_CELLS, 2, dtype=cells.dtype, device=cells.device)
        counts = counts.scatter_add(1, entity_cell.unsqueeze(-1).expand(-1, -1, 2), both)
        board = self.board_norm(board) + self.board_count(torch.log1p(counts)).view(
            n, BOARD_H, BOARD_W, -1
        )
        board = self.board_out(self.board_blocks(board))

        parts = [entities, tok["hazard"], tok["deck"], tok["reveal"], tok["play"]]
        ones = torch.ones(n, 8, dtype=torch.bool, device=entities.device)
        masks = [entity_mask, tok["hazard_mask"], ones, tok["reveal_mask"], tok["play_mask"]]
        if self.config.board_in_memory:
            parts.append(self.cell_token(board.view(n, BOARD_CELLS, -1)) + self.cell_pos)
            masks.append(torch.ones(n, BOARD_CELLS, dtype=torch.bool, device=entities.device))
        memory = torch.cat(parts, dim=1)
        mask = torch.cat(masks, dim=1)
        queries = torch.cat(
            [tok["glob"].unsqueeze(1), tok["hand"], self.ability_query + tok["ability"]], dim=1
        )
        for block in self.query_blocks:
            queries = block(queries, memory, mask)
        queries = self.query_norm(queries)
        glob, hand, abil = queries[:, 0], queries[:, 1:5], queries[:, 5:7]
        summary = torch.cat([glob, hand.mean(1), board.mean((1, 2)), board.amax((1, 2))], dim=-1)
        # The public value and auxiliary heads read the trunk. Scale the gradient
        # they send back so they inform the encoder without steering it.
        scale = self.config.aux_grad_scale
        trunk = self.trunk(summary.detach() + scale * (summary - summary.detach()))
        # Candidates: wait, four primary plays, four secondary plays, two abilities.
        cand = torch.cat(
            [glob.unsqueeze(1), hand + self.option_emb[0], hand + self.option_emb[1], abil], dim=1
        )
        return {"glob": glob, "cand": cand, "board": board, "trunk": trunk}

    # ------------------------------------------------------------------ heads

    def _type_logits(self, enc, type_mask):
        cand = enc["cand"]
        context = enc["glob"].unsqueeze(1).expand(-1, cand.shape[1], -1)
        logits = self.type_head(torch.cat([cand, context], dim=-1)).squeeze(-1) + self.type_bias
        return logits.float().masked_fill(~type_mask, ILLEGAL)

    def _cell_logits(self, glob, chosen, board, cell_mask):
        """Placement logits ``[N, 576]`` given each row's chosen candidate."""
        cond = torch.cat([chosen, glob], dim=-1)
        gamma, beta = self.place_cond(cond).chunk(2, dim=-1)
        x = board * (1 + gamma[:, None, None, :]) + beta[:, None, None, :]
        fine = self.place_out(F.gelu(self.place_block(x)))
        n = fine.shape[0]
        # Each coarse cell predicts its four tiles (pixel shuffle).
        fine = fine.view(n, BOARD_H, BOARD_W, 2, 2).permute(0, 1, 3, 2, 4).reshape(n, A.CELLS)
        prior = self.cell_query(cond) @ self.cell_table.t()
        return (fine + prior).float().masked_fill(~cell_mask, ILLEGAL)

    def expected(self, logits: torch.Tensor) -> torch.Tensor:
        return (logits.softmax(-1) * self.support).sum(-1)

    def two_hot(self, target: torch.Tensor) -> torch.Tensor:
        """Two-hot encoding of scalar targets over the value support."""
        support = self.support
        target = target.clamp(support[0], support[-1])
        step = support[1] - support[0]
        position = (target - support[0]) / step
        low = position.floor().long().clamp(0, len(support) - 2)
        frac = (position - low).clamp(0, 1)
        out = torch.zeros(*target.shape, len(support), device=target.device)
        out.scatter_(-1, low.unsqueeze(-1), (1 - frac).unsqueeze(-1))
        out.scatter_add_(-1, (low + 1).unsqueeze(-1), frac.unsqueeze(-1))
        return out

    def hl_gauss(self, target: torch.Tensor) -> torch.Tensor:
        """Gaussian histogram encoding: the mass a normal distribution centred
        on the target puts in each bin, renormalized over the support."""
        support = self.support
        step = support[1] - support[0]
        target = target.clamp(support[0], support[-1])
        edges = torch.cat([support - step / 2, support[-1:] + step / 2])
        sigma = self.config.hl_gauss_sigma * step
        cdf = torch.erf((edges - target.unsqueeze(-1)) / (math.sqrt(2.0) * sigma))
        return (cdf[..., 1:] - cdf[..., :-1]) / (cdf[..., -1:] - cdf[..., :1]).clamp(min=1e-6)

    def value_target(self, target: torch.Tensor) -> torch.Tensor:
        """Scalar returns as a distribution over the value support."""
        if self.config.value_target == "hl_gauss":
            return self.hl_gauss(target)
        return self.two_hot(target)

    # ----------------------------------------------------------------- acting

    @torch.no_grad()
    def act(
        self,
        obs: dict[str, torch.Tensor],
        deterministic: bool = False,
        generator: torch.Generator | None = None,
        temperature: float = 1.0,
        with_value: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Choose one action per row.

        ``deterministic`` takes the most likely type and cell. A ``generator``
        makes sampling reproducible: noise is drawn on the CPU from it.
        ``with_value`` also runs the critic, which needs the ``priv_*`` planes.
        """
        enc = self.encode(obs)
        type_mask, cell_mask = A.split_legal(obs["legal"])
        rows = torch.arange(type_mask.shape[0], device=type_mask.device)

        def pick(logits):
            if deterministic:
                return logits.argmax(-1)
            if generator is not None:
                uniform = torch.rand(logits.shape, generator=generator).to(logits.device)
            else:
                uniform = torch.rand(logits.shape, device=logits.device)
            gumbel = -torch.log(-torch.log(uniform.clamp(1e-10, 1 - 1e-7)))
            return (logits / temperature + gumbel).argmax(-1)

        type_logp = self._type_logits(enc, type_mask).log_softmax(-1)
        action_type = pick(type_logp)
        is_play = (action_type >= 1) & (action_type <= A.PLAY_TYPES)
        play_index = (action_type - 1).clamp(0, A.PLAY_TYPES - 1)
        mask = cell_mask[rows, play_index] | ~is_play.unsqueeze(-1)
        cell_logp = self._cell_logits(
            enc["glob"], enc["cand"][rows, action_type], enc["board"], mask
        ).log_softmax(-1)
        cell = torch.where(is_play, pick(cell_logp), torch.zeros_like(action_type))
        logp = type_logp[rows, action_type] + torch.where(
            is_play, cell_logp[rows, cell], torch.zeros_like(cell_logp[:, 0])
        )
        out = {"type": action_type, "cell": cell, "flat": A.to_flat(action_type, cell), "logp": logp}
        if with_value:
            out["value"] = self.expected(self.critic(obs))
        return out

    @torch.no_grad()
    def value(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.expected(self.critic(obs))

    @torch.no_grad()
    def full_log_probs(self, obs: dict[str, torch.Tensor]):
        """Joint log-probability of every flat action, ``[N, 4611]``, with
        ``-inf`` where illegal, and the public value estimate. This is the
        deployment interface: it needs only public inputs."""
        enc = self.encode(obs)
        type_mask, cell_mask = A.split_legal(obs["legal"])
        n = type_mask.shape[0]
        type_logp = self._type_logits(enc, type_mask).log_softmax(-1)

        plays = A.PLAY_TYPES
        glob = enc["glob"].repeat_interleave(plays, dim=0)
        board = enc["board"].repeat_interleave(plays, dim=0)
        chosen = enc["cand"][:, 1 : 1 + plays].reshape(n * plays, -1)
        mask = cell_mask.reshape(n * plays, -1)
        open_mask = mask | ~mask.any(-1, keepdim=True)
        cell_logp = self._cell_logits(glob, chosen, board, open_mask).log_softmax(-1)
        cell_logp = cell_logp.masked_fill(~mask, float("-inf")).view(n, plays, A.CELLS)
        joint = type_logp[:, 1 : 1 + plays].unsqueeze(-1) + cell_logp
        half = A.HAND_SLOTS
        flat = torch.cat(
            [
                type_logp[:, :1],
                joint[:, :half].reshape(n, -1),
                type_logp[:, 9:11],
                joint[:, half:].reshape(n, -1),
            ],
            dim=1,
        )
        flat = flat.masked_fill(flat < ILLEGAL / 2, float("-inf"))
        return flat, self.expected(self.value_public(enc["trunk"]).float())

    # --------------------------------------------------------------- training

    def evaluate(
        self, obs: dict[str, torch.Tensor], action_type: torch.Tensor, cell: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Score stored actions for a flat batch of decisions."""
        enc = self.encode(obs)
        type_mask, cell_mask = A.split_legal(obs["legal"])
        rows = torch.arange(type_mask.shape[0], device=type_mask.device)

        type_logp = self._type_logits(enc, type_mask).log_softmax(-1)
        type_entropy = -(type_logp.exp() * type_logp).sum(-1)
        is_play = (action_type >= 1) & (action_type <= A.PLAY_TYPES)
        play_index = (action_type - 1).clamp(0, A.PLAY_TYPES - 1)
        mask = cell_mask[rows, play_index] | ~is_play.unsqueeze(-1)
        cell_logp = self._cell_logits(
            enc["glob"], enc["cand"][rows, action_type], enc["board"], mask
        ).log_softmax(-1)
        cell_entropy = -(cell_logp.exp() * cell_logp).sum(-1)
        zero = torch.zeros_like(type_entropy)
        trunk = enc["trunk"]
        return {
            "logp": type_logp[rows, action_type] + torch.where(is_play, cell_logp[rows, cell], zero),
            "type_entropy": type_entropy,
            "cell_entropy": torch.where(is_play, cell_entropy, zero),
            "value_public": self.value_public(trunk).float(),
            "value_private": self.critic(obs),
            "aux_hand": self.aux_hand(trunk).float(),
            "aux_elixir": self.aux_elixir(trunk).float().squeeze(-1),
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
