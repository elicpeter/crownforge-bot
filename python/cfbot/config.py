"""Model and training configuration.

Everything a run needs is in these two dataclasses; a checkpoint stores both, so
a saved model can always be rebuilt and a run can always be resumed.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    d_model: int = 192
    n_heads: int = 6
    mlp_ratio: int = 3
    d_card: int = 64
    # Self-attention layers among units, then query layers that read everything.
    entity_layers: int = 1
    query_layers: int = 2
    # Board image at half resolution (16 x 9 cells).
    d_board: int = 96
    board_blocks: int = 2
    # Also let the queries attend to the 144 board cells. Costs about a third
    # more compute and memory; unit tokens already carry positions.
    board_in_memory: bool = False
    d_trunk: int = 384
    fourier_bands: int = 6
    # Separate critic network.
    d_critic: int = 128
    critic_hidden: int = 512
    # Distributional value heads: two-hot targets over this support.
    value_bins: int = 61
    value_range: float = 1.5
    wait_bias: float = 2.0


MODEL_PRESETS = {
    "small": dict(d_model=128, n_heads=4, d_card=48, d_board=64, d_trunk=256, d_critic=96,
                  critic_hidden=384),
    "base": dict(),
    "large": dict(d_model=256, n_heads=8, d_card=96, entity_layers=2, query_layers=3,
                  d_board=128, board_blocks=3, d_trunk=512, d_critic=192, critic_hidden=768),
}

# Cards the default ruleset compiles from incomplete source graphs
# (crownforge docs/fidelity.md). Left out of random decks by default.
DRAFT_CARDS = (
    "MergeMaiden", "LittlePrince", "BossBandit", "MightyMiner", "Clone",
    "GoblinMachine", "GiantBuffer",  # GiantBuffer is Rune Giant
)


@dataclass
class TrainConfig:
    run_dir: str = "runs/default"
    seed: int = 0
    device: str = "auto"

    # Environment
    num_envs: int = 256
    workers: int = 0
    decision_ticks: int = 6
    # Decisions a step may fast-forward while neither player can act.
    max_skip: int = 8
    max_entities: int = 128
    exclude_cards: tuple[str, ...] = DRAFT_CARDS
    mirror_decks: bool = False
    # Deck curriculum: "archetypes" uses the built-in list in decks.py, or give
    # explicit decks. Each new battle draws a deck from the pool with
    # probability pool_prob, otherwise a random deck; the probability anneals.
    deck_pool: str | list[list[str]] | None = "archetypes"
    pool_prob: float = 0.9
    pool_prob_final: float = 0.3
    pool_anneal_iters: int = 3_000

    # Rollout and PPO
    rollout_steps: int = 128
    # Fresh experience is cheap here and the gradient step is the bottleneck,
    # so each decision is used once.
    epochs: int = 1
    # Decisions per gradient step. On Apple GPUs keep this within memory: the
    # base model needs roughly 5 MB per decision. Raise it on larger GPUs.
    minibatch: int = 2048
    # Discount per decision_ticks of game time; longer steps discount more.
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip: float = 0.2
    target_kl: float = 0.03
    value_coef: float = 1.0
    public_value_coef: float = 0.25
    aux_coef: float = 0.1
    entropy_coef: float = 0.01
    entropy_coef_final: float = 0.003
    lr: float = 2e-4
    lr_final: float = 2e-5
    warmup_iters: int = 20
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    flip_augment: bool = True
    amp: bool = True
    compile: bool = False

    # Shaped reward, zero-sum between the two players, annealed to zero so the
    # final objective is winning. Tower health is a potential; trade and leak
    # are in elixir.
    shaping_coef: float = 1.0
    shaping_anneal_iters: int = 4_000
    tower_coef: float = 0.5
    trade_coef: float = 0.01
    leak_coef: float = 0.02

    # League
    self_play_frac: float = 0.6
    baseline_frac: float = 0.15
    baseline_frac_final: float = 0.02
    baseline_anneal_iters: int = 2_000
    league_slots: int = 3
    league_refresh_iters: int = 10
    snapshot_iters: int = 50
    pfsp_power: float = 2.0

    # Schedule and bookkeeping
    total_iters: int = 100_000
    log_iters: int = 1
    checkpoint_iters: int = 25

    model: ModelConfig = field(default_factory=ModelConfig)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @staticmethod
    def from_dict(data: dict) -> "TrainConfig":
        data = dict(data)
        model = ModelConfig(**data.pop("model", {}))
        if data.get("exclude_cards") is not None:
            data["exclude_cards"] = tuple(data["exclude_cards"])
        return TrainConfig(model=model, **data)


def pick_device(name: str = "auto"):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
