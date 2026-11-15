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
    # Share of the public-value and auxiliary gradient that reaches the actor
    # encoder. At 1 those losses outweigh the policy loss there several times.
    aux_grad_scale: float = 0.1


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
    # Rules: None for the bundled level-11 rules, "complete" for the bundled bank
    # with every card level, or a path to a rules document. level_jitter draws a
    # per-card level offset from an inclusive range and needs rules that hold
    # those levels ("complete").
    rules: str | None = None
    level_jitter: tuple[int, int] | None = None

    # Decks. Each side of a new battle draws a pool deck with probability
    # pool_prob, a pool deck with one to three cards replaced with probability
    # mutant_prob, and a random playable deck otherwise. The pool is the
    # archetype list in decks.py unless explicit decks are given. The shares
    # move linearly to their final values over the first deck_ramp_frac of the
    # run; random decks stay the largest source so every card keeps exposure.
    deck_pool: str | list[list[str]] | None = "archetypes"
    pool_prob: float = 0.10
    mutant_prob: float = 0.20
    pool_prob_final: float = 0.30
    mutant_prob_final: float = 0.25
    deck_ramp_frac: float = 0.15

    # Rollout and PPO
    rollout_steps: int = 128
    # Fresh experience is cheap here and the gradient step is the bottleneck,
    # so each decision is used once by the policy.
    epochs: int = 1
    # Extra passes of the critic alone over the rollout. It is small and separate.
    critic_epochs: int = 1
    # Decisions per gradient step. The base model keeps about 4 MB of
    # activations per decision at full entity width; 2048 fits a 16 GB GPU.
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
    # Show the acting policy a left-right mirrored view for half its decisions.
    # Off by default: the simulator is not exactly mirror-symmetric.
    flip_augment: bool = False
    amp: bool = True
    # Moving average of the actor's weights, per gradient step. 0 disables.
    ema_decay: float = 0.999

    # Shaped reward, zero-sum between the two players, annealed to zero over
    # the first shaping_anneal_frac of the run so the final objective is
    # winning. Tower health is a potential; trade and leak are in elixir.
    shaping_coef: float = 1.0
    shaping_anneal_frac: float = 0.25
    tower_coef: float = 0.5
    trade_coef: float = 0.01
    leak_coef: float = 0.02

    # League
    self_play_frac: float = 0.6
    baseline_frac: float = 0.15
    baseline_frac_final: float = 0.02
    baseline_anneal_frac: float = 0.15
    league_slots: int = 3
    league_refresh_iters: int = 10
    snapshot_iters: int = 50
    pfsp_power: float = 2.0
    # Share of slot draws that ignore win rates, and how many of the newest
    # snapshots the reserved "recent" slot draws from.
    league_explore: float = 0.10
    league_recent: int = 5
    # Snapshots kept on disk besides the rating anchors; older ones are thinned.
    max_snapshots: int = 300
    # Train only against this checkpoint (an exploiter probe): no self-play,
    # no scripted bots, no snapshots.
    fixed_opponent: str | None = None

    # Schedule. Progress runs from 0 to 1 and drives every schedule above. With
    # budget_hours it is elapsed training time over the budget, and the run
    # stops when the budget is spent; otherwise it is iteration / total_iters.
    total_iters: int = 100_000
    budget_hours: float | None = None
    log_iters: int = 1
    checkpoint_iters: int = 25

    # Gauntlet: paired games against fixed anchors every gauntlet_frac of the
    # run, on gauntlet_pairs deck pairs played from both seats. 0 disables.
    gauntlet_frac: float = 0.02
    gauntlet_pairs: int = 32
    gauntlet_anchors: int = 6

    model: ModelConfig = field(default_factory=ModelConfig)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @staticmethod
    def from_dict(data: dict) -> "TrainConfig":
        """Rebuild a config, ignoring settings that no longer exist."""
        data = dict(data)
        model_data = data.pop("model", {})
        model_fields = {f.name for f in dataclasses.fields(ModelConfig)}
        model = ModelConfig(**{k: v for k, v in model_data.items() if k in model_fields})
        known = {f.name for f in dataclasses.fields(TrainConfig)}
        dropped = sorted(set(data) - known)
        if dropped:
            print(f"ignoring unknown config keys: {', '.join(dropped)}")
        data = {k: v for k, v in data.items() if k in known}
        for key in ("exclude_cards", "level_jitter"):
            if data.get(key) is not None:
                data[key] = tuple(data[key])
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
