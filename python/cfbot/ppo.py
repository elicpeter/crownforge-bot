"""PPO update for the actor and its separate critic."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grads_with_norm_, get_total_norm

from . import actions as A
from .config import TrainConfig
from .env import CRITIC_KEYS, OBS_KEYS, gather_rows
from .model import Policy


def autocast(config: TrainConfig, device: torch.device):
    """Mixed precision for forward passes: bfloat16 on CUDA when ``amp`` is
    set, nothing elsewhere. Half precision on an Apple GPU was measured and
    is slower than float32 there, several times slower when acting."""
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.amp and device.type == "cuda")


def fused_optimizer(config: TrainConfig, device: torch.device) -> bool:
    """Whether AdamW runs as one fused kernel per tensor. Without it PyTorch
    loops over about ten small operations per tensor on Apple GPUs."""
    return config.fused_optimizer and device.type in ("cuda", "mps")


def make_optimizer(model: Policy, config: TrainConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        plain = parameter.ndim < 2 or "emb" in name or "board_base" in name or "cell_" in name
        (no_decay if plain else decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    device = next(model.parameters()).device
    # None, not False: False would also switch off the multi-tensor path that
    # PyTorch picks by itself on CUDA.
    fused = True if fused_optimizer(config, device) else None
    return torch.optim.AdamW(groups, lr=config.lr, betas=(0.9, 0.999), eps=1e-5, fused=fused)


def load_optimizer(optimizer: torch.optim.Optimizer, state: dict) -> None:
    """Restore optimizer state from a checkpoint made on any device or with
    another setting of ``fused_optimizer``.

    A checkpoint stores each group's ``fused`` and ``foreach`` switches, and
    loading it would silently bring them back; the live optimizer's own are
    kept instead. PyTorch then puts every step counter where that
    implementation wants it."""
    state = dict(state)
    groups = []
    for saved, live in zip(state["param_groups"], optimizer.param_groups):
        saved = dict(saved)
        for key in ("fused", "foreach"):
            saved[key] = live.get(key)
        groups.append(saved)
    state["param_groups"] = groups
    optimizer.load_state_dict(state)


class Passes:
    """The two forward passes of a gradient step: the actor with its heads,
    and the critic alone. With ``config.compile`` they go through
    ``torch.compile``; the model itself is never wrapped, so its parameters,
    ``state_dict`` and checkpoints are the same either way.

    A pass whose compiled forward raises falls back to the plain one for good.
    Running out of memory is not such a failure and is raised as it is. The
    backward half is compiled on the first ``backward()``, outside this class:
    a failure there is raised too, so try ``compile`` on a short run first."""

    def __init__(self, model: Policy, config: TrainConfig | None = None):
        self.plain = {"evaluate": model.evaluate, "critic": model.critic}
        self.fast = dict(self.plain)
        if config is not None and config.compile:
            for name, function in self.plain.items():
                self.fast[name] = torch.compile(function, backend=config.compile_backend)

    def _run(self, name: str, *args):
        function = self.fast[name]
        if function is self.plain[name]:
            return function(*args)
        try:
            return function(*args)
        except torch.OutOfMemoryError:
            raise
        except Exception as error:  # compiler errors come in many types
            print(f"torch.compile: {name} failed ({type(error).__name__}: {error}); "
                  f"continuing without it", flush=True)
            self.fast[name] = self.plain[name]
        return self.plain[name](*args)

    def evaluate(self, obs, action_type, cell, plays=None):
        return self._run("evaluate", obs, action_type, cell, plays)

    def critic(self, obs):
        return self._run("critic", obs)


def schedule(config: TrainConfig, progress: float, iteration: int) -> tuple[float, float]:
    """Learning rate and entropy coefficient at a point of the run."""
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    lr = config.lr_final + (config.lr - config.lr_final) * cosine
    lr *= min(1.0, (iteration + 1) / max(1, config.warmup_iters))
    entropy = config.entropy_coef_final + (config.entropy_coef - config.entropy_coef_final) * cosine
    return lr, entropy


class Ema:
    """Exponential moving average of the actor's weights."""

    def __init__(self, model: Policy, decay: float):
        self.decay = decay
        self.names = [name for name, _ in model.named_parameters() if not name.startswith("critic.")]
        params = dict(model.named_parameters())
        # The parameter objects live as long as the model: loading weights or
        # moving the model changes their contents, not their identity.
        self.params = [params[name] for name in self.names]
        self.shadow = {name: params[name].detach().clone() for name in self.names}
        self.shadows = list(self.shadow.values())

    @torch.no_grad()
    def step(self, model: Policy | None = None) -> None:
        torch._foreach_lerp_(self.shadows, self.params, 1 - self.decay)

    def state_dict(self) -> dict:
        return {name: value.cpu() for name, value in self.shadow.items()}

    def load_state_dict(self, state: dict) -> None:
        for name in self.names:
            if name in state:
                self.shadow[name].copy_(state[name])


def _minibatches(index: torch.Tensor, size: int, rng: np.random.Generator):
    """Shuffled minibatches of exactly ``size`` rows (the tail wraps around),
    so tensor shapes do not change between gradient steps."""
    if len(index) == 0:
        return
    permutation = rng.permutation(len(index))
    padded = np.resize(permutation, -(-len(permutation) // size) * size)
    order = index[torch.from_numpy(padded)]
    for start in range(0, len(order), size):
        yield order[start : start + size]


# The placement head runs on the play rows of a minibatch, padded to a
# multiple of this many so only a few tensor shapes ever occur.
PLAY_BUCKET = 256


def _plays(kinds: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Which rows of a minibatch stored a play, for ``Policy.evaluate``:
    their positions and a weight of one each, padded with row 0 at weight
    zero. Built from the stored action types on the CPU."""
    index = ((kinds >= 1) & (kinds <= A.PLAY_TYPES)).nonzero().squeeze(-1)
    bucket = min(PLAY_BUCKET, len(kinds))
    size = max(1, -(-len(index) // bucket)) * bucket
    rows = torch.zeros(size, dtype=torch.int64)
    rows[: len(index)] = index
    weight = torch.zeros(size)
    weight[: len(index)] = 1.0
    return rows.to(device), weight.to(device)


def _norm(parameters) -> torch.Tensor:
    """Gradient norm of ``parameters`` as ``clip_grad_norm_`` computes it."""
    return get_total_norm([p.grad for p in parameters if p.grad is not None])


def _clip(parameters, norm: torch.Tensor, max_norm: float, scale: float) -> None:
    """Scale gradients as ``clip_grad_norm_`` does. ``scale`` is the factor it
    would use, already read back: at exactly one the multiplication changes
    nothing and is skipped."""
    if scale < 1.0:
        clip_grads_with_norm_(parameters, max_norm, norm)


def update(
    model: Policy,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    config: TrainConfig,
    entropy_coef: float,
    device: torch.device,
    rng: np.random.Generator,
    ema: Ema | None = None,
    passes: Passes | None = None,
) -> dict[str, float]:
    """One pass of PPO over a rollout.

    The rollout is gathered where it lives: on the training device when the
    collector left it there, so a gradient step uploads one index instead of
    twenty-five tensors. Each step reads its numbers back from the device
    once, after the gradient norms are known."""
    passes = passes or Passes(model)
    # Rows are addressed by a flat index over [T, rows].
    obs = {key: value.flatten(0, 1) for key, value in batch["obs"].items()}
    counts = batch["entity_count"].flatten()
    kinds = batch["type"].flatten()
    fields = {key: batch[key].flatten().to(device)
              for key in ("type", "cell", "logp", "advantage", "return")}
    # The actor trains on the learner's decisions that had a choice. A row
    # where only waiting is legal has probability one under any weights and so
    # no policy gradient; the critic still learns from it, in its own pass.
    acting = (batch["train"] & batch["choice"]).flatten().nonzero().squeeze(-1)
    waiting = (batch["train"] & ~batch["choice"]).flatten().nonzero().squeeze(-1)
    everything = batch["train"].flatten().nonzero().squeeze(-1)
    actor_params = model.actor_parameters()
    critic_params = list(model.critic.parameters())
    max_norm = config.max_grad_norm
    model.train()

    # Normalize advantages over the decisions the actor trains on.
    advantages = batch["advantage"][batch["train"] & batch["choice"]]
    if advantages.numel() > 1:
        adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-6
    else:
        adv_mean, adv_std = 0.0, 1.0

    names = ("loss", "policy_loss", "value_loss", "public_value_loss", "hand_loss", "elixir_loss",
             "entropy", "type_entropy", "kl", "clip_frac", "grad_norm", "critic_grad_norm")
    totals: dict[str, float] = {}
    count = 0
    policy_on = True
    size = config.minibatch
    for _ in range(config.epochs):
        for pick in _minibatches(acting, size, rng):
            mb_obs = gather_rows(obs, counts, pick, OBS_KEYS, device)
            where = pick.to(device)
            mb = {key: value.index_select(0, where) for key, value in fields.items()}

            plays = _plays(kinds[pick], device)

            with autocast(config, device):
                out = passes.evaluate(mb_obs, mb["type"], mb["cell"], plays)

            advantage = (mb["advantage"] - adv_mean) / adv_std
            log_ratio = out["logp"] - mb["logp"]
            ratio = log_ratio.exp()
            clipped = ratio.clamp(1 - config.clip, 1 + config.clip)
            policy_loss = -torch.min(ratio * advantage, clipped * advantage).mean()

            target = model.value_target(mb["return"])
            value_loss = -(target * out["value_private"].log_softmax(-1)).sum(-1).mean()
            public_loss = -(target * out["value_public"].log_softmax(-1)).sum(-1).mean()

            # Auxiliary targets from the opponent's hidden state.
            hand = mb_obs["priv_hand"].long()
            hand_target = torch.zeros_like(out["aux_hand"]).scatter_(-1, hand, 1.0)
            hand_target[..., 0] = 0
            hand_loss = F.binary_cross_entropy_with_logits(
                out["aux_hand"], hand_target, reduction="none"
            ).sum(-1).mean()
            elixir_target = mb_obs["priv_global"][:, 0] / model.max_elixir
            elixir_loss = ((out["aux_elixir"] - elixir_target) ** 2).mean()

            entropy = (out["type_entropy"] + out["cell_entropy"]).mean()
            loss = (
                config.value_coef * value_loss
                + config.public_value_coef * public_loss
                + config.aux_coef * (hand_loss / 4 + elixir_loss)
            )
            if policy_on:
                loss = loss + policy_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                # The two networks share nothing, so clip them separately.
                grad_norm, critic_norm = _norm(actor_params), _norm(critic_params)
                # KL(old || new) estimated as E[(r - 1) - log r]; never negative.
                kl = ((ratio - 1) - log_ratio).mean()
                # Everything this step reports, and the two clip factors, in
                # one transfer.
                numbers = torch.stack([
                    loss, policy_loss, value_loss, public_loss, hand_loss, elixir_loss, entropy,
                    out["type_entropy"].mean(), kl,
                    ((ratio - 1).abs() > config.clip).float().mean(), grad_norm, critic_norm,
                    torch.clamp(max_norm / (grad_norm + 1e-6), max=1.0),
                    torch.clamp(max_norm / (critic_norm + 1e-6), max=1.0),
                ]).tolist()
            metrics = dict(zip(names, numbers))
            if not (math.isfinite(metrics["grad_norm"]) and math.isfinite(metrics["critic_grad_norm"])):
                # Skip a poisoned step rather than write NaNs into the weights.
                totals["skipped"] = totals.get("skipped", 0.0) + 1
                continue
            _clip(actor_params, grad_norm, max_norm, numbers[-2])
            _clip(critic_params, critic_norm, max_norm, numbers[-1])
            optimizer.step()
            if ema is not None:
                ema.step(model)

            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
            if policy_on and metrics["kl"] > 1.5 * config.target_kl:
                # The policy has moved far enough for this rollout. Keep
                # training the value and auxiliary heads on the rest of it.
                policy_on = False
                totals["kl_stops"] = 1.0

    # Critic alone: the forced-wait rows once, then any extra passes over
    # every learner row.
    critic_steps = 0
    returns_all = fields["return"]
    for epoch in range(max(1, config.critic_epochs)):
        for pick in _minibatches(waiting if epoch == 0 else everything, size, rng):
            mb_obs = gather_rows(obs, counts, pick, CRITIC_KEYS, device)
            returns = returns_all.index_select(0, pick.to(device))
            with autocast(config, device):
                logits = passes.critic(mb_obs)
            loss = -(model.value_target(returns) * logits.log_softmax(-1)).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True)
            (config.value_coef * loss).backward()
            with torch.no_grad():
                norm = _norm(critic_params)
                value, scale = torch.stack([norm, torch.clamp(max_norm / (norm + 1e-6), max=1.0)]).tolist()
            if math.isfinite(value):
                _clip(critic_params, norm, max_norm, scale)
                optimizer.step()
                critic_steps += 1
            else:
                totals["skipped"] = totals.get("skipped", 0.0) + 1

    special = ("skipped", "kl_stops")
    result = {key: value / max(1, count) for key, value in totals.items() if key not in special}
    result["updates"] = count
    result["critic_updates"] = critic_steps
    result["skipped"] = totals.get("skipped", 0.0)
    result["kl_stop"] = totals.get("kl_stops", 0.0)
    # How much of the return variance the critic explains on learner decisions.
    mask = batch["train"]
    returns, values = batch["return"][mask], batch["value"][mask]
    variance = returns.var()
    result["explained_variance"] = float(1 - (returns - values).var() / variance) if variance > 0 else 0.0
    # Targets outside the value support are clamped; this should stay near zero.
    result["return_clamped"] = float((returns.abs() > config.model.value_range).float().mean())
    result["choice_frac"] = float(len(acting) / max(1, len(everything)))
    if device.type == "cuda":
        # Peak since the last update, so a leak shows as a rising number.
        result["gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
        torch.cuda.reset_peak_memory_stats(device)
    return result
