"""Frozen copy of the PPO update as it was before the throughput work
(commit 223f26d plus the value_target knob). The tests compare the optimized
update against it; do not edit it to make a test pass."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from cfbot.config import TrainConfig
from cfbot.env import trim_entities
from reference_model import Policy


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
    return torch.optim.AdamW(groups, lr=config.lr, betas=(0.9, 0.999), eps=1e-5)


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
        self.shadow = {name: params[name].detach().clone() for name in self.names}

    @torch.no_grad()
    def step(self, model: Policy) -> None:
        params = dict(model.named_parameters())
        for name in self.names:
            self.shadow[name].lerp_(params[name].detach(), 1 - self.decay)

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


def _gather(obs: dict, pick: torch.Tensor, device: torch.device) -> dict:
    rows = {key: value[pick[:, 0], pick[:, 1]] for key, value in obs.items()}
    return {key: value.to(device) for key, value in trim_entities(rows).items()}


def update(
    model: Policy,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    config: TrainConfig,
    entropy_coef: float,
    device: torch.device,
    rng: np.random.Generator,
    ema: Ema | None = None,
) -> dict[str, float]:
    obs = batch["obs"]
    fields = {key: batch[key] for key in ("type", "cell", "logp", "advantage", "return")}
    # The actor trains on the learner's decisions that had a choice. A row
    # where only waiting is legal has probability one under any weights and so
    # no policy gradient; the critic still learns from it, in its own pass.
    acting = (batch["train"] & batch["choice"]).nonzero()
    waiting = (batch["train"] & ~batch["choice"]).nonzero()
    everything = batch["train"].nonzero()
    use_amp = config.amp and device.type == "cuda"
    actor_params = model.actor_parameters()
    critic_params = list(model.critic.parameters())
    model.train()

    # Normalize advantages over the decisions the actor trains on.
    advantages = batch["advantage"][batch["train"] & batch["choice"]]
    if advantages.numel() > 1:
        adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-6
    else:
        adv_mean, adv_std = 0.0, 1.0

    totals: dict[str, float] = {}
    count = 0
    policy_on = True
    size = config.minibatch
    for _ in range(config.epochs):
        for pick in _minibatches(acting, size, rng):
            mb_obs = _gather(obs, pick, device)
            mb = {key: value[pick[:, 0], pick[:, 1]].to(device) for key, value in fields.items()}

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model.evaluate(mb_obs, mb["type"], mb["cell"])

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
            # The two networks share nothing, so clip them separately.
            grad_norm = torch.nn.utils.clip_grad_norm_(actor_params, config.max_grad_norm)
            critic_norm = torch.nn.utils.clip_grad_norm_(critic_params, config.max_grad_norm)
            if not (torch.isfinite(grad_norm) and torch.isfinite(critic_norm)):
                # Skip a poisoned step rather than write NaNs into the weights.
                totals["skipped"] = totals.get("skipped", 0.0) + 1
                continue
            optimizer.step()
            if ema is not None:
                ema.step(model)

            with torch.no_grad():
                # KL(old || new) estimated as E[(r - 1) - log r]; never negative.
                kl = float(((ratio - 1) - log_ratio).mean())
                metrics = {
                    "loss": float(loss),
                    "policy_loss": float(policy_loss),
                    "value_loss": float(value_loss),
                    "public_value_loss": float(public_loss),
                    "hand_loss": float(hand_loss),
                    "elixir_loss": float(elixir_loss),
                    "entropy": float(entropy),
                    "type_entropy": float(out["type_entropy"].mean()),
                    "kl": kl,
                    "clip_frac": float(((ratio - 1).abs() > config.clip).float().mean()),
                    "grad_norm": float(grad_norm),
                    "critic_grad_norm": float(critic_norm),
                }
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
            if policy_on and kl > 1.5 * config.target_kl:
                # The policy has moved far enough for this rollout. Keep
                # training the value and auxiliary heads on the rest of it.
                policy_on = False
                totals["kl_stops"] = 1.0

    # Critic alone: the forced-wait rows once, then any extra passes over
    # every learner row.
    critic_steps = 0
    for epoch in range(max(1, config.critic_epochs)):
        for pick in _minibatches(waiting if epoch == 0 else everything, size, rng):
            mb_obs = _gather(obs, pick, device)
            returns = batch["return"][pick[:, 0], pick[:, 1]].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model.critic(mb_obs)
            loss = -(model.value_target(returns) * logits.log_softmax(-1)).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True)
            (config.value_coef * loss).backward()
            norm = torch.nn.utils.clip_grad_norm_(critic_params, config.max_grad_norm)
            if torch.isfinite(norm):
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
        result["gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
    return result
