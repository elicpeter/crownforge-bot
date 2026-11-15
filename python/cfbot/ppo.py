"""PPO update for the actor and its separate critic."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from . import actions as A
from .config import TrainConfig
from .model import Policy


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


def schedule(config: TrainConfig, iteration: int) -> tuple[float, float]:
    """Learning rate and entropy coefficient for an iteration."""
    progress = min(1.0, iteration / max(1, config.total_iters))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    lr = config.lr_final + (config.lr - config.lr_final) * cosine
    lr *= min(1.0, (iteration + 1) / max(1, config.warmup_iters))
    entropy = config.entropy_coef_final + (config.entropy_coef - config.entropy_coef_final) * cosine
    return lr, entropy


def update(
    model: Policy,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    config: TrainConfig,
    entropy_coef: float,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    obs = batch["obs"]
    fields = {key: batch[key] for key in ("type", "cell", "logp", "advantage", "return")}
    # Train only on decisions the learner made.
    usable = batch["train"].nonzero()
    use_amp = config.amp and device.type == "cuda"
    actor_params = model.actor_parameters()
    critic_params = list(model.critic.parameters())
    model.train()

    # Normalize advantages over the whole rollout, not per minibatch.
    advantages = batch["advantage"][batch["train"]]
    adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-6

    totals: dict[str, float] = {}
    count = 0
    stop = False
    size = config.minibatch
    for _ in range(config.epochs):
        # Every minibatch has exactly `size` decisions (the tail wraps around),
        # so tensor shapes never change between gradient steps.
        permutation = rng.permutation(len(usable))
        padded = np.resize(permutation, -(-len(permutation) // size) * size)
        order = usable[torch.from_numpy(padded)]
        for start in range(0, len(order), size):
            pick = order[start : start + size]
            step_index, row_index = pick[:, 0], pick[:, 1]
            mb_obs = {key: value[step_index, row_index].to(device) for key, value in obs.items()}
            mb = {key: value[step_index, row_index].to(device) for key, value in fields.items()}
            cell = mb["cell"]
            if config.flip_augment and rng.random() < 0.5:
                mb_obs = A.flip_observation(mb_obs)
                cell = A.flip_cell(cell)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model.evaluate(mb_obs, mb["type"], cell)

            advantage = (mb["advantage"] - adv_mean) / adv_std
            log_ratio = out["logp"] - mb["logp"]
            ratio = log_ratio.exp()
            clipped = ratio.clamp(1 - config.clip, 1 + config.clip)
            policy_loss = -torch.min(ratio * advantage, clipped * advantage).mean()

            target = model.two_hot(mb["return"])
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
                policy_loss
                + config.value_coef * value_loss
                + config.public_value_coef * public_loss
                + config.aux_coef * (hand_loss / 4 + elixir_loss)
                - entropy_coef * entropy
            )

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

            with torch.no_grad():
                kl = float(-log_ratio.mean())
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
            if kl > 1.5 * config.target_kl:
                stop = True
                break
        if stop:
            break

    result = {key: value / max(1, count) for key, value in totals.items() if key != "skipped"}
    result["updates"] = count
    result["skipped"] = totals.get("skipped", 0.0)
    # How much of the return variance the critic explains on learner decisions.
    mask = batch["train"]
    returns, values = batch["return"][mask], batch["value"][mask]
    variance = returns.var()
    result["explained_variance"] = float(1 - (returns - values).var() / variance) if variance > 0 else 0.0
    return result
