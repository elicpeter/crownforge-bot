"""Self-play training entry point.

    python -m cfbot.train --run runs/main --set budget_hours=330
    python -m cfbot.train --run runs/main --resume
    python -m cfbot.train --run runs/big --preset large --set num_envs=512

Stop with Ctrl-C; the run checkpoints and can be resumed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import signal
import statistics
import time

import numpy as np
import torch

from .config import MODEL_PRESETS, ModelConfig, TrainConfig, override, pick_device
from .decks import DeckStats, resolve_pool
from .env import Env
from .evaluate import run_gauntlet
from .league import League, rules_label
from .model import Policy, count_parameters
from .ppo import Ema, Passes, load_optimizer, make_optimizer, schedule, update
from .rollout import Collector


def build_config(args) -> TrainConfig:
    config_path = os.path.join(args.run, "config.json")
    if args.resume and os.path.exists(config_path):
        with open(config_path) as handle:
            config = TrainConfig.from_dict(json.load(handle))
    else:
        config = TrainConfig(run_dir=args.run, model=ModelConfig(**MODEL_PRESETS[args.preset]))
    for item in args.set or []:
        override(config, item)
    config.run_dir = args.run
    return config


def run_progress(config: TrainConfig, iteration: int, elapsed: float) -> float:
    """How far the run is, from 0 to 1. Every schedule follows this."""
    if config.budget_hours:
        return min(1.0, elapsed / (config.budget_hours * 3600.0))
    return min(1.0, iteration / max(1, config.total_iters))


def deck_mix(config: TrainConfig, progress: float) -> tuple[float, float]:
    ramp = min(1.0, progress / max(1e-9, config.deck_ramp_frac))
    pool = config.pool_prob + (config.pool_prob_final - config.pool_prob) * ramp
    mutant = config.mutant_prob + (config.mutant_prob_final - config.mutant_prob) * ramp
    return pool, mutant


def alarms(record: dict, speeds: list[float]) -> list[str]:
    """Signs that an unattended run has gone wrong."""
    found = []
    if record.get("skipped", 0) > 0:
        found.append("non-finite gradients skipped")
    if not all(math.isfinite(record.get(key, 0.0)) for key in ("loss", "entropy", "kl")):
        found.append("non-finite loss")
    if record["iteration"] > 50:
        if record.get("play_rate", 1.0) < 0.005:
            found.append("policy has stopped playing cards")
        if record.get("entropy", 1.0) < 0.02:
            found.append("policy entropy has collapsed")
        if record.get("explained_variance", 0.0) < -0.5:
            found.append("critic is worse than a constant")
    if record.get("return_clamped", 0.0) > 0.05:
        found.append("many value targets fall outside the value range")
    if len(speeds) >= 20 and record["samples_per_s"] < 0.5 * statistics.median(speeds):
        found.append("throughput has halved")
    if record.get("crashes", 0) > 0:
        found.append("simulator crashes")
    return found


def save_checkpoint(path, model, optimizer, league, ema, deck_stats, config, spec, state) -> None:
    tmp = path + ".tmp"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "league": league.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
            "deck_stats": deck_stats.state_dict(),
            "config": dataclasses.asdict(config),
            "rules": rules_label(spec),
            **state,
        },
        tmp,
    )
    os.replace(tmp, path)
    with open(os.path.join(config.run_dir, "deck_stats.json.tmp"), "w") as handle:
        json.dump(deck_stats.summary(spec["card_names"]), handle)
    os.replace(os.path.join(config.run_dir, "deck_stats.json.tmp"),
               os.path.join(config.run_dir, "deck_stats.json"))
    league.save_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="runs/default", help="run directory")
    parser.add_argument("--preset", default="base", choices=sorted(MODEL_PRESETS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--iters", type=int, help="stop after this many iterations in this invocation")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config field")
    args = parser.parse_args()

    config = build_config(args)
    os.makedirs(config.run_dir, exist_ok=True)
    device = pick_device(config.device)
    torch.manual_seed(config.seed)

    checkpoint_path = os.path.join(config.run_dir, "checkpoint.pt")
    data = None
    if args.resume and os.path.exists(checkpoint_path):
        # Load to the CPU: the model, optimizer and averaged weights each move
        # their tensors where they belong. Mapping straight to the device would
        # also move the optimizer's step counters there, and AdamW then waits
        # for the GPU once per parameter tensor in every gradient step.
        data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {"iteration": 0, "samples": 0, "elapsed": 0.0, "next_gauntlet": 0.0}
    if data is not None:
        state.update({key: data[key] for key in state if key in data})
    iteration = state["iteration"]

    # A resumed run continues with fresh battles instead of replaying the
    # deck and seed sequence it started with.
    env = Env(
        config.num_envs,
        workers=config.workers,
        decision_ticks=config.decision_ticks,
        seed=config.seed + iteration,
        max_entities=config.max_entities,
        deck_pool=resolve_pool(config.deck_pool),
        pool_prob=config.pool_prob,
        mirror_decks=config.mirror_decks,
        max_skip=config.max_skip,
        exclude_cards=config.exclude_cards,
        rules=config.rules,
        level_jitter=config.level_jitter,
        device=device,
    )
    spec = env.spec
    model = Policy(spec, config.model).to(device)
    optimizer = make_optimizer(model, config)
    league = League(config, spec, device)
    ema = Ema(model, config.ema_decay) if config.ema_decay > 0 else None
    deck_stats = DeckStats(spec["n_cards"])
    if data is not None:
        trained_on = data.get("rules")
        if trained_on and trained_on != rules_label(spec):
            print(f"warning: checkpoint was trained on {trained_on}, now running {rules_label(spec)}")
        model.load_state_dict(data["model"])
        load_optimizer(optimizer, data["optimizer"])
        league.load_state_dict(data["league"])
        if ema is not None and data.get("ema"):
            ema.load_state_dict(data["ema"])
        if data.get("deck_stats"):
            deck_stats.load_state_dict(data["deck_stats"])
        print(f"resumed {checkpoint_path} at iteration {iteration}")
    with open(os.path.join(config.run_dir, "config.json"), "w") as handle:
        handle.write(config.to_json())

    collector = Collector(config, env, model, league, deck_stats, seed=config.seed + iteration)
    passes = Passes(model, config)
    rng = np.random.default_rng(config.seed + iteration)
    limit = f"{config.budget_hours} h budget" if config.budget_hours else f"{config.total_iters} iterations"
    print(
        f"device {device}, {count_parameters(model) / 1e6:.1f}M parameters, "
        f"{count_parameters(model.critic) / 1e6:.1f}M of them critic, "
        f"{config.num_envs} battles x {config.rollout_steps} decisions per iteration, "
        f"{spec['random_deck_cards']} cards in random decks, rules {spec['rules_id']}, {limit}"
    )

    stopping = False

    def request_stop(*_):
        nonlocal stopping
        stopping = True
        print("stopping after this iteration...")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def checkpoint():
        save_checkpoint(checkpoint_path, model, optimizer, league, ema, deck_stats, config, spec, state)

    log = open(os.path.join(config.run_dir, "metrics.jsonl"), "a")
    end = min(config.total_iters, iteration + args.iters) if args.iters else config.total_iters
    speeds: list[float] = []
    progress = run_progress(config, iteration, state["elapsed"])
    while iteration < end and not stopping and not (config.budget_hours and progress >= 1.0):
        started = time.time()
        if iteration % config.league_refresh_iters == 0:
            if league.snapshots:
                collector.taint(league.refresh_slots())
            collector.relayout(progress)
        shaping = config.shaping_coef * max(0.0, 1 - progress / max(1e-9, config.shaping_anneal_frac))
        pool_prob, mutant_prob = deck_mix(config, progress)
        env.set_mix(pool_prob, mutant_prob)
        lr, entropy_coef = schedule(config, progress, iteration)
        for group in optimizer.param_groups:
            group["lr"] = lr

        batch, stats = collector.collect(iteration, shaping)
        collected = time.time()
        metrics = update(model, optimizer, batch, config, entropy_coef, device, rng, ema, passes)
        finished = time.time()

        iteration += 1
        learner_samples = int(batch["train"].sum())
        value_mean = float(batch["value"][batch["train"]].mean())
        # Let go of the rollout now, so the next one is not collected next to
        # it: it is about a gigabyte, on the GPU when it is kept there.
        del batch
        state["iteration"] = iteration
        state["samples"] += learner_samples
        if iteration % config.snapshot_iters == 0 or iteration == 1:
            league.add_snapshot(model, iteration)
            if len(league.snapshots) == 1:
                collector.taint(league.refresh_slots())
                collector.relayout(progress)

        gauntlet = {}
        if config.gauntlet_frac > 0 and progress >= state["next_gauntlet"]:
            gauntlet = run_gauntlet(model, league, config, device, iteration)
            state["next_gauntlet"] = progress + config.gauntlet_frac

        # Evaluation time counts against the budget too.
        state["elapsed"] += time.time() - started
        progress = run_progress(config, iteration, state["elapsed"])
        last = iteration == end or stopping or (config.budget_hours and progress >= 1.0)
        if iteration % config.checkpoint_iters == 0 or last:
            checkpoint()

        record = {
            "iteration": iteration,
            "time": round(time.time(), 1),
            "hours": round(state["elapsed"] / 3600.0, 3),
            "progress": round(progress, 5),
            "samples": state["samples"],
            "samples_per_s": round(learner_samples / (finished - started), 1),
            "collect_s": round(collected - started, 2),
            "update_s": round(finished - collected, 2),
            "lr": lr,
            "entropy_coef": entropy_coef,
            "shaping": shaping,
            "pool_prob": pool_prob,
            "mutant_prob": mutant_prob,
            "value_mean": value_mean,
            **{key: round(value, 5) for key, value in metrics.items()},
            **{key: round(value, 3) for key, value in stats.items()},
            **league.summary(),
            **gauntlet,
        }
        found = alarms(record, speeds)
        if found:
            record["alarms"] = found
        speeds = (speeds + [record["samples_per_s"]])[-50:]
        log.write(json.dumps(record) + "\n")
        log.flush()
        if iteration % config.log_iters == 0:
            wins = " ".join(f"{k[4:]}={v:.2f}" for k, v in record.items() if k.startswith("win_"))
            rating = f" | rating {gauntlet['gauntlet_elo']:+.0f}" if gauntlet else ""
            print(
                f"it {iteration:5d} {100 * progress:5.1f}% | {record['samples_per_s']:6.0f} sps | "
                f"ep {int(stats.get('episodes', 0)):3d} {stats['mean_seconds']:3.0f}s | "
                f"pi {metrics.get('policy_loss', 0):+.4f} v {metrics.get('value_loss', 0):.3f} "
                f"ent {metrics.get('entropy', 0):.2f} kl {metrics.get('kl', 0):.4f} "
                f"ev {metrics['explained_variance']:+.2f} play {stats['play_rate']:.3f} | {wins}{rating}",
                flush=True,
            )
        for message in found:
            print(f"  ALARM: {message}", flush=True)

    checkpoint()
    for blue, red in env.crashes()[:20]:
        print("simulator crash:", blue, "vs", red)
    print(f"saved {checkpoint_path} at iteration {iteration}")


if __name__ == "__main__":
    main()
