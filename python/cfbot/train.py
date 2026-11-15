"""Self-play training entry point.

    python -m cfbot.train --run runs/main
    python -m cfbot.train --run runs/main --resume
    python -m cfbot.train --run runs/big --preset large --set num_envs=512

Stop with Ctrl-C; the run checkpoints and can be resumed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import signal
import time

import numpy as np
import torch

from .config import MODEL_PRESETS, ModelConfig, TrainConfig, pick_device
from .decks import resolve_pool
from .env import Env
from .league import League
from .model import Policy, count_parameters
from .ppo import make_optimizer, schedule, update
from .rollout import Collector


def parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def build_config(args) -> TrainConfig:
    config_path = os.path.join(args.run, "config.json")
    if args.resume and os.path.exists(config_path):
        with open(config_path) as handle:
            config = TrainConfig.from_dict(json.load(handle))
    else:
        config = TrainConfig(run_dir=args.run, model=ModelConfig(**MODEL_PRESETS[args.preset]))
    for item in args.set or []:
        key, _, value = item.partition("=")
        target = config
        if key.startswith("model."):
            target, key = config.model, key[len("model."):]
        if not hasattr(target, key):
            raise SystemExit(f"unknown config key: {item}")
        setattr(target, key, parse_value(value))
    config.run_dir = args.run
    return config


def save_checkpoint(path, model, optimizer, league, config, iteration, samples) -> None:
    tmp = path + ".tmp"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "league": league.state_dict(),
            "config": dataclasses.asdict(config),
            "iteration": iteration,
            "samples": samples,
        },
        tmp,
    )
    os.replace(tmp, path)


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

    env = Env(
        config.num_envs,
        workers=config.workers,
        decision_ticks=config.decision_ticks,
        seed=config.seed,
        max_entities=config.max_entities,
        deck_pool=resolve_pool(config.deck_pool),
        pool_prob=config.pool_prob,
        mirror_decks=config.mirror_decks,
        max_skip=config.max_skip,
        exclude_cards=config.exclude_cards,
        device=device,
    )
    model = Policy(env.spec, config.model).to(device)
    optimizer = make_optimizer(model, config)
    league = League(config, env.spec, device)
    checkpoint_path = os.path.join(config.run_dir, "checkpoint.pt")
    iteration = samples = 0
    if args.resume and os.path.exists(checkpoint_path):
        data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(data["model"])
        optimizer.load_state_dict(data["optimizer"])
        league.load_state_dict(data["league"])
        iteration, samples = data["iteration"], data["samples"]
        print(f"resumed {checkpoint_path} at iteration {iteration}")
    with open(os.path.join(config.run_dir, "config.json"), "w") as handle:
        handle.write(config.to_json())

    train_model = torch.compile(model) if config.compile and device.type == "cuda" else model
    collector = Collector(config, env, model, league)
    rng = np.random.default_rng(config.seed + iteration)
    print(
        f"device {device}, {count_parameters(model) / 1e6:.1f}M parameters, "
        f"{count_parameters(model.critic) / 1e6:.1f}M of them critic, "
        f"{config.num_envs} battles x {config.rollout_steps} decisions per iteration, "
        f"{env.spec['random_deck_cards']} cards in random decks"
    )

    stopping = False

    def request_stop(*_):
        nonlocal stopping
        stopping = True
        print("stopping after this iteration...")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    log = open(os.path.join(config.run_dir, "metrics.jsonl"), "a")
    end = min(config.total_iters, iteration + args.iters) if args.iters else config.total_iters
    while iteration < end and not stopping:
        started = time.time()
        if iteration % config.league_refresh_iters == 0:
            if league.snapshots:
                collector.taint(league.refresh_slots())
            collector.relayout(iteration)
        shaping = config.shaping_coef * max(0.0, 1 - iteration / max(1, config.shaping_anneal_iters))
        mix = min(1.0, iteration / max(1, config.pool_anneal_iters))
        pool_prob = config.pool_prob + (config.pool_prob_final - config.pool_prob) * mix
        env.set_pool_prob(pool_prob)
        lr, entropy_coef = schedule(config, iteration)
        for group in optimizer.param_groups:
            group["lr"] = lr

        batch, stats = collector.collect(iteration, shaping)
        collected = time.time()
        metrics = update(train_model, optimizer, batch, config, entropy_coef, device, rng)
        finished = time.time()

        iteration += 1
        learner_samples = int(batch["train"].sum())
        samples += learner_samples
        if iteration % config.snapshot_iters == 0 or iteration == 1:
            league.add_snapshot(model, iteration)
            if len(league.snapshots) == 1:
                collector.taint(league.refresh_slots())
                collector.relayout(iteration)
        if iteration % config.checkpoint_iters == 0 or iteration == end or stopping:
            save_checkpoint(checkpoint_path, model, optimizer, league, config, iteration, samples)
            league.save_index()

        record = {
            "iteration": iteration,
            "samples": samples,
            "samples_per_s": round(learner_samples / (finished - started), 1),
            "collect_s": round(collected - started, 2),
            "update_s": round(finished - collected, 2),
            "lr": lr,
            "entropy_coef": entropy_coef,
            "shaping": shaping,
            "pool_prob": pool_prob,
            "value_mean": float(batch["value"][batch["train"]].mean()),
            **{key: round(value, 5) for key, value in metrics.items()},
            **{key: round(value, 3) for key, value in stats.items()},
            **league.summary(),
        }
        log.write(json.dumps(record) + "\n")
        log.flush()
        if iteration % config.log_iters == 0:
            wins = " ".join(f"{k[4:]}={v:.2f}" for k, v in record.items() if k.startswith("win_"))
            print(
                f"it {iteration:5d} | {record['samples_per_s']:6.0f} sps | "
                f"ep {int(stats.get('episodes', 0)):3d} {stats['mean_seconds']:3.0f}s | "
                f"pi {metrics.get('policy_loss', 0):+.4f} v {metrics.get('value_loss', 0):.3f} "
                f"ent {metrics.get('entropy', 0):.2f} kl {metrics.get('kl', 0):.4f} "
                f"ev {metrics['explained_variance']:+.2f} | elo {league.elo:6.1f} {wins}",
                flush=True,
            )
        if stats.get("crashes"):
            print(f"  simulator crashes this iteration: {int(stats['crashes'])}; see env.crashes()")

    save_checkpoint(checkpoint_path, model, optimizer, league, config, iteration, samples)
    league.save_index()
    for blue, red in env.crashes()[:20]:
        print("simulator crash:", blue, "vs", red)
    print(f"saved {checkpoint_path} at iteration {iteration}")


if __name__ == "__main__":
    main()
