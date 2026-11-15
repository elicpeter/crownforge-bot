"""Time one PPO minibatch (forward, backward, optimizer step) per preset."""
import sys
import time

import torch

from cfbot.config import MODEL_PRESETS, ModelConfig, TrainConfig, pick_device
from cfbot.env import Env
from cfbot.league import League
from cfbot.model import Policy
from cfbot.ppo import make_optimizer
from cfbot.rollout import Collector

device = pick_device()
presets = sys.argv[1].split(",") if len(sys.argv) > 1 else ["small", "base"]
sizes = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1024, 4096]


def sync():
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


for preset in presets:
    config = TrainConfig(run_dir="runs/profile", num_envs=128, rollout_steps=32,
                         model=ModelConfig(**MODEL_PRESETS[preset]))
    env = Env(config.num_envs, device=device, exclude_cards=config.exclude_cards)
    model = Policy(env.spec, config.model).to(device)
    collector = Collector(config, env, model, League(config, env.spec, device))
    start = time.time()
    batch, stats = collector.collect(0, 1.0)
    sync()
    print(f"{preset}: collect 32 steps x {env.rows} rows {time.time() - start:.1f}s, "
          f"entity width {stats['entities']}", flush=True)
    optimizer = make_optimizer(model, config)
    model.train()
    flat = {k: v.flatten(0, 1) for k, v in batch["obs"].items()}
    for m in sizes:
        obs = {k: v[:m].to(device) for k, v in flat.items()}
        kind, cell = (batch[k].flatten()[:m].to(device) for k in ("type", "cell"))
        for _ in range(3):
            sync(); t0 = time.time()
            out = model.evaluate(obs, kind, cell)
            sync(); t1 = time.time()
            loss = (out["logp"].mean() + out["value_private"].logsumexp(-1).mean()
                    + out["value_public"].logsumexp(-1).mean()
                    + out["type_entropy"].mean() + out["cell_entropy"].mean() + out["aux_hand"].mean())
            optimizer.zero_grad()
            loss.backward()
            sync(); t2 = time.time()
            optimizer.step()
            sync(); t3 = time.time()
            memory = f"  gpu mem {torch.mps.driver_allocated_memory() / 1e9:.1f}GB" if device.type == "mps" else ""
            print(f"  {m} decisions: forward {t1 - t0:.2f}s backward {t2 - t1:.2f}s "
                  f"step {t3 - t2:.2f}s -> {m / (t3 - t0):.0f}/s{memory}", flush=True)
