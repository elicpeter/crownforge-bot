"""Where one training iteration spends its time.

    python scripts/profile_iteration.py                    # base preset, 256 x 128
    python scripts/profile_iteration.py --iters 3 --warmup 2
    python scripts/profile_iteration.py --set num_envs=512 --set rollout_steps=64

Builds the same objects as ``cfbot.train`` (league slots loaded with three
snapshots, scripted bots in place), then reports two things:

* plain timings of whole iterations, with nothing instrumented;
* a split of collection and of the update, taken with the device synchronized
  around every probed call so that queued GPU work is charged to the call that
  queued it. The synchronized run is slower than the plain one; read the split
  as shares, and the plain numbers as the truth.

The probes are attached from outside (no hooks in the training code), by name.
A name that does not exist in this version of the code is skipped.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from cfbot import ppo, rollout
from cfbot.config import MODEL_PRESETS, ModelConfig, TrainConfig, override, pick_device
from cfbot.decks import DeckStats, resolve_pool
from cfbot.env import Env, make_env, simulator_of
from cfbot.league import League
from cfbot.model import Policy, count_parameters
from cfbot.ppo import Ema, make_optimizer, schedule
from cfbot.rollout import Collector


class Probe:
    """Exclusive wall time per label, measured around named callables and
    grouped by the outermost open section."""

    def __init__(self, device: torch.device):
        self.device = device
        self.enabled = False
        self.total: dict[tuple[str, str], float] = defaultdict(float)
        self.calls: dict[tuple[str, str], int] = defaultdict(int)
        self.stack: list[list] = []  # [label, seconds spent in probed children]
        self.context = ""

    def sync(self) -> None:
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()

    def record(self, label: str, elapsed: float, child: float = 0.0) -> None:
        root = self.stack[0][0] if self.stack else label
        self.total[root, label] += elapsed - child
        self.calls[root, label] += 1
        if self.stack:
            self.stack[-1][1] += elapsed

    def section(self, label: str):
        probe = self

        class Section:
            def __enter__(self):
                probe.sync()
                self.start = time.perf_counter()
                probe.stack.append([label, 0.0])

            def __exit__(self, *_):
                probe.sync()
                elapsed = time.perf_counter() - self.start
                _, child = probe.stack.pop()
                probe.record(label, elapsed, child)

        return Section()

    def wrap(self, owner, name: str, label: str, top_only: bool = False, context: str | None = None) -> bool:
        """Time ``owner.name``. ``top_only`` ignores calls made inside another
        probed call. A ``context`` given here tags later labels that contain
        ``{ctx}``, when this call is made directly from the outer section."""
        if owner is None or not hasattr(owner, name):
            return False
        original = getattr(owner, name)
        probe = self

        def timed(*args, **kwargs):
            if not probe.enabled or (top_only and len(probe.stack) > 1):
                return original(*args, **kwargs)
            if context is not None and len(probe.stack) <= 1:
                probe.context = context
            with probe.section(label.format(ctx=probe.context)):
                return original(*args, **kwargs)

        setattr(owner, name, timed)
        return True

    def wrap_transfers(self) -> None:
        """Time tensors that change device, and reads that force a sync."""
        probe = self

        def moving(original):
            def timed(tensor, *args, **kwargs):
                if not probe.enabled:
                    return original(tensor, *args, **kwargs)
                probe.sync()  # work already queued belongs to the caller
                start = time.perf_counter()
                out = original(tensor, *args, **kwargs)
                if out.device == tensor.device:
                    return out
                probe.sync()
                label = "transfer: device to cpu" if out.device.type == "cpu" else "transfer: cpu to device"
                probe.record(label, time.perf_counter() - start)
                return out

            return timed

        def reading(original):
            def timed(tensor, *args, **kwargs):
                if not probe.enabled or tensor.device.type == "cpu":
                    return original(tensor, *args, **kwargs)
                probe.sync()
                start = time.perf_counter()
                out = original(tensor, *args, **kwargs)
                probe.record("transfer: scalar readback", time.perf_counter() - start)
                return out

            return timed

        for name in ("to", "cpu"):
            setattr(torch.Tensor, name, moving(getattr(torch.Tensor, name)))
        for name in ("__float__", "__bool__", "__int__", "item", "tolist"):
            setattr(torch.Tensor, name, reading(getattr(torch.Tensor, name)))

    def report(self, title: str, root: str, per: float, unit: str, iters: int) -> None:
        labels = sorted((k[1] for k in self.total if k[0] == root and k[1] != root),
                        key=lambda label: -self.total[root, label])
        whole = sum(self.total[root, label] for label in labels) + self.total[root, root]
        per = max(1.0, per)
        print(f"\n{title}: {whole / iters:.2f} s per iteration synchronized, "
              f"{1e3 * whole / iters / per:.1f} ms per {unit}")
        rows = [(label, self.total[root, label], self.calls[root, label]) for label in labels]
        rows.append(("python and untracked", self.total[root, root], 0))
        for label, seconds, calls in rows:
            each = f"{calls / iters / per:6.1f} calls/{unit}" if calls else ""
            print(f"  {label:34s} {seconds / iters:7.2f} s {100 * seconds / whole:5.1f}%  "
                  f"{1e3 * seconds / iters / per:7.2f} ms/{unit} {each}")


class StepProxy:
    """Stands in for the Rust environment so its ``step`` can be timed."""

    def __init__(self, inner, probe: Probe):
        self._inner, self._probe = inner, probe

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def step(self, *args):
        if not self._probe.enabled:
            return self._inner.step(*args)
        with self._probe.section("simulator: bridge step"):
            return self._inner.step(*args)


class Null:
    def __enter__(self):
        return None

    def __exit__(self, *_):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="base", choices=sorted(MODEL_PRESETS))
    parser.add_argument("--warmup", type=int, default=2, help="iterations before any timing")
    parser.add_argument("--iters", type=int, default=2, help="iterations per timing mode")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a TrainConfig field")
    parser.add_argument("--no-split", action="store_true", help="plain timings only")
    args = parser.parse_args()

    run_dir = tempfile.mkdtemp(prefix="cfbot-profile-")
    config = TrainConfig(run_dir=run_dir, model=ModelConfig(**MODEL_PRESETS[args.preset]))
    for item in args.set or []:
        override(config, item)
    device = pick_device(config.device)
    torch.manual_seed(config.seed)

    env = make_env(
        config.num_envs, **simulator_of(config), workers=config.workers, decision_ticks=config.decision_ticks, seed=config.seed,
        max_entities=config.max_entities, deck_pool=resolve_pool(config.deck_pool, config.simulator),
        pool_prob=config.pool_prob, mirror_decks=config.mirror_decks, max_skip=config.max_skip,
        exclude_cards=config.exclude_cards, rules=config.rules, level_jitter=config.level_jitter,
        device=device,
    )
    model = Policy(env.spec, config.model).to(device)
    optimizer = make_optimizer(model, config)
    league = League(config, env.spec, device)
    ema = Ema(model, config.ema_decay) if config.ema_decay > 0 else None
    collector = Collector(config, env, model, league, DeckStats(env.spec["n_cards"]), seed=config.seed)
    rng = np.random.default_rng(config.seed)
    # Three snapshots, so every league slot holds its own opponent as in a
    # long run.
    for index in range(3):
        league.add_snapshot(model, index + 1)
    collector.taint(league.refresh_slots())
    collector.relayout(0.0)
    print(f"device {device}, preset {args.preset} ({count_parameters(model) / 1e6:.1f}M parameters), "
          f"{config.num_envs} battles x {config.rollout_steps} steps, torch {torch.__version__}")

    probe = Probe(device)

    def iteration(index: int) -> dict:
        env.set_mix(config.pool_prob, config.mutant_prob)
        lr, entropy_coef = schedule(config, 0.0, index)
        for group in optimizer.param_groups:
            group["lr"] = lr
        probe.sync()
        probe.context = ""
        start = time.perf_counter()
        with probe.section("collect") if probe.enabled else Null():
            batch, stats = collector.collect(index, config.shaping_coef)
        probe.sync()
        probe.context = ""
        middle = time.perf_counter()
        with probe.section("update") if probe.enabled else Null():
            # Built here, after the probes are attached, so it wraps the
            # probed passes; with compile set it reuses the compiled code.
            metrics = ppo.update(model, optimizer, batch, config, entropy_coef, device, rng, ema,
                                 ppo.Passes(model, config))
        probe.sync()
        end = time.perf_counter()
        decisions = int(batch["train"].sum())
        return {
            "collect_s": middle - start, "update_s": end - middle, "decisions": decisions,
            "per_s": decisions / (end - start), "updates": metrics["updates"],
            "critic_updates": metrics["critic_updates"], "entities": stats["entities"],
            "forced_wait": stats["forced_wait"], "train_frac": stats["train_frac"],
            "entropy": metrics.get("entropy", 0.0), "value_loss": metrics.get("value_loss", 0.0),
        }

    def show(tag: str, r: dict) -> None:
        print(f"  {tag}: collect {r['collect_s']:6.2f} s  update {r['update_s']:6.2f} s  "
              f"{r['decisions']:6d} decisions  {r['per_s']:6.0f}/s | {r['updates']}+{r['critic_updates']} steps, "
              f"width {r['entities']}, forced wait {r['forced_wait']:.2f}, entropy {r['entropy']:.5f}, "
              f"value loss {r['value_loss']:.5f}", flush=True)

    count = 0
    for _ in range(args.warmup):
        show("warm-up", iteration(count))
        count += 1
    plain = []
    for _ in range(args.iters):
        plain.append(iteration(count))
        show("plain  ", plain[-1])
        count += 1
    if plain:
        mean = {key: float(np.mean([r[key] for r in plain])) for key in ("collect_s", "update_s", "decisions")}
        print(f"plain mean: collect {mean['collect_s']:.2f} s, update {mean['update_s']:.2f} s, "
              f"{mean['decisions'] / (mean['collect_s'] + mean['update_s']):.0f} decisions/s")
    if args.no_split:
        shutil.rmtree(run_dir, ignore_errors=True)
        return

    # Probes. Names from older and newer versions of the code are both listed.
    probe.wrap_transfers()
    env.env = StepProxy(env.env, probe)
    probe.wrap(env, "_obs", "obs: wrap arrays as tensors")
    probe.wrap(env, "_observe", "obs: copy blocks, wrap as tensors")
    probe.wrap(env, "_refill", "decks: sample next matches")
    probe.wrap(model, "act", "learner: actor forward")
    probe.wrap(model.critic, "forward", "critic forward{ctx}", context=" [critic step]")
    for module in league.slot_modules:
        probe.wrap(module, "act", "opponents: actor forward")
    probe.wrap(league, "act_slots", "opponents: actor forward")
    for bot in league.baselines:
        probe.wrap(bot, "act", "scripted bots")
    probe.wrap(rollout, "select_rows", "row selection on device")
    probe.wrap(rollout, "tower_potential", "tower potential")
    probe.wrap(rollout, "observed_towers", "tower potential")
    probe.wrap(rollout, "gae", "advantage estimation")
    probe.wrap(collector, "_values", "critic: values over the rollout")
    probe.wrap(torch.Tensor, "index_select", "row selection on device", top_only=True)
    probe.wrap(torch, "rand", "sampling noise", top_only=True)
    probe.wrap(collector, "_stack", "stack and pad the rollout")
    probe.wrap(torch, "stack", "stack and pad the rollout", top_only=True)
    probe.wrap(F, "pad", "stack and pad the rollout", top_only=True)

    probe.wrap(ppo, "_gather", "gather and trim minibatch")
    probe.wrap(ppo, "gather_rows", "gather and trim minibatch")
    probe.wrap(model, "evaluate", "forward: actor and heads", context=" [actor step]")
    probe.wrap(torch.Tensor, "backward", "backward{ctx}")
    probe.wrap(torch.nn.utils, "clip_grad_norm_", "gradient norms and clipping")
    probe.wrap(ppo, "_norm", "gradient norms and clipping")
    probe.wrap(ppo, "_clip", "gradient norms and clipping")
    probe.wrap(optimizer, "step", "optimizer step")
    probe.wrap(optimizer, "zero_grad", "zero_grad")
    probe.wrap(ema, "step", "ema step")

    probe.enabled = True
    split = []
    for _ in range(args.iters):
        split.append(iteration(count))
        show("split  ", split[-1])
        count += 1
    probe.enabled = False
    gradient_steps = float(np.mean([r["updates"] + r["critic_updates"] for r in split]))
    probe.report("collection", "collect", config.rollout_steps, "step", args.iters)
    probe.report("update", "update", gradient_steps, "gradient step", args.iters)
    shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
