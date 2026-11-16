"""The optimized collector and update against frozen copies of the originals.

``reference_rollout.py`` and ``reference_ppo.py`` hold the code as it was
before the throughput work. The same seeds must give the same rollout and the
same gradient steps: integers and flags exactly, floats to rounding, because
the optimized path batches rows differently (actor on rows with a choice,
critic over several steps at once).

Rounding can in principle flip one sampled action on a near tie, after which
two rollouts differ for good. The seeds here do not hit such a tie; if a new
platform or PyTorch build does, the first assertion to fail is the action
comparison, at a single row.
"""

import copy
import dataclasses

import numpy as np
import pytest
import torch

import reference_model
import reference_ppo
import reference_rollout
from cfbot.config import DRAFT_CARDS, MODEL_PRESETS, ModelConfig, TrainConfig
from cfbot.decks import DeckStats, resolve_pool
from cfbot.env import Env
from cfbot.league import League
from cfbot import ppo, rollout
from cfbot.model import Embedding, Policy
from cfbot.ppo import PLAY_BUCKET, Ema, _plays, make_optimizer, update
from cfbot.rollout import Collector, padded

STEPS = 24
ENVS = 16


def available(device):
    return device == "cpu" or (device == "mps" and torch.backends.mps.is_available()) or (
        device == "cuda" and torch.cuda.is_available())


@pytest.fixture(params=["cpu", "mps", "cuda"])
def device(request):
    """The CPU, and whichever accelerator the machine has. Several paths exist
    only off the CPU: packed uploads, the rollout kept on the device, fused
    AdamW, and on Apple GPUs the embedding gradient. The CUDA variants have
    never run (no such machine yet): they are the pre-flight for it."""
    if not available(request.param):
        pytest.skip(f"no {request.param} device")
    return request.param


def tolerance(device, cpu, gpu):
    return cpu if device == "cpu" else gpu


def scrambled(spec, config, seed):
    """A policy whose every head gives non-trivial output (several output
    layers start at zero)."""
    torch.manual_seed(seed)
    model = Policy(spec, config.model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
        # Padding rows of embeddings are zero in any trained model.
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding) and module.padding_idx is not None:
                module.weight[module.padding_idx] = 0
    return model


def setup(run_dir, flip, seed=3, device="cpu", **overrides):
    """A collector with every kind of controller: self-play, three league
    slots with different weights, and the three scripted bots."""
    settings = dict(
        run_dir=str(run_dir), num_envs=ENVS, rollout_steps=STEPS, minibatch=96, epochs=1,
        critic_epochs=2, flip_augment=flip, workers=2, device=device, rollout_on_device=True,
        # Float32 everywhere: half precision cannot be compared row for row
        # (a sampled action flips on the first near tie).
        amp=False, model=ModelConfig(**MODEL_PRESETS["small"]),
    )
    config = TrainConfig(**{**settings, **overrides})
    env = Env(ENVS, workers=2, seed=seed, exclude_cards=DRAFT_CARDS,
              deck_pool=resolve_pool("archetypes"), pool_prob=0.3, device=device)
    model = scrambled(env.spec, config, seed).to(device)
    league = League(config, env.spec, torch.device(device))
    for index in range(3):
        league.add_snapshot(scrambled(env.spec, config, 100 + index), index + 1)
    return config, env, model, league


def frozen(model, spec, config):
    """The original network with the weights of ``model``."""
    twin = reference_model.Policy(spec, config.model).to(next(model.parameters()).device)
    twin.load_state_dict(model.state_dict())
    return twin.eval()


def build(cls, run_dir, flip, device="cpu", **overrides):
    """``(config, model, league, deck stats, collector)``. For the original
    collector, the learner and the league opponents are the original network
    too, so the comparison covers the whole pipeline."""
    config, env, model, league = setup(run_dir, flip, device=device, **overrides)
    original = cls is reference_rollout.Collector
    if original:
        model = frozen(model, env.spec, config)
    stats = DeckStats(env.spec["n_cards"])
    # Constructing the slot policies draws from the global generator.
    torch.manual_seed(11)
    collector = cls(config, env, model, league, stats, seed=5)
    collector.taint(league.refresh_slots())
    collector.relayout(0.0)
    if original:
        league.slot_modules = [frozen(slot, env.spec, config) for slot in league.slot_modules]
    assert len(set(collector.opponent.tolist())) == 7  # self-play, 3 slots, 3 bots
    return config, model, league, stats, collector


def assert_same_batch(old, new, device="cpu"):
    for key in ("type", "cell", "done", "train", "choice"):
        assert torch.equal(old[key], new[key]), key
    for key in ("reward", "discount", "logp"):
        assert torch.allclose(old[key], new[key], atol=tolerance(device, 1e-6, 1e-5)), key
    for key in ("value", "advantage", "return"):
        assert torch.allclose(old[key], new[key], atol=2e-5), key
    assert old["value"].abs().max() > 0.01
    for key, value in old["obs"].items():
        assert torch.equal(value.cpu(), new["obs"][key].cpu()), key
    assert torch.equal(new["entity_count"], old["obs"]["entity_mask"].cpu().sum(-1))


@pytest.mark.parametrize("flip", [False, True])
def test_collector_matches_the_original(tmp_path, flip, device):
    old = build(reference_rollout.Collector, tmp_path / "old", flip, device=device)
    new = build(Collector, tmp_path / "new", flip, device=device)
    for iteration in range(2):  # the second rollout starts where the first ended
        torch.manual_seed(20 + iteration)
        old_batch, old_stats = old[4].collect(iteration, shaping=1.0)
        torch.manual_seed(20 + iteration)
        new_batch, new_stats = new[4].collect(iteration, shaping=1.0)
        assert_same_batch(old_batch, new_batch, device)
        assert old_stats.keys() == new_stats.keys()
        for key, value in old_stats.items():
            assert value == pytest.approx(new_stats[key], rel=1e-6, abs=1e-9), key
        forced = new_batch["train"] & ~new_batch["choice"]
        assert forced.any() and (new_batch["logp"][forced] == 0).all()
        assert (new_batch["type"][forced] == 0).all()
    assert_same_state(old, new)
    # Rows that could only wait are never shown to the learner's actor.
    shown = []
    act = new[1].act
    new[1].act = lambda obs, **kwargs: shown.append(obs["legal"].cpu()) or act(obs, **kwargs)
    torch.manual_seed(22)
    batch, _ = new[4].collect(2, shaping=1.0)
    del new[1].act
    learner_rows = int(batch["train"][0].sum())
    assert shown and all(len(legal) <= learner_rows for legal in shown)
    assert all(legal[:, 1:].bool().any(-1).all() for legal in shown)
    assert (batch["train"] & ~batch["choice"]).any()


def assert_same_state(old, new):
    """Everything a collector carries between rollouts, and everything it
    reports to the league and the deck statistics."""
    assert np.array_equal(old[4].learner_side, new[4].learner_side)
    assert np.array_equal(old[4].tainted, new[4].tainted)
    assert old[4].rng.bit_generator.state == new[4].rng.bit_generator.state
    assert torch.equal(old[4].flip_rng.get_state(), new[4].flip_rng.get_state())
    assert old[2].state_dict() == new[2].state_dict()
    old_decks, new_decks = old[3].state_dict(), new[3].state_dict()
    assert np.array_equal(old_decks["card_games"], new_decks["card_games"])
    assert np.array_equal(old_decks["card_score"], new_decks["card_score"])
    assert old_decks["decks"] == new_decks["decks"]
    for key, value in old[4].obs_cpu.items():
        assert torch.equal(value, new[4].obs_cpu[key]), key


def test_collector_matches_the_original_through_finished_games(tmp_path, monkeypatch):
    """A long rollout: battles end and restart, the league and the deck
    statistics record results, sides are redrawn, and the critic needs several
    padded passes over the stored rows."""
    monkeypatch.setattr(rollout, "VALUE_ROWS", 1024)
    old = build(reference_rollout.Collector, tmp_path / "old", True, rollout_steps=220)
    new = build(Collector, tmp_path / "new", True, rollout_steps=220)
    for built in (old, new):
        built[4].tainted[:] = False  # rate the games in progress, as in a settled run
    torch.manual_seed(51)
    old_batch, old_stats = old[4].collect(0, shaping=1.0)
    torch.manual_seed(51)
    new_batch, new_stats = new[4].collect(0, shaping=1.0)
    assert_same_batch(old_batch, new_batch)
    assert new_stats["episodes"] >= 4 and int(new_batch["train"].sum()) > 3 * 1024
    for key, value in old_stats.items():
        assert value == pytest.approx(new_stats[key], rel=1e-6, abs=1e-9), key
    assert_same_state(old, new)
    league = new[2].state_dict()
    rated = sum(entry["games"] for entry in league["baseline_stats"])
    rated += sum(snapshot["games"] for snapshot in league["snapshots"])
    assert rated > 0 and new[3].state_dict()["card_games"].sum() > 0


def test_collector_matches_the_original_without_league_slots(tmp_path):
    """The first iteration of a run: no snapshot yet, scripted bots and
    self-play only."""
    collectors = []
    for cls in (reference_rollout.Collector, Collector):
        config, env, model, _ = setup(tmp_path / cls.__module__, False)
        league = League(config, env.spec, torch.device("cpu"))
        if cls is reference_rollout.Collector:
            model = frozen(model, env.spec, config)
        collectors.append(cls(config, env, model, league, DeckStats(env.spec["n_cards"]), seed=5))
        assert sorted(set(collectors[-1].opponent.tolist())) == [0, 4, 5, 6]
    batches = []
    for collector in collectors:
        torch.manual_seed(61)
        batches.append(collector.collect(0, shaping=1.0)[0])
    assert_same_batch(*batches)


def test_collector_matches_the_original_against_a_fixed_opponent(tmp_path):
    """The exploiter probe: one frozen opponent in every battle, the learner on
    one side of each. Early steps have no learner row with a choice at all."""
    collectors = []
    for cls in (reference_rollout.Collector, Collector):
        config, env, model, _ = setup(tmp_path / cls.__module__, False)
        opponent = scrambled(env.spec, config, 77)
        path = str(tmp_path / f"{cls.__module__}.pt")
        torch.save({"model": opponent.state_dict(), "config": dataclasses.asdict(config)}, path)
        config.fixed_opponent = path
        league = League(config, env.spec, torch.device("cpu"))
        if cls is reference_rollout.Collector:
            model = frozen(model, env.spec, config)
            league.slot_modules[0] = frozen(league.slot_modules[0], env.spec, config)
        collectors.append(cls(config, env, model, league, DeckStats(env.spec["n_cards"]), seed=5))
        assert set(collectors[-1].opponent.tolist()) == {1}
    batches = []
    for collector in collectors:
        torch.manual_seed(71)
        batches.append(collector.collect(0, shaping=1.0)[0])
    assert_same_batch(*batches)
    assert float(batches[1]["train"].float().mean()) == 0.5


def test_rollout_storage_place_does_not_matter(tmp_path, device):
    """Rollout kept on the training device or on the CPU: same batch, same
    update."""
    if device == "cpu":
        pytest.skip("on the CPU the two are one and the same")
    on = build(Collector, tmp_path / "on", False, device=device)
    off = build(Collector, tmp_path / "off", False, device=device)
    off[0].rollout_on_device = False
    torch.manual_seed(7)
    first, _ = on[4].collect(0, shaping=1.0)
    torch.manual_seed(7)
    second, _ = off[4].collect(0, shaping=1.0)
    assert first["obs"]["legal"].device.type == device and second["obs"]["legal"].device.type == "cpu"
    assert_same_batch(first, second, device)
    results = []
    for (config, model, *_), batch in ((on, first), (off, second)):
        net = copy.deepcopy(model)
        results.append(update(net, make_optimizer(net, config), batch, config, 0.01, torch.device(device),
                              np.random.default_rng(4), Ema(net, 0.9)))
    for key, value in results[0].items():
        assert value == pytest.approx(results[1][key], rel=tolerance(device, 1e-5, 2e-4), abs=2e-6), key


def test_update_matches_the_original_when_clipping_is_rare(tmp_path, monkeypatch):
    """With a higher limit some steps clip and some do not. The ones that do
    not skip the scaling by one altogether; the result has to be the same."""
    calls = []
    clip = ppo._clip
    monkeypatch.setattr(ppo, "_clip", lambda params, norm, limit, scale: (calls.append(scale),
                                                                             clip(params, norm, limit, scale)))
    old = build(reference_rollout.Collector, tmp_path / "old", False, max_grad_norm=5.0)
    new = build(Collector, tmp_path / "new", False, max_grad_norm=5.0)
    torch.manual_seed(31)
    old_batch, _ = old[4].collect(0, shaping=1.0)
    torch.manual_seed(31)
    new_batch, _ = new[4].collect(0, shaping=1.0)
    old_net, new_net = copy.deepcopy(old[1]), copy.deepcopy(new[1])
    old_metrics = reference_ppo.update(old_net, reference_ppo.make_optimizer(old_net, old[0]), old_batch, old[0],
                                       0.01, torch.device("cpu"), np.random.default_rng(4))
    new_metrics = update(new_net, make_optimizer(new_net, new[0]), new_batch, new[0], 0.01,
                         torch.device("cpu"), np.random.default_rng(4))
    assert any(scale == 1.0 for scale in calls) and any(scale < 1.0 for scale in calls)
    for key, value in old_metrics.items():
        assert value == pytest.approx(new_metrics[key], rel=2e-4, abs=2e-6), key
    for (name, a), b in zip(old_net.named_parameters(), new_net.parameters()):
        assert torch.allclose(a, b, atol=2e-6), name


@pytest.mark.parametrize("flip", [False, True])
def test_update_matches_the_original(tmp_path, flip, device):
    old = build(reference_rollout.Collector, tmp_path / "old", flip, device=device)
    new = build(Collector, tmp_path / "new", flip, device=device)
    torch.manual_seed(31)
    old_batch, _ = old[4].collect(0, shaping=1.0)
    torch.manual_seed(31)
    new_batch, _ = new[4].collect(0, shaping=1.0)

    results = []
    for module, (config, model, *_), batch in (
        (reference_ppo, old, old_batch), (None, new, new_batch),
    ):
        model = copy.deepcopy(model)
        if module is None:
            optimizer, ema, step = make_optimizer(model, config), Ema(model, 0.9), update
        else:
            optimizer = module.make_optimizer(model, config)
            ema, step = module.Ema(model, 0.9), module.update
        metrics = step(model, optimizer, batch, config, 0.01, torch.device(device),
                       np.random.default_rng(4), ema)
        results.append((metrics, model, ema, optimizer))
    (old_metrics, old_model, old_ema, old_opt), (new_metrics, new_model, new_ema, new_opt) = results
    assert old_metrics["updates"] >= 2 and old_metrics["critic_updates"] >= 2
    assert old_metrics.keys() == new_metrics.keys()
    # On the Apple GPU the new update also runs fused AdamW. A parameter whose
    # true gradient is zero (a bias softmax ignores) then moves by rounding
    # noise in either implementation, up to a learning-rate step per update;
    # everything that matters stays far closer. A real difference in a
    # gradient shows up ten times above these bounds.
    close = tolerance(device, 2e-6, 3e-4)
    for key, value in old_metrics.items():
        assert value == pytest.approx(new_metrics[key], rel=tolerance(device, 2e-4, 5e-4), abs=2e-6), key
    for (name, a), b in zip(old_model.named_parameters(), new_model.parameters()):
        assert torch.allclose(a, b, atol=close), name
    for name, value in old_ema.shadow.items():
        assert torch.allclose(value, new_ema.shadow[name], atol=close), name
    # Same optimizer state: every moment estimate and step count.
    for a, b in zip(old_opt.state_dict()["state"].values(), new_opt.state_dict()["state"].values()):
        assert float(a["step"]) == float(b["step"])
        assert torch.allclose(a["exp_avg"], b["exp_avg"].to(a["exp_avg"].device), atol=tolerance(device, 2e-6, 2e-5))
        # Second moments span many orders of magnitude: compare their roots.
        assert torch.allclose(a["exp_avg_sq"].sqrt(), b["exp_avg_sq"].to(a["exp_avg_sq"].device).sqrt(),
                              rtol=tolerance(device, 1e-3, 2e-2), atol=tolerance(device, 1e-7, 2e-6))


def test_noise_argument_reproduces_a_full_pass(tmp_path):
    """Running only some rows with their slice of the noise picks the same
    actions as running every row."""
    config, env, model, _ = setup(tmp_path, False)
    obs = env.reset(seed=9)
    for _ in range(30):
        obs, _ = env.step(model.act(obs, with_value=False)["flat"])
    rows = obs["legal"].shape[0]
    noise = (torch.rand(rows, 11), torch.rand(rows, 576))
    full = model.act(obs, with_value=False, noise=noise)
    assert (full["type"] > 0).any()
    some = padded(obs["legal"][:, 1:].bool().any(-1).nonzero().squeeze(-1), 8)
    part = model.act({key: value[some] for key, value in obs.items()}, with_value=False,
                     noise=(noise[0][some], noise[1][some]))
    assert torch.equal(part["flat"], full["flat"][some])
    assert torch.allclose(part["logp"], full["logp"][some], atol=1e-5)
    # A row that can only wait is fully determined without the network.
    forced = ~obs["legal"][:, 1:].bool().any(-1)
    assert forced.any()
    assert (full["flat"][forced] == 0).all() and (full["logp"][forced] == 0).all()


def played(env, model, steps=40, seed=9):
    """Observations from the middle of games, so boards hold units."""
    obs = env.reset(seed=seed)
    torch.manual_seed(seed)
    for _ in range(steps):
        obs, _ = env.step(model.act(obs, with_value=False)["flat"])
    return obs


@pytest.fixture(params=["lookup", "matmul"])
def embedding_gradient(request, monkeypatch):
    """Both ways an embedding table can take its gradient. The matrix-product
    way is what Apple GPUs use; here it is switched on for the CPU."""
    if request.param == "matmul":
        monkeypatch.setattr(Embedding, "matmul_devices", ("mps", "cpu"))
    return request.param


def as_on_apple_gpus(model):
    """Make a network's embedding lookups send gradient to their padding rows,
    which is what PyTorch does on Apple GPUs (and nowhere else). The matrix
    product reproduces that, so it is the reference for it."""
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            module.padding_idx = None
    return model


def test_embedding_matmul_gradient_is_the_apple_lookup_gradient(monkeypatch):
    torch.manual_seed(0)
    ids = torch.randint(0, 12, (64, 9))
    ids[:, 5:] = 0  # padding, as in real rows
    scale = torch.randn(64, 9, 16)
    torch.manual_seed(1)
    table = Embedding(12, 16, padding_idx=0)
    with torch.no_grad():
        table.weight[0] = 0.3  # a padding row that has been trained on an Apple GPU
    plain = table(ids)
    (plain * scale).sum().backward()
    padded_gradient = table.weight.grad.clone()
    assert (padded_gradient[0] == 0).all()  # the CPU honours padding_idx

    monkeypatch.setattr(Embedding, "matmul_devices", ("mps", "cpu"))
    table.weight.grad = None
    out = table(ids)
    (out * scale).sum().backward()
    assert torch.equal(out, plain)  # the same rows, bit for bit
    with torch.no_grad():
        assert torch.equal(table(ids), plain)  # acting takes the lookup
    assert torch.allclose(table.weight.grad[1:], padded_gradient[1:], atol=1e-5)
    # The padding row gets the gradient a lookup without padding_idx gives it.
    free = torch.nn.Embedding(12, 16)
    free.load_state_dict(table.state_dict())
    (free(ids) * scale).sum().backward()
    assert torch.allclose(table.weight.grad, free.weight.grad, atol=1e-5)
    assert table.weight.grad[0].abs().max() > 0


def test_network_matches_the_original(tmp_path, embedding_gradient):
    """Same weights, same inputs: the current network and the frozen original
    agree on every output and on every gradient."""
    config, env, model, _ = setup(tmp_path, False)
    twin = frozen(model, env.spec, config)
    if embedding_gradient == "matmul":
        twin = as_on_apple_gpus(twin)
    assert [name for name, _ in model.named_parameters()] == [name for name, _ in twin.named_parameters()]
    assert model.state_dict().keys() == twin.state_dict().keys()
    obs = played(env, model)
    rows = obs["legal"].shape[0]
    assert obs["entity_mask"].sum() > 4 * rows and obs["play_mask"].sum() > 0

    model.eval()
    torch.manual_seed(1)
    old = twin.act(obs)
    torch.manual_seed(1)
    new = model.act(obs)
    assert torch.equal(old["flat"], new["flat"]) and (new["type"] > 0).any()
    assert torch.allclose(old["logp"], new["logp"], atol=1e-5)
    assert torch.allclose(old["value"], new["value"], atol=1e-5) and old["value"].abs().max() > 0.01
    old_full, old_public = twin.full_log_probs(obs)
    new_full, new_public = model.full_log_probs(obs)
    legal = obs["legal"].bool()
    assert torch.equal(torch.isinf(old_full), torch.isinf(new_full))
    assert torch.allclose(old_full[legal], new_full[legal], atol=1e-5)
    assert torch.allclose(old_public, new_public, atol=1e-5)

    # The training pass, and the gradient of a loss that touches every head.
    # The current network scores cells on the play rows only; the original
    # scored every row and threw the rest away.
    plays = _plays(new["type"], torch.device("cpu"))
    assert 0 < plays[1].sum() < len(plays[1]) == min(PLAY_BUCKET, rows)
    grads = []
    for net in (twin, model, model):
        net.train()
        net.zero_grad(set_to_none=True)
        if net is twin or len(grads) == 2:
            out = net.evaluate(obs, new["type"], new["cell"])
        else:
            out = net.evaluate(obs, new["type"], new["cell"], plays)
        weight = torch.linspace(0.5, 1.5, rows)
        loss = sum(
            (value.reshape(rows, -1).mean(-1) * weight).sum() * scale
            for scale, value in zip((1.0, 0.7, 1.3, 0.9, 1.1, 0.8, 1.2),
                                    (out[key] for key in sorted(out)))
        )
        loss.backward()
        grads.append((out, {name: p.grad for name, p in net.named_parameters()}))
    (old_out, old_grad), with_plays, every_row = grads
    assert (old_out["cell_entropy"] > 0).any()
    for new_out, new_grad in (with_plays, every_row):
        for key, value in old_out.items():
            assert torch.allclose(value, new_out[key], atol=2e-5), key
        for name, value in old_grad.items():
            if value is None:
                # Parameters nothing reads get no gradient, so AdamW leaves
                # them alone. That has to stay so.
                assert new_grad[name] is None, name
                continue
            assert new_grad[name] is not None, name
            scale = float(value.abs().max())
            assert torch.allclose(value, new_grad[name], atol=1e-6 + 2e-4 * scale), name
    # With the lookup, padding rows of the embedding tables get no gradient.
    # (Some get one with the matrix product, as on Apple GPUs; the comparison
    # above covered those against a reference that does the same.)
    padding = [module.weight.grad[0] for module in model.modules()
               if isinstance(module, Embedding) and module.padding_idx is not None]
    if embedding_gradient == "lookup":
        assert all((grad == 0).all() for grad in padding)
    else:
        assert any(grad.abs().max() > 0 for grad in padding)


def test_network_matches_the_original_on_the_apple_gpu(tmp_path):
    """On the device itself: outputs and every gradient of the current network
    against the frozen original, padding rows included."""
    if not torch.backends.mps.is_available():
        pytest.skip("needs an Apple GPU")
    device = torch.device("mps")
    config, env, model, _ = setup(tmp_path, False)
    obs = {key: value.to(device) for key, value in played(env, model).items()}
    twin = frozen(model, env.spec, config).to(device)
    model = model.to(device)
    rows = obs["legal"].shape[0]
    with torch.no_grad():
        chosen = model.act(obs, with_value=False)
    plays = _plays(chosen["type"].cpu(), device)
    grads = []
    for net in (twin, model):
        net.train()
        net.zero_grad(set_to_none=True)
        if net is twin:
            out = net.evaluate(obs, chosen["type"], chosen["cell"])
        else:
            out = net.evaluate(obs, chosen["type"], chosen["cell"], plays)
        weight = torch.linspace(0.5, 1.5, rows, device=device)
        loss = sum(
            (value.reshape(rows, -1).mean(-1) * weight).sum() * scale
            for scale, value in zip((1.0, 0.7, 1.3, 0.9, 1.1, 0.8, 1.2),
                                    (out[key] for key in sorted(out)))
        )
        loss.backward()
        grads.append((out, {name: p.grad for name, p in net.named_parameters()}))
    (old_out, old_grad), (new_out, new_grad) = grads
    for key, value in old_out.items():
        assert torch.allclose(value, new_out[key], atol=5e-5), key
    for name, value in old_grad.items():
        assert (value is None) == (new_grad[name] is None), name
        if value is not None:
            scale = float(value.abs().max())
            assert torch.allclose(value, new_grad[name], atol=2e-6 + 1e-3 * scale), name
    # The backend quirk the matrix product has to keep: a padding row with a
    # gradient.
    assert old_grad["embed.tower_emb.weight"][0].abs().max() > 0
    assert new_grad["embed.tower_emb.weight"][0].abs().max() > 0


def test_placement_head_still_gets_a_gradient_without_plays(tmp_path):
    """A minibatch with no play at all must leave the placement head with a
    zero gradient, not with none: AdamW skips a parameter that has none, and
    the original always stepped it."""
    config, env, model, _ = setup(tmp_path, False)
    obs = played(env, model)
    rows = obs["legal"].shape[0]
    waits = torch.zeros(rows, dtype=torch.int64)
    model.train()
    model.zero_grad(set_to_none=True)
    out = model.evaluate(obs, waits, waits, _plays(waits, torch.device("cpu")))
    (out["logp"].sum() + out["cell_entropy"].sum()).backward()
    assert (out["cell_entropy"] == 0).all()
    grad = model.place_out.weight.grad
    assert grad is not None and (grad == 0).all()


def test_optimizer_state_survives_a_change_of_implementation(tmp_path, monkeypatch):
    """A checkpoint written with one AdamW implementation resumes under the
    other: the live optimizer keeps its own switches, the step counters end up
    where that implementation wants them, and the next steps agree."""
    config, env, model, _ = setup(tmp_path, False)
    obs = played(env, model)

    def step(net, optimizer):
        optimizer.zero_grad(set_to_none=True)
        out = net.evaluate(obs, torch.zeros(obs["legal"].shape[0], dtype=torch.int64),
                           torch.zeros(obs["legal"].shape[0], dtype=torch.int64))
        (out["logp"].sum() + out["value_private"].logsumexp(-1).sum()).backward()
        optimizer.step()

    def fresh(fused):
        monkeypatch.setattr(ppo, "fused_optimizer", lambda *_: fused)
        net = copy.deepcopy(model).train()
        return net, make_optimizer(net, config)

    for saved_fused, live_fused in ((False, True), (True, False)):
        source, source_opt = fresh(saved_fused)
        assert all(bool(group["fused"]) == saved_fused for group in source_opt.param_groups)
        for _ in range(3):
            step(source, source_opt)
        state = copy.deepcopy(source_opt.state_dict())
        target, target_opt = fresh(live_fused)
        target.load_state_dict(source.state_dict())
        ppo.load_optimizer(target_opt, state)
        assert all(bool(group["fused"]) == live_fused for group in target_opt.param_groups)
        for entry in target_opt.state_dict()["state"].values():
            assert float(entry["step"]) == 3
        step(source, source_opt)
        step(target, target_opt)
        for (name, a), b in zip(source.named_parameters(), target.parameters()):
            assert torch.allclose(a, b, atol=1e-6), name


def test_compiled_passes_leave_the_model_alone(tmp_path):
    """``compile`` wraps the two training passes, never the model: same
    parameter names, same checkpoint keys, same update."""
    new = build(Collector, tmp_path / "new", False)
    config, model = new[0], new[1]
    torch.manual_seed(41)
    batch, _ = new[4].collect(0, shaping=1.0)
    keys = list(model.state_dict())
    results = []
    for compiled in (False, True):
        net = copy.deepcopy(model)
        settings = copy.copy(config)
        settings.compile, settings.compile_backend = compiled, "aot_eager"
        passes = ppo.Passes(net, settings)
        assert (passes.fast["evaluate"] is not passes.plain["evaluate"]) == compiled
        metrics = update(net, make_optimizer(net, settings), batch, settings, 0.01,
                         torch.device("cpu"), np.random.default_rng(4), Ema(net, 0.9), passes)
        assert list(net.state_dict()) == keys
        assert all("_orig_mod" not in name for name, _ in net.named_parameters())
        results.append((metrics, net, passes))
    (plain_metrics, plain_net, _), (fast_metrics, fast_net, passes) = results
    # Still compiled after the update: nothing fell back to the plain pass.
    assert passes.fast["evaluate"] is not passes.plain["evaluate"]
    assert passes.fast["critic"] is not passes.plain["critic"]
    for key, value in plain_metrics.items():
        assert value == pytest.approx(fast_metrics[key], rel=1e-3, abs=1e-5), key
    for (name, a), b in zip(plain_net.named_parameters(), fast_net.parameters()):
        assert torch.allclose(a, b, atol=1e-5), name


def test_packed_upload_is_the_same_observation(device):
    """Sending a step's planes as two blocks changes how they travel, not what
    arrives. Needs a device other than the CPU, where nothing is sent."""
    if device == "cpu":
        pytest.skip("no accelerator to upload to")
    env = Env(4, workers=1, seed=2, exclude_cards=DRAFT_CARDS, device=device)
    env.reset(seed=3)
    packed = env.upload(env.last_cpu, packed=True)
    separate = env.upload(env.last_cpu)
    assert packed.keys() == separate.keys() == env.last_cpu.keys()
    for key, value in env.last_cpu.items():
        assert packed[key].dtype == value.dtype and packed[key].shape == value.shape
        assert torch.equal(packed[key].cpu(), value) and torch.equal(separate[key].cpu(), value)


def test_bridge_summaries_match_the_planes(tmp_path):
    """The bridge's per-row ``choice`` and ``entity_count`` are what the
    trainer used to compute from the legal mask and the entity mask."""
    env = Env(ENVS, workers=2, seed=6, exclude_cards=DRAFT_CARDS)
    obs = env.reset(seed=4)
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(150):
        assert torch.equal(env.last_choice, obs["legal"][:, 1:].bool().any(-1))
        assert torch.equal(env.last_entity_count, obs["entity_mask"].sum(-1))
        seen.update(env.last_choice.tolist())
        legal = obs["legal"].numpy()
        actions = np.zeros(legal.shape[0], dtype=np.int64)
        for row in range(legal.shape[0]):
            options = np.nonzero(legal[row])[0]
            if len(options) > 1 and rng.random() < 0.3:
                actions[row] = rng.choice(options[1:])
        obs, _ = env.step(torch.from_numpy(actions))
    assert seen == {True, False}


def test_mixed_precision_path_runs_and_agrees(tmp_path, monkeypatch):
    """NVIDIA GPUs run the networks under bfloat16 autocast, which no machine
    here has. Autocast on the CPU is a weaker stand-in: it keeps normalization
    and softmax in bfloat16 where CUDA returns float32, so it shows that the
    changed code runs in half precision and agrees with the original to
    half-precision rounding (not to float32 rounding: the cross-attention
    rounds in different places), not that every CUDA cast is right. The test
    after this one uses the Apple GPU's autocast, whose rules match CUDA's."""
    half = lambda config, device: torch.autocast("cpu", dtype=torch.bfloat16)  # noqa: E731
    monkeypatch.setattr(ppo, "autocast", half)
    monkeypatch.setattr(rollout, "autocast", half)
    config, model, _, _, collector = build(Collector, tmp_path / "new", False)
    torch.manual_seed(81)
    batch, _ = collector.collect(0, shaping=1.0)
    for key in ("logp", "value", "advantage", "return"):
        assert batch[key].dtype == torch.float32 and torch.isfinite(batch[key]).all(), key
    assert batch["value"].abs().max() > 0.01

    step = STEPS - 1
    obs = {key: value[step] for key, value in batch["obs"].items()}
    kind, cell = batch["type"][step], batch["cell"][step]
    twin = frozen(model, collector.env.spec, config).train()
    net = copy.deepcopy(model).train()
    with half(config, None):
        old = twin.evaluate(obs, kind, cell)
        new = net.evaluate(obs, kind, cell, _plays(kind, torch.device("cpu")))
        old_value, new_value = twin.critic(obs), net.critic(obs)
    for key, value in old.items():
        assert new[key].dtype == torch.float32 and torch.isfinite(new[key]).all(), key
        assert torch.allclose(value, new[key], atol=5e-2), key
    assert torch.allclose(old_value, new_value, atol=5e-2)
    sum(value.sum() for value in new.values()).backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)

    metrics = update(net, make_optimizer(net, config), batch, config, 0.01, torch.device("cpu"),
                     np.random.default_rng(4), Ema(net, 0.9))
    assert metrics["updates"] > 0 and metrics["skipped"] == 0
    assert all(np.isfinite(value) for value in metrics.values())


def test_mixed_precision_casts_like_an_nvidia_gpu(tmp_path, monkeypatch):
    """Apple-GPU autocast casts the way CUDA's does: linear, matmul and
    convolution to bfloat16, normalization, softmax and sums back to float32.
    So the mixing of the two that the NVIDIA run will see can be run here."""
    if not torch.backends.mps.is_available():
        pytest.skip("needs an Apple GPU")
    device = torch.device("mps")
    config, env, model, _ = setup(tmp_path, False)
    obs = {key: value.to(device) for key, value in played(env, model).items()}
    twin = frozen(model, env.spec, config).to(device).train()
    monkeypatch.setattr(Embedding, "matmul_devices", ())  # CUDA takes the plain lookup
    net = copy.deepcopy(model).to(device).train()
    with torch.autocast("mps", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("mps")
        with torch.no_grad():
            chosen = net.act(obs)
        plays = _plays(chosen["type"].cpu(), device)
        old = twin.evaluate(obs, chosen["type"], chosen["cell"])
        new = net.evaluate(obs, chosen["type"], chosen["cell"], plays)
    assert obs["legal"].gather(1, chosen["flat"][:, None]).bool().all()
    for key in ("logp", "value"):
        assert chosen[key].dtype == torch.float32 and torch.isfinite(chosen[key]).all(), key
    for key, value in old.items():
        assert new[key].dtype == torch.float32 and torch.isfinite(new[key]).all(), key
        assert torch.allclose(value, new[key], atol=5e-2), key
    sum(value.sum() for value in new.values()).backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_first_run_on_an_nvidia_gpu(tmp_path):
    """Collection and an update as the real run will do them: bfloat16
    autocast, fused AdamW, the rollout kept on the GPU. Never run so far."""
    if not torch.cuda.is_available():
        pytest.skip("needs an NVIDIA GPU")
    config, model, _, _, collector = build(Collector, tmp_path / "run", False, device="cuda", amp=True,
                                           rollout_on_device=None)
    batch, stats = collector.collect(0, shaping=1.0)
    assert stats["rejected"] == 0 and batch["obs"]["legal"].device.type == "cuda"
    for key in ("logp", "value", "advantage", "return"):
        assert batch[key].dtype == torch.float32 and torch.isfinite(batch[key]).all(), key
    optimizer = make_optimizer(model, config)
    assert all(group["fused"] for group in optimizer.param_groups)
    metrics = update(model, optimizer, batch, config, 0.01, torch.device("cuda"),
                     np.random.default_rng(4), Ema(model, 0.9))
    assert metrics["updates"] > 0 and metrics["skipped"] == 0 and metrics["gpu_mem_gb"] > 0
    assert all(np.isfinite(value) for value in metrics.values())
