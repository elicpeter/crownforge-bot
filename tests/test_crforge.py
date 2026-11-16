"""The CRForge simulator behind the bot's environment interface.

Skipped unless CRForge is set up: ``CRFORGE_INSTALL`` and ``CRFORGE_TABLES``
name its installed env distribution and a data version's game tables, the
bridge class is built (``scripts/build_crforge.sh``) and ``jpype1`` is there.
"""

import os

import numpy as np
import pytest
import torch

from cfbot.config import DRAFT_CARDS, MODEL_PRESETS, ModelConfig, TrainConfig
from cfbot.decks import ARCHETYPES, DeckStats, resolve_pool
from cfbot.env import POLICY_KEYS, card_ids, make_env, observed_towers, simulator_of
from cfbot.evaluate import play_paired
from cfbot.league import League
from cfbot.model import Policy
from cfbot.ppo import Ema, make_optimizer, update
from cfbot.rollout import Collector

pytest.importorskip("jpype")
if not (os.environ.get("CRFORGE_INSTALL") and os.environ.get("CRFORGE_TABLES")):
    pytest.skip("CRFORGE_INSTALL and CRFORGE_TABLES are not set", allow_module_level=True)

SIM = simulator_of({"simulator": "crforge"})


@pytest.fixture(scope="module")
def env():
    return make_env(8, **SIM, workers=2, seed=3, exclude_cards=DRAFT_CARDS)


@pytest.fixture(scope="module")
def model(env):
    torch.manual_seed(0)
    return Policy(env.spec, ModelConfig(**MODEL_PRESETS["small"]))


def only_mirror_refused(env):
    # CRForge declines to model a Mirror given while a play of its side is
    # waiting to run, and refuses it; the mask offers nothing else it refuses.
    return all(r["reason"] == "NOT_MODELLED" and r["card"] == "Mirror" for r in env.refusals())


def random_actions(obs, rng, prob=0.3):
    legal = obs["legal"].numpy()
    actions = np.zeros(legal.shape[0], dtype=np.int64)
    for row in range(legal.shape[0]):
        options = np.nonzero(legal[row])[0]
        if len(options) > 1 and rng.random() < prob:
            actions[row] = rng.choice(options[1:])
    return torch.from_numpy(actions)


def test_observation_has_the_bots_planes(env):
    spec = env.spec
    obs = env.reset(seed=7)
    assert spec["simulator"] == "crforge" and spec["schema_version"] == 2
    assert obs["global"].shape == (env.rows, 4)
    assert obs["entities"].shape[2] == 24 and obs["entity_extras"].shape[2] == 5
    assert obs["hand"].shape == (env.rows, 4, 4)
    assert obs["own_deck"].shape == (env.rows, 8, 2)
    assert obs["recent_plays"].shape == (env.rows, 32, 7)
    assert obs["revealed_deck"].shape == (env.rows, 8, 3)
    assert obs["legal"].shape == (env.rows, 4611)
    # Six towers a side at full health, and nothing else on the field.
    assert torch.allclose(observed_towers(obs), torch.ones(env.rows, 6))
    assert (env.last_entity_count == 6).all()
    # Secondary plays do not exist here.
    assert not obs["legal"][:, 2307:].any()
    assert spec["card_table"].shape == (spec["n_cards"], spec["card_f"])
    assert spec["unit_table"].shape == (spec["n_units"], spec["unit_f"])


def test_every_archetype_card_is_played(env):
    spec = env.spec
    supported = dict(zip(spec["card_names"], spec["card_supported"]))
    for deck in ARCHETYPES.values():
        for card in deck:
            assert supported.get(card), card
    pooled = make_env(8, **SIM, workers=2, seed=1, deck_pool=resolve_pool("archetypes"), pool_prob=1.0)
    obs = pooled.reset(seed=2)
    assert pooled.deck_retries() == 0
    decks = {tuple(sorted(card_ids(spec, deck))) for deck in ARCHETYPES.values()}
    for row in obs["own_deck"][..., 0].long().tolist():
        assert tuple(sorted(row)) in decks


def test_forced_waits_are_skipped(env):
    env.reset(seed=13)
    # Nobody can play during the opening lockout, so the first step runs long.
    _, info = env.step(torch.zeros(env.rows, dtype=torch.int64))
    assert (info["ticks"] > env.spec["decision_ticks"]).all()


def test_tokens_stay_in_their_vocabularies_and_units_are_sorted(env):
    spec = env.spec
    obs = env.reset(seed=4)
    rng = np.random.default_rng(0)
    seen_hazard = False
    for _ in range(300):
        obs, info = env.step(random_actions(obs, rng))
        entities = obs["entities"]
        assert torch.isfinite(entities).all() and torch.isfinite(obs["recent_plays"]).all()
        assert entities[..., 2].max() < spec["n_units"] and obs["hazards"][..., 0].max() < spec["n_hazards"]
        assert obs["hand"][..., :2].max() < spec["n_cards"]
        assert ((entities[..., 4:6] >= 0) & (entities[..., 4:6] <= 1)).all()
        for row in range(env.rows):
            n = int(env.last_entity_count[row])
            rows = [tuple(values) for values in torch.cat([entities[row, :n], obs["entity_extras"][row, :n]], 1).tolist()]
            assert rows == sorted(rows)
            assert not entities[row, n:].any() and int(obs["entity_mask"][row].sum()) == n
        seen_hazard |= bool(obs["hazard_mask"].any())
        assert (info["verdict"] < 3).all() and only_mirror_refused(env)
        assert (info["reward"].reshape(-1, 2).sum(-1) == 0).all()
    assert seen_hazard and not env.crashes()


def test_hazard_rows_go_to_areas_and_cast_projectiles_first(env):
    spec = env.spec
    ranks = env.hazard_ranks()
    names = [str(name) for name in env.env.hazardNames()]
    arrows = card_ids(spec, ["Arrows", "Fireball", "TheLog", "Poison", "Princess", "Hunter", "Firecracker", "Knight"])
    for index in range(env.num_envs):
        env.queue_match(index, arrows, arrows, seed=60 + index)
    obs = env.reset()
    rng = np.random.default_rng(4)
    seen = set()
    dropped = env.shots_dropped()
    for _ in range(400):
        obs, info = env.step(random_actions(obs, rng, prob=0.6))
        # No area and no projectile a card casts is left out, however many shots are in flight.
        assert (info["hazard_overflow"] == 0).all()
        for row in range(env.rows):
            n = int(obs["hazard_mask"][row].sum())
            listed = [names[int(token)] for token in obs["hazards"][row, :n, 0]]
            sides = obs["hazards"][row, :n, 1].tolist()
            rank = [3 if name in ("TowerPrincessProjectile", "KingProjectile") else ranks[name] for name in listed]
            assert rank == sorted(rank), listed
            # An Arrows volley is one hazard a side, not thirty.
            for own in (0.0, 1.0):
                assert sum(name == "ArrowsSpell" and side == own for name, side in zip(listed, sides)) <= 1
            seen.update(listed)
    assert {"ArrowsSpell", "FireballSpell", "LogProjectileRolling", "Poison", "HunterProjectile"} <= seen
    assert env.shots_dropped() >= dropped


def test_enemy_elixir_count_and_play_history(env):
    obs = env.reset(seed=9)
    rng = np.random.default_rng(1)
    off = total = 0
    for _ in range(300):
        obs, _ = env.step(random_actions(obs, rng))
        off += int(((obs["extras"][:, 0] - obs["priv_global"][:, 0]).abs() > 0.05).sum())
        total += env.rows
        for row in range(env.rows):
            n = int(obs["play_mask"][row].sum())
            assert obs["play_mask"][row, :n].all()
            # Own plays always show their point.
            own = obs["recent_plays"][row, :n, 0] > 0
            assert (obs["recent_plays"][row, :n, 3][own] == 1).all()
    assert off < 0.02 * total
    assert obs["play_mask"].any() and obs["reveal_mask"].any()


def test_public_planes_do_not_depend_on_the_hidden_deck(env):
    spec = env.spec
    a = card_ids(spec, ARCHETYPES["starter"])
    b = card_ids(spec, ARCHETYPES["hog_cycle"])
    c = card_ids(spec, ARCHETYPES["golem"])
    env.queue_match(0, a, b, seed=77)
    env.queue_match(1, a, c, seed=77)
    obs = env.reset()
    rng = np.random.default_rng(2)
    for _ in range(120):
        for key in POLICY_KEYS:
            assert torch.equal(obs[key][0], obs[key][2]), key
        actions = torch.zeros(env.rows, dtype=torch.int64)
        options = np.nonzero(obs["legal"][0].numpy())[0]
        if len(options) > 1 and rng.random() < 0.3:
            actions[0] = actions[2] = int(rng.choice(options[1:]))
        obs, info = env.step(actions)
        if info["done"][0] or info["done"][2]:
            break
    assert not torch.equal(obs["priv_deck"][0], obs["priv_deck"][2])


def test_trade_credit_stays_within_elixir_spent(env):
    spec = env.spec
    decks = [card_ids(spec, ARCHETYPES[name]) for name in ("graveyard", "golem", "lava_loon", "skeleton_king")]
    for index in range(env.num_envs):
        env.queue_match(index, decks[index % 4], decks[(index + 1) % 4], seed=40 + index)
    obs = env.reset()
    rng = np.random.default_rng(3)
    spent = np.zeros(env.rows)
    credit = np.zeros(env.rows)
    live = np.ones(env.num_envs, dtype=bool)
    for _ in range(400):
        actions = random_actions(obs, rng, prob=0.5)
        cost = obs["hand"][..., 3].numpy()
        obs, info = env.step(actions)
        rows = np.repeat(live, 2)
        played = (info["verdict"] == 1) & (actions.numpy() > 0) & (actions.numpy() < 2305) & rows
        slot = (actions.numpy() - 1) // 576
        spent[played] += cost[np.nonzero(played)[0], slot[played]]
        credit += np.where(rows, info["trade"], 0)
        live &= ~info["done"].reshape(-1, 2).any(1)
    # What the opponent is credited for damage to my troops cannot pass what I paid for them.
    theirs = credit.reshape(-1, 2)[:, ::-1].reshape(-1)
    assert spent.sum() > 50 and credit.sum() > 0
    assert (theirs <= spent + 1e-3).all()


def test_reset_from_a_seed_is_reproducible(env, model):
    runs = []
    for _ in range(2):
        obs = env.reset(seed=21)
        generator = torch.Generator().manual_seed(5)
        taken = []
        for _ in range(40):
            flat = model.act({key: obs[key] for key in POLICY_KEYS}, generator=generator, with_value=False)["flat"]
            taken.append(flat.clone())
            obs, _ = env.step(flat)
        runs.append((torch.stack(taken), obs["entities"].clone()))
    assert torch.equal(runs[0][0], runs[1][0]) and torch.equal(runs[0][1], runs[1][1])


def test_sampled_actions_are_legal_and_accepted(env, model):
    obs = env.reset(seed=5)
    for _ in range(80):
        out = model.act(obs)
        assert obs["legal"].gather(1, out["flat"][:, None]).bool().all()
        full, _ = model.full_log_probs(obs)
        picked = full.gather(1, out["flat"][:, None]).squeeze(1)
        assert torch.allclose(picked, out["logp"], atol=1e-4)
        obs, info = env.step(out["flat"])
        assert (info["verdict"] < 3).all() and only_mirror_refused(env)


def test_training_replays_rollout_and_learns(tmp_path, env):
    config = TrainConfig(
        run_dir=str(tmp_path), num_envs=8, rollout_steps=32, minibatch=128, epochs=1,
        flip_augment=True, critic_epochs=2, model=ModelConfig(**MODEL_PRESETS["small"]),
    )
    torch.manual_seed(1)
    model = Policy(env.spec, config.model)
    league = League(config, env.spec, torch.device("cpu"))
    collector = Collector(config, env, model, league, DeckStats(env.spec["n_cards"]))
    batch, stats = collector.collect(0, shaping=1.0)
    assert only_mirror_refused(env)
    pair_sums = batch["reward"].view(32, -1, 2).sum(-1)
    assert torch.allclose(pair_sums, torch.zeros_like(pair_sums), atol=1e-5)
    assert batch["reward"].abs().sum() > 0
    obs = {key: value[:16].flatten(0, 1) for key, value in batch["obs"].items()}
    with torch.no_grad():
        out = model.evaluate(obs, batch["type"][:16].flatten(), batch["cell"][:16].flatten())
    mask = batch["train"][:16].flatten()
    assert torch.allclose(out["logp"][mask], batch["logp"][:16].flatten()[mask], atol=1e-4)
    forced = mask & ~batch["choice"][:16].flatten()
    assert forced.any() and (out["logp"][forced] == 0).all()
    metrics = update(model, make_optimizer(model, config), batch, config, 0.01, torch.device("cpu"),
                     np.random.default_rng(0), Ema(model, 0.9))
    assert metrics["updates"] > 0 and metrics["skipped"] == 0
    assert all(np.isfinite(v) for v in metrics.values())


def test_paired_evaluation_counts_each_queued_game_once():
    result = play_paired("spend8", "random", pairs=6, seed=3, simulator=SIM)
    assert result["games"] + result["crashes"] == 12 and result["unplayed"] == 0
    again = play_paired("spend8", "random", pairs=6, seed=3, simulator=SIM)
    assert again["score"] == result["score"]
