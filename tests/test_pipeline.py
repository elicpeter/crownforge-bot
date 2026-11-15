"""End-to-end checks on the CPU with a small model."""

import copy

import numpy as np
import pytest
import torch

from cfbot import actions as A
from cfbot.agent import Agent
from cfbot.config import DRAFT_CARDS, MODEL_PRESETS, ModelConfig, TrainConfig
from cfbot.decks import ARCHETYPES, resolve_pool
from cfbot.env import POLICY_KEYS, PRIVATE_KEYS, Env, observed_towers
from cfbot.league import League
from cfbot.model import Policy
from cfbot.ppo import make_optimizer, update
from cfbot.rollout import Collector, gae, opponent_rows


@pytest.fixture(scope="module")
def env():
    return Env(8, workers=2, seed=3, exclude_cards=DRAFT_CARDS)


@pytest.fixture(scope="module")
def model(env):
    torch.manual_seed(0)
    return Policy(env.spec, ModelConfig(**MODEL_PRESETS["small"]))


def random_actions(obs, rng, prob=0.3):
    legal = obs["legal"].numpy()
    actions = np.zeros(legal.shape[0], dtype=np.int64)
    for row in range(legal.shape[0]):
        options = np.nonzero(legal[row])[0]
        if len(options) > 1 and rng.random() < prob:
            actions[row] = rng.choice(options[1:])
    return torch.from_numpy(actions)


def test_flat_action_round_trip():
    flat = torch.arange(A.NUM_ACTIONS)
    action_type, cell = A.from_flat(flat)
    assert torch.equal(A.to_flat(action_type, cell), flat)
    assert action_type.max() == A.NUM_TYPES - 1


def test_observation_follows_schema_two(env):
    spec = env.spec
    assert spec["schema_version"] == 2
    obs = env.reset(seed=4)
    assert obs["global"].shape[1:] == (4,)
    assert obs["entities"].shape[2] == 24
    assert obs["hand"].shape[1:] == (4, 4)
    assert obs["own_deck"].shape[1:] == (8, 2)
    assert obs["recent_plays"].shape[1:] == (32, 7)
    assert obs["revealed_deck"].shape[1:] == (8, 3)
    # Both towers of each side and both kings are on the board at the start.
    assert torch.allclose(observed_towers(obs), torch.ones(env.rows, 6))
    # Valid entity rows are sorted, so their order carries no hidden information.
    rows = obs["entities"][0][obs["entity_mask"][0].bool()].tolist()
    assert rows == sorted(rows)


def test_public_history_and_elixir_count(env):
    obs = env.reset(seed=8)
    rng = np.random.default_rng(1)
    for _ in range(150):
        obs, info = env.step(random_actions(obs, rng))
    # Counted opponent elixir matches the hidden truth when nothing but card
    # plays moves elixir; allow slack for collectors and abilities.
    error = (obs["extras"][:, 0] - obs["priv_global"][:, 0]).abs()
    assert error.median() < 0.05
    plays = obs["play_mask"].sum(1)
    assert plays.max() > 0
    row = int(plays.argmax())
    ticks = obs["recent_plays"][row][obs["play_mask"][row].bool()][:, 6]
    assert torch.all(ticks[1:] >= ticks[:-1])  # oldest first
    assert obs["reveal_mask"].sum() > 0


def test_flip_is_an_involution_and_keeps_actions_legal(env):
    obs = env.reset(seed=11)
    rng = np.random.default_rng(0)
    for _ in range(60):
        twice = A.flip_observation(A.flip_observation(obs))
        for key in obs:
            assert torch.allclose(obs[key].float(), twice[key].float(), atol=1e-6), key
        flat = random_actions(obs, rng)
        action_type, cell = A.from_flat(flat)
        plays = (action_type >= 1) & (action_type <= 8)
        mirrored = A.to_flat(action_type, torch.where(plays, A.flip_cell(cell), cell))
        flipped = A.flip_legal(obs["legal"])
        assert flipped.gather(1, mirrored[:, None]).squeeze(1)[plays].bool().all()
        obs, _ = env.step(flat)
    cells = torch.arange(A.CELLS)
    assert torch.equal(A.flip_cell(A.flip_cell(cells)), cells)


def test_sampled_actions_are_legal_and_accepted(env, model):
    obs = env.reset(seed=5)
    for _ in range(80):
        out = model.act(obs)
        assert obs["legal"].gather(1, out["flat"][:, None]).bool().all()
        full, _ = model.full_log_probs(obs)
        picked = full.gather(1, out["flat"][:, None]).squeeze(1)
        assert torch.allclose(picked, out["logp"], atol=1e-4)
        assert torch.allclose(full.exp().sum(-1), torch.ones(env.rows), atol=1e-4)
        assert (full[~obs["legal"].bool()] == float("-inf")).all()
        obs, info = env.step(out["flat"])
        assert (info["verdict"] < 2).all()


def test_policy_ignores_private_inputs(env, model):
    obs = env.reset(seed=6)
    public = {key: obs[key] for key in POLICY_KEYS}
    scrambled = dict(obs)
    for key in PRIVATE_KEYS:
        scrambled[key] = torch.ones_like(obs[key])
    first, value = model.full_log_probs(public)
    second, _ = model.full_log_probs(scrambled)
    assert torch.equal(first, second)
    assert torch.isfinite(value).all()
    # The critic does read them. Its output layer starts at zero, so compare
    # a copy with a random output layer.
    critic = copy.deepcopy(model.critic)
    torch.nn.init.normal_(critic.head[-1].weight)
    assert not torch.equal(critic(obs), critic(scrambled))


def test_actor_and_critic_share_no_parameters(model):
    actor = {id(p) for p in model.actor_parameters()}
    critic = {id(p) for p in model.critic.parameters()}
    assert actor and critic and not actor & critic


def test_agent_is_reproducible(env, model):
    def run():
        obs = env.reset(seed=21)
        agent = Agent(model, mode="sample", seed=9)
        chosen = []
        for _ in range(40):
            action = agent.act(obs)
            chosen.append(action.clone())
            obs, _ = env.step(action)
        return torch.stack(chosen)

    assert torch.equal(run(), run())


def test_archetype_decks_are_accepted(env):
    names = set(n for n, ok in zip(env.spec["card_names"], env.spec["card_supported"]) if ok)
    for name, deck in ARCHETYPES.items():
        assert len(set(deck)) == 8 and set(deck) <= names, name
    pooled = Env(16, workers=2, seed=2, deck_pool=resolve_pool("archetypes"), pool_prob=1.0)
    obs = pooled.reset(seed=2)
    assert pooled.deck_retries() == 0
    tokens = {tuple(sorted(row.tolist())) for row in obs["own_deck"][:, :, 0].long()}
    lookup = {name: index for index, name in enumerate(env.spec["card_names"]) if name}
    known = {tuple(sorted(lookup[card] for card in deck)) for deck in ARCHETYPES.values()}
    assert tokens <= known


def test_forced_waits_are_skipped(env):
    obs = env.reset(seed=13)
    # Nobody can play during the opening lockout, so the first step runs long.
    _, info = env.step(torch.zeros(env.rows, dtype=torch.int64))
    assert (info["ticks"] > env.spec["decision_ticks"]).all()


def test_gae_matches_direct_computation():
    reward = torch.tensor([[0.0], [0.0], [1.0], [0.0]])
    value = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
    done = torch.tensor([[False], [False], [True], [False]])
    ones = torch.ones_like(reward)
    advantage, returns = gae(reward, value, done, ones, torch.tensor([0.5]), lam=1.0)
    # Without discounting the return is the remaining reward plus bootstrap.
    assert torch.allclose(returns[:, 0], torch.tensor([1.0, 1.0, 1.0, 0.5]))
    assert torch.allclose(advantage, returns - value)
    # A longer step discounts the future more.
    _, shorter = gae(reward, value, done, ones * 0.5, torch.tensor([0.5]), lam=1.0)
    assert torch.allclose(shorter[:, 0], torch.tensor([0.25, 0.5, 1.0, 0.25]))
    assert torch.equal(opponent_rows(torch.tensor([1.0, 2.0, 3.0, 4.0])), torch.tensor([2.0, 1.0, 4.0, 3.0]))


def test_training_replays_rollout_and_learns(tmp_path, env):
    config = TrainConfig(
        run_dir=str(tmp_path), num_envs=8, rollout_steps=32, minibatch=128, epochs=1,
        flip_augment=False, model=ModelConfig(**MODEL_PRESETS["small"]),
    )
    torch.manual_seed(1)
    model = Policy(env.spec, config.model)
    league = League(config, env.spec, torch.device("cpu"))
    collector = Collector(config, env, model, league)
    batch, stats = collector.collect(0, shaping=1.0)
    assert stats["rejected"] == 0
    # Shaped rewards are zero-sum between the two players of a battle.
    pair_sums = batch["reward"].view(32, -1, 2).sum(-1)
    assert torch.allclose(pair_sums, torch.zeros_like(pair_sums), atol=1e-5)
    assert batch["reward"].abs().sum() > 0

    # Re-scoring stored decisions must reproduce the rollout's log-probabilities.
    obs = {key: value[:16].flatten(0, 1) for key, value in batch["obs"].items()}
    with torch.no_grad():
        out = model.evaluate(obs, batch["type"][:16].flatten(), batch["cell"][:16].flatten())
    mask = batch["train"][:16].flatten()
    assert torch.allclose(out["logp"][mask], batch["logp"][:16].flatten()[mask], atol=1e-4)

    before = [p.detach().clone() for p in model.parameters()]
    metrics = update(model, make_optimizer(model, config), batch, config, 0.01, torch.device("cpu"),
                     np.random.default_rng(0))
    assert metrics["updates"] > 0 and metrics["skipped"] == 0
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert all(torch.isfinite(p).all() for p in model.parameters())
