"""End-to-end checks on the CPU with a small model."""

import copy
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from cfbot import actions as A
from cfbot.agent import Agent
from cfbot.config import DRAFT_CARDS, MODEL_PRESETS, ModelConfig, TrainConfig
from cfbot.decks import ARCHETYPES, DeckStats, resolve_pool
from cfbot.env import POLICY_KEYS, PRIVATE_KEYS, Env, card_ids, observed_towers
from cfbot.evaluate import fit_ratings, play_paired
from cfbot.league import League
from cfbot.model import Policy
from cfbot.ppo import Ema, make_optimizer, update
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


def test_public_history_and_elixir_count():
    # Champion and Elixir Golem decks included: abilities and death rewards
    # move elixir outside card plays, and the count has to follow them.
    env = Env(32, workers=2, seed=5, exclude_cards=DRAFT_CARDS,
              deck_pool=resolve_pool("archetypes"), pool_prob=0.5)
    obs = env.reset(seed=8)
    rng = np.random.default_rng(1)
    wrong = rows = 0
    for _ in range(300):
        obs, info = env.step(random_actions(obs, rng))
        error = (obs["extras"][:, 0] - obs["priv_global"][:, 0]).abs()
        wrong += int((error > 0.05).sum())
        rows += env.rows
    assert wrong / rows < 0.01
    plays = obs["play_mask"].sum(1)
    assert plays.max() > 0
    row = int(plays.argmax())
    ticks = obs["recent_plays"][row][obs["play_mask"][row].bool()][:, 6]
    assert torch.all(ticks[1:] >= ticks[:-1])  # oldest first
    assert obs["reveal_mask"].sum() > 0


def test_cycle_features_match_the_hidden_hand():
    env = Env(16, workers=2, seed=7, exclude_cards=DRAFT_CARDS)
    obs = env.reset(seed=3)
    rng = np.random.default_rng(2)
    checked = 0
    for _ in range(300):
        obs, _ = env.step(random_actions(obs, rng))
        seen = obs["revealed_deck"][:, :, 0].long()
        back = obs["reveal_extras"][:, :, 1] > 0.5
        hand = obs["priv_hand"].long()
        in_hand = (seen.unsqueeze(-1) == hand.unsqueeze(1)).any(-1)
        valid = obs["reveal_mask"].bool()
        refilling = (hand == 0).any(-1, keepdim=True)
        # A card still queued is never in the hand; a card counted as back is
        # in the hand unless a refill is still on its way.
        assert not (valid & ~back & in_hand).any()
        assert not (valid & back & ~in_hand & ~refilling).any()
        checked += int(valid.sum())
    assert checked > 1000


def test_public_planes_do_not_depend_on_the_hidden_deck():
    # Same match twice, except for the deck of an opponent who never plays.
    spec = Env(1, workers=1).spec
    decks = [card_ids(spec, ARCHETYPES[name]) for name in ("hog_cycle", "golem", "log_bait")]
    env = Env(2, workers=1, seed=1, exclude_cards=DRAFT_CARDS)
    env.queue_match(0, decks[0], decks[1], seed=77)
    env.queue_match(1, decks[0], decks[2], seed=77)
    obs = env.reset()
    rng = np.random.default_rng(4)
    for _ in range(200):
        for key in POLICY_KEYS:
            assert torch.equal(obs[key][0], obs[key][2]), key
        blue = random_actions({"legal": obs["legal"][:1]}, rng, prob=0.4)
        actions = torch.zeros(4, dtype=torch.int64)
        actions[0] = actions[2] = blue[0]
        obs, info = env.step(actions)
        if info["done"].any():
            break
    assert not torch.equal(obs["priv_deck"][0], obs["priv_deck"][2])


def test_opponent_spell_targets_are_hidden():
    spec = Env(1, workers=1).spec
    deck = card_ids(spec, ARCHETYPES["hog_cycle"])
    spells = {spec["card_names"].index(name) for name in ("Fireball", "TheLog")}
    env = Env(4, workers=1, seed=2, exclude_cards=DRAFT_CARDS)
    for index in range(4):
        env.queue_match(index, deck, deck)
    obs = env.reset()
    rng = np.random.default_rng(5)
    hidden = 0
    for _ in range(400):
        obs, info = env.step(random_actions(obs, rng, prob=0.5))
        if info["done"].any():
            break
        plays = obs["recent_plays"][obs["play_mask"].bool()]
        own, card, shown = plays[:, 0] > 0, plays[:, 1].long(), plays[:, 3] > 0
        spell = torch.tensor([int(c) in spells for c in card], dtype=torch.bool)
        assert shown[own].all()  # a player always knows where it played
        assert not shown[~own & spell].any()
        assert shown[~own & ~spell].all()
        hidden += int((~own & spell).sum())
    assert hidden > 0


def test_trade_credit_never_exceeds_elixir_spent():
    # Graveyard, Witch-style spawners and death spawns used to be credited at
    # the full card cost per spawned body.
    spec = Env(1, workers=1).spec
    env = Env(8, workers=2, seed=3, exclude_cards=DRAFT_CARDS)
    for index, name in enumerate(("graveyard", "golem", "lava_loon", "skeleton_king") * 2):
        deck = card_ids(spec, ARCHETYPES[name])
        env.queue_match(index, deck, deck)
    obs = env.reset()
    rng = np.random.default_rng(6)
    spent = torch.zeros(env.rows)
    credited = torch.zeros(env.rows)
    for _ in range(500):
        actions = random_actions(obs, rng, prob=0.5)
        action_type, _ = A.from_flat(actions)
        slot = ((action_type - 1) % 4).clamp(0, 3)
        cost = obs["hand"][torch.arange(env.rows), slot, 3]
        obs, info = env.step(actions)
        if info["done"].any():
            break
        played = torch.from_numpy(info["verdict"] == 1) & (action_type >= 1) & (action_type <= 4)
        spent += cost * played
        credited += opponent_rows(torch.from_numpy(info["trade"]))
    assert spent.sum() > 50
    assert (credited <= spent + 1e-3).all()


def test_levels_can_vary_per_card():
    env = Env(8, workers=2, seed=4, exclude_cards=DRAFT_CARDS, rules="complete", level_jitter=(-1, 2))
    obs = env.reset(seed=1)
    levels = obs["own_deck"][:, :, 1]
    assert levels.min() >= 10 and levels.max() <= 13 and levels.unique().numel() > 1
    rng = np.random.default_rng(3)
    for _ in range(100):
        obs, info = env.step(random_actions(obs, rng))
    assert not env.crashes()
    assert obs["entities"][..., 3][obs["entity_mask"].bool()].unique().numel() > 1


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


def test_shape_knobs_default_to_the_original_network(env):
    small = MODEL_PRESETS["small"]
    torch.manual_seed(4)
    plain = Policy(env.spec, ModelConfig(**small))
    torch.manual_seed(4)
    spelled = Policy(env.spec, ModelConfig(**small, board_kernel=3, value_target="two_hot"))
    assert all(torch.equal(a, b) for a, b in zip(plain.state_dict().values(), spelled.state_dict().values()))
    returns = torch.tensor([-2.0, -0.33, 0.0, 0.512, 1.5])
    assert torch.equal(plain.value_target(returns), plain.two_hot(returns))

    wide = Policy(env.spec, ModelConfig(**small, board_kernel=7))
    assert wide.board_blocks[0].depthwise.weight.shape[-2:] == (7, 7)
    assert wide.place_block.depthwise.weight.shape[-2:] == (3, 3)
    obs = env.reset(seed=8)
    out = wide.act(obs)
    assert obs["legal"].gather(1, out["flat"][:, None]).bool().all() and torch.isfinite(out["logp"]).all()
    with pytest.raises(ValueError):
        Policy(env.spec, ModelConfig(**small, board_kernel=4))


def test_hl_gauss_targets_keep_the_mean(env):
    model = Policy(env.spec, ModelConfig(**MODEL_PRESETS["small"], value_target="hl_gauss"))
    returns = torch.tensor([-2.0, -1.0, -0.33, 0.0, 0.512, 1.0, 1.5])
    target = model.value_target(returns)
    assert torch.allclose(target.sum(-1), torch.ones(len(returns)), atol=1e-5) and (target >= 0).all()
    # Interior targets keep their mean; a target at the edge of the support is
    # pulled in by less than one bin.
    mean = (target * model.support).sum(-1)
    clamped = returns.clamp(-1.5, 1.5)
    assert torch.allclose(mean[1:-1], clamped[1:-1], atol=1e-3)
    assert ((mean - clamped).abs() < 0.05).all()
    assert ((target > 1e-3).sum(-1) > 2).all() and ((model.two_hot(returns) > 0).sum(-1) <= 2).all()


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
    never = torch.zeros_like(done)
    advantage, returns = gae(reward, value, done, never, ones, torch.tensor([0.5]), lam=1.0)
    # Without discounting the return is the remaining reward plus bootstrap.
    assert torch.allclose(returns[:, 0], torch.tensor([1.0, 1.0, 1.0, 0.5]))
    assert torch.allclose(advantage, returns - value)
    # A longer step discounts the future more.
    _, shorter = gae(reward, value, done, never, ones * 0.5, torch.tensor([0.5]), lam=1.0)
    assert torch.allclose(shorter[:, 0], torch.tensor([0.25, 0.5, 1.0, 0.25]))
    # A truncated step has no target and passes nothing back: the step before
    # it bootstraps from the value of the state the episode was cut in.
    cut = torch.tensor([[False], [True], [False], [False]])
    advantage, _ = gae(reward, value, done | cut, cut, ones, torch.tensor([0.5]), lam=1.0)
    assert advantage[1, 0] == 0 and torch.isclose(advantage[0, 0], torch.tensor(0.2 - 0.1))
    assert torch.equal(opponent_rows(torch.tensor([1.0, 2.0, 3.0, 4.0])), torch.tensor([2.0, 1.0, 4.0, 3.0]))


def test_training_replays_rollout_and_learns(tmp_path, env):
    config = TrainConfig(
        run_dir=str(tmp_path), num_envs=8, rollout_steps=32, minibatch=128, epochs=1,
        flip_augment=True, critic_epochs=2, model=ModelConfig(**MODEL_PRESETS["small"]),
    )
    torch.manual_seed(1)
    model = Policy(env.spec, config.model)
    league = League(config, env.spec, torch.device("cpu"))
    stats_by_deck = DeckStats(env.spec["n_cards"])
    collector = Collector(config, env, model, league, stats_by_deck)
    batch, stats = collector.collect(0, shaping=1.0)
    assert stats["rejected"] == 0
    # Shaped rewards are zero-sum between the two players of a battle.
    pair_sums = batch["reward"].view(32, -1, 2).sum(-1)
    assert torch.allclose(pair_sums, torch.zeros_like(pair_sums), atol=1e-5)
    assert batch["reward"].abs().sum() > 0

    # Re-scoring stored decisions must reproduce the rollout's log-probabilities,
    # including the decisions the policy made on a mirrored view.
    obs = {key: value[:16].flatten(0, 1) for key, value in batch["obs"].items()}
    with torch.no_grad():
        out = model.evaluate(obs, batch["type"][:16].flatten(), batch["cell"][:16].flatten())
    mask = batch["train"][:16].flatten()
    assert torch.allclose(out["logp"][mask], batch["logp"][:16].flatten()[mask], atol=1e-4)
    # A row that could only wait has probability one and nothing to learn.
    forced = mask & ~batch["choice"][:16].flatten()
    assert forced.any() and (out["logp"][forced] == 0).all()

    ema = Ema(model, 0.9)
    before = [p.detach().clone() for p in model.parameters()]
    metrics = update(model, make_optimizer(model, config), batch, config, 0.01, torch.device("cpu"),
                     np.random.default_rng(0), ema)
    assert metrics["updates"] > 0 and metrics["critic_updates"] > 0 and metrics["skipped"] == 0
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(torch.isfinite(v).all() for v in ema.shadow.values())


def test_paired_evaluation_counts_each_queued_game_once():
    result = play_paired("spend8", "random", pairs=6, seed=3)
    assert result["games"] + result["crashes"] == 12 and result["unplayed"] == 0
    again = play_paired("spend8", "random", pairs=6, seed=3)
    assert again["score"] == result["score"]  # same decks and seeds


def test_ratings_follow_results():
    records = [
        {"iteration": 1, "results": {"random": {"score": 0.5, "games": 64}}},
        {"iteration": 50, "results": {"random": {"score": 0.8, "games": 64},
                                      "anchor1": {"score": 0.75, "games": 64}}},
    ]
    ratings = fit_ratings(records)
    assert ratings["random"] == 0 and abs(ratings["it1"]) < 60 and ratings["it50"] > ratings["it1"] + 100


def test_league_survives_a_moved_run_and_thins_snapshots(tmp_path, env):
    run = tmp_path / "run"
    config = TrainConfig(run_dir=str(run), max_snapshots=60, model=ModelConfig(**MODEL_PRESETS["small"]))
    model = Policy(env.spec, config.model)
    league = League(config, env.spec, torch.device("cpu"))
    for iteration in range(1, 131):
        league.add_snapshot(model, iteration * 50)
    assert sum(not s["anchor"] for s in league.snapshots) == 60
    assert [s["iteration"] for s in league.anchors()][:4] == [50, 100, 200, 400]
    assert all(os.path.exists(league.path(s)) for s in league.snapshots)
    assert len(os.listdir(league.dir)) == len(league.snapshots)
    league.refresh_slots()
    assert len(set(league.slot_snapshot)) == config.league_slots  # distinct opponents
    assert league.slot_snapshot[0] >= len(league.snapshots) - config.league_recent

    # Resume from a different working directory and spelling of the run path.
    state = league.state_dict()
    moved = tmp_path / "moved"
    os.rename(run, moved)
    resumed = League(TrainConfig(run_dir=str(moved), model=config.model), env.spec, torch.device("cpu"))
    resumed.load_state_dict(state)
    assert len(resumed.snapshots) == len(league.snapshots)
    resumed.add_snapshot(model, 99_999)
    assert resumed.snapshots[-1]["id"] == league.next_id  # never reuses a file name


def test_train_command_runs_resumes_and_rates(tmp_path):
    run = str(tmp_path / "run")
    base = [sys.executable, "-m", "cfbot.train", "--run", run]
    small = ["--preset", "small", "--set", "num_envs=8", "--set", "rollout_steps=16", "--set", "minibatch=64",
             "--set", "gauntlet_pairs=2", "--set", "device=\"cpu\"", "--set", "workers=2"]
    first = subprocess.run(base + small + ["--iters", "2"], capture_output=True, text=True)
    assert first.returncode == 0, first.stderr[-2000:]
    second = subprocess.run(base + ["--resume", "--iters", "1"], capture_output=True, text=True,
                            cwd=str(tmp_path))
    assert second.returncode == 0 and "resumed" in second.stdout, second.stderr[-2000:]
    lines = open(os.path.join(run, "metrics.jsonl")).read().splitlines()
    assert len(lines) == 3 and '"gauntlet_elo"' in lines[0] and '"play_rate"' in lines[2]
    assert os.path.exists(os.path.join(run, "gauntlet.jsonl"))
    assert os.path.exists(os.path.join(run, "deck_stats.json"))
