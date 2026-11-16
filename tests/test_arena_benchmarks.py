"""Regression checks for the arena's fixed-deck evaluation opponents."""

import pytest
import torch

from cfbot import actions as A
from cfbot.arena import ArenaEnv
from cfbot.arena_benchmarks import ArenaBenchmarkBot
from cfbot.baselines import RushBot
from cfbot.decks import ARENA_ARCHETYPES
from cfbot.env import POLICY_KEYS, card_ids, select_rows
from cfbot.evaluate import ARENA_BENCHMARK_DECKS, Player, gauntlet_bots, play_paired


NAMES = ("hog-cycle", "golem-beatdown", "bridge-pressure", "xbow-siege", "log-bait", "graveyard-control")


def test_benchmark_names_are_arena_only():
    assert set(NAMES) == set(ARENA_BENCHMARK_DECKS)
    # The profiles are played with --suite. A run's gauntlet keeps its scripted
    # anchors, so its rating scale does not change opponents half way.
    arena = gauntlet_bots("arena")
    assert set(NAMES).isdisjoint(arena) and "rush" in arena
    assert len(arena) == len(set(arena))
    for simulator in ("crownforge", "crforge"):
        assert set(NAMES).isdisjoint(gauntlet_bots(simulator))
        with pytest.raises(ValueError, match="arena"):
            Player("hog-cycle", {"simulator": simulator}, "cpu", "greedy", 0)
        with pytest.raises(ValueError, match="arena"):
            play_paired("rush", "hog-cycle", simulator={"simulator": simulator})


@pytest.mark.parametrize("name", NAMES)
def test_benchmark_plays_legal_actions_from_public_rows(name):
    pytest.importorskip("cfbot_arena")
    env = ArenaEnv(2, workers=1)
    fixed = card_ids(env.spec, ARENA_BENCHMARK_DECKS[name])
    opponent = card_ids(env.spec, ARENA_ARCHETYPES["starter"])
    env.queue_match(0, fixed, opponent, 17)
    env.queue_match(1, opponent, fixed, 17)
    obs = env.reset()
    benchmark = ArenaBenchmarkBot(name, env.spec)
    rush = RushBot(seed=19)
    benchmark_rows, rush_rows = torch.tensor([0, 3]), torch.tensor([1, 2])
    plays = torch.zeros(2, dtype=torch.int64)

    for step in range(120):
        public = {key: obs[key] for key in POLICY_KEYS}
        rows = select_rows(public, benchmark_rows)
        actions = benchmark.act(rows)
        assert actions.shape == (2,) and actions.dtype == torch.int64
        assert rows["legal"].gather(1, actions[:, None]).all()

        if step % 20 == 0:
            assert torch.equal(actions, benchmark.act(rows))
            reversed_rows = select_rows(public, benchmark_rows.flip(0))
            assert torch.equal(actions.flip(0), benchmark.act(reversed_rows))
            wait_only = dict(rows, legal=rows["legal"].clone())
            wait_only["legal"][0, 1:] = False
            assert benchmark.act(wait_only).tolist() == [0, int(actions[1])]

        batch = torch.zeros(env.rows, dtype=torch.int64)
        batch[benchmark_rows] = actions
        batch[rush_rows] = rush.act(select_rows(public, rush_rows))
        obs, info = env.step(batch)
        played = actions != 0
        assert (info["verdict"][benchmark_rows.tolist()][played.numpy()] == 1).all()
        plays += played
        if info["done"].any():
            break

    assert (plays > 0).all()
    assert not env.crashes()


@pytest.mark.parametrize("name, card, depth", [
    ("hog-cycle", "hogrider", range(12, 15)),
    ("golem-beatdown", "golem", range(0, 6)),
    ("bridge-pressure", "battle_ram", range(12, 15)),
    ("xbow-siege", "xbow", range(12, 15)),
    ("log-bait", "goblin_barrel", range(24, 28)),
    ("graveyard-control", "graveyard", range(24, 28)),
])
def test_benchmark_uses_its_win_condition_in_the_weaker_lane(name, card, depth):
    pytest.importorskip("cfbot_arena")
    env = ArenaEnv(1, workers=1)
    fixed = card_ids(env.spec, ARENA_BENCHMARK_DECKS[name])
    opponent = card_ids(env.spec, ARENA_ARCHETYPES["starter"])
    # This deal includes the win condition and, for bait, leaves Princess queued.
    env.queue_match(0, fixed, opponent, 0)
    obs = env.reset()
    for _ in range(100):
        if obs["global"][0, 1] >= 9.8:
            break
        obs, _ = env.step(torch.zeros(env.rows, dtype=torch.int64))
    assert obs["global"][0, 1] >= 9.8

    public = select_rows({key: obs[key] for key in POLICY_KEYS}, torch.tensor([0]))
    entities = public["entities"][0]
    right_princess = (entities[:, 0] == 2) & (entities[:, 1] == 0) & (entities[:, 4] > 0.5)
    assert right_princess.sum() == 1
    entities[right_princess, 6] = 0.5

    action = ArenaBenchmarkBot(name, env.spec).act(public)
    assert public["legal"][0, action.item()]
    action_type, cell = (int(value.item()) for value in A.from_flat(action))
    assert 1 <= action_type <= A.HAND_SLOTS
    token = int(public["hand"][0, action_type - 1, 0])
    assert env.spec["card_names"][token] == card
    assert cell % A.WIDTH >= A.WIDTH // 2
    assert cell // A.WIDTH in depth


@pytest.mark.parametrize("sides, simulator", [
    (("hog-cycle", "xbow-siege"), None),
    (("rush", "log-bait"), {"simulator": "arena"}),
])
def test_paired_evaluation_queues_fixed_decks_in_both_seats(monkeypatch, sides, simulator):
    pytest.importorskip("cfbot_arena")
    queued = []
    given = (ARENA_ARCHETYPES["starter"], ARENA_ARCHETYPES["guide_example"])
    queue_match = ArenaEnv.queue_match

    def record_match(env, index, blue, red, seed=None):
        names = env.spec["card_names"]
        queued.append((index, [names[token] for token in blue], [names[token] for token in red], seed))
        queue_match(env, index, blue, red, seed)

    monkeypatch.setattr(ArenaEnv, "queue_match", record_match)
    result = play_paired(*sides, pairs=1, seed=31, max_envs=2, simulator=simulator, matchups=[given])

    first, second = (list(ARENA_BENCHMARK_DECKS.get(name, deck)) for name, deck in zip(sides, given))
    assert len(queued) == 2
    assert queued[0][:3] == (0, first, second)
    assert queued[1][:3] == (1, second, first)
    assert queued[0][3] == queued[1][3] is not None
    assert result["games"] == 2
    assert result["crashes"] == result["unplayed"] == 0
