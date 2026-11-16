"""The cr-rl-arena bridge. Skipped unless ``cfbot_arena`` is built; the parity
tests also need the stock simulator (``cr_sim``, built from the same checkout).

The claim under test is the one the tournament depends on: the rows the policy
is trained on are the rows the client computes from what the server sends."""

import numpy as np
import pytest
import torch

cfbot_arena = pytest.importorskip("cfbot_arena")

from cfbot import actions as A  # noqa: E402
from cfbot.baselines import RushBot  # noqa: E402
from cfbot.decks import ARENA_ARCHETYPES, ARENA_POOL  # noqa: E402
from cfbot.env import INFO_KEYS, OBS_KEYS, POLICY_KEYS, card_ids, make_env  # noqa: E402

DT, SKIP = 6, 8
SPECIAL = [
    (["knight", "mirror", "goblin_gang", "graveyard", "goblin_barrel", "witch", "tombstone", "miner"],
     ["archer_queen", "elixir_collector", "royal_giant", "skeletons", "rascals", "fireball", "goblin_hut", "golem"]),
    (["little_prince", "lava_hound", "battle_ram", "night_witch", "goblin_drill", "zap", "the_log", "phoenix"],
     ["golden_knight", "mother_witch", "goblin_curse", "clone", "elixir_golem", "skeleton_barrel", "furnace", "mirror"]),
]


def tokens(spec, deck):
    return card_ids(spec, deck)


def random_actions(obs, rng, play_prob):
    legal = obs["legal"]
    acts = np.zeros(legal.shape[0], dtype=np.int64)
    for row in np.flatnonzero(rng.random(legal.shape[0]) < play_prob):
        options = np.flatnonzero(legal[row][1:]) + 1
        if len(options):
            acts[row] = rng.choice(options)
    return acts


def test_spec_and_planes():
    env = cfbot_arena.VecEnv(num_envs=3, workers=1)
    spec = env.spec()
    assert spec["simulator"] == "arena" and spec["num_actions"] == A.NUM_ACTIONS
    assert sum(spec["card_supported"]) == 116
    obs = env.reset(1)
    for key in OBS_KEYS:
        assert obs[key].shape[0] == 6, key
    assert obs["legal"].shape == (6, A.NUM_ACTIONS) and (obs["legal"][:, 0] == 1).all()
    # Nothing is legal in the secondary block, and the hazard plane is empty.
    assert not obs["legal"][:, A.SECONDARY_PLAY_START:].any() and not obs["hazard_mask"].any()
    # Six towers each, own king at the bottom.
    assert (obs["entity_count"] == 6).all()
    ent = obs["entities"][0]
    king = ent[(ent[:, 0] == 1) & (ent[:, 1] == 1)][0]
    assert king[4] == pytest.approx(0.5) and king[5] == pytest.approx(3 / 32)
    out = env.step(np.zeros(6, dtype=np.int64))
    for key in INFO_KEYS:
        assert key in out, key


def test_archetypes_are_legal_decks():
    env = cfbot_arena.VecEnv(num_envs=1, workers=1)
    spec = env.spec()
    env.set_pool([tokens(spec, deck) for deck in ARENA_ARCHETYPES.values()])
    env.set_pool([tokens(spec, deck) for deck in ARENA_POOL.values()])
    assert len(ARENA_POOL) == len(ARENA_ARCHETYPES) + 12
    assert not any("barbarian_barrel" in deck for deck in ARENA_POOL.values())
    with pytest.raises(ValueError):
        env.set_pool([tokens(spec, ["knight"] * 8)])
    with pytest.raises(ValueError):
        env.queue_match(0, tokens(spec, ["archer_queen", "monk", "knight", "archers", "giant", "zap", "bats", "pekka"]),
                        tokens(spec, ARENA_ARCHETYPES["starter"]))


def test_same_seed_same_battle():
    def run(workers):
        env = cfbot_arena.VecEnv(num_envs=8, workers=workers, seed=5)
        obs = env.reset(5)
        rng = np.random.default_rng(3)
        trace = []
        for _ in range(150):
            obs = env.step(random_actions(obs, rng, 0.3))
            trace.append((obs["entities"].sum(), obs["reward"].sum(), obs["legal"].sum()))
        return trace

    assert run(1) == run(4)


def test_every_offered_play_is_accepted():
    env = cfbot_arena.VecEnv(num_envs=32, workers=4, seed=2)
    obs = env.reset(2)
    rng = np.random.default_rng(2)
    plays = finished = 0
    for _ in range(400):
        acts = random_actions(obs, rng, 0.4)
        obs = env.step(acts)
        plays += int((acts != 0).sum())
        assert (obs["verdict"][acts != 0] == 1).all()
        assert (obs["verdict"][acts == 0] == 0).all()
        finished += int((obs["outcome"] >= 0).sum())
    assert plays > 2000 and finished > 10 and not env.crashes()


def test_rewards_follow_the_outcome():
    env = cfbot_arena.VecEnv(num_envs=16, workers=2, seed=4)
    obs = env.reset(4)
    rng = np.random.default_rng(4)
    seen = set()
    for _ in range(1500):
        obs = env.step(random_actions(obs, rng, 0.5))
        for index in np.flatnonzero(obs["outcome"] >= 0):
            outcome = int(obs["outcome"][index])
            seen.add(outcome)
            reward = obs["reward"][2 * index: 2 * index + 2]
            assert list(reward) == {0: [1, -1], 1: [-1, 1], 2: [0, 0]}[outcome]
            assert obs["done"][2 * index] and obs["episode_ticks"][index] > 0
    assert {0, 1} <= seen


def test_queued_match_plays_the_given_decks():
    env = cfbot_arena.VecEnv(num_envs=2, workers=1)
    spec = env.spec()
    blue, red = tokens(spec, ARENA_ARCHETYPES["hog_cycle"]), tokens(spec, ARENA_ARCHETYPES["golem"])
    env.queue_match(1, blue, red, 77)
    obs = env.reset()
    assert sorted(obs["own_deck"][2, :, 0].astype(int)) == sorted(blue)
    assert sorted(obs["own_deck"][3, :, 0].astype(int)) == sorted(red)
    assert sorted(obs["priv_deck"][2].astype(int)) == sorted(red)
    # The hand is four of the deck's cards and the critic sees the opponent's.
    assert set(obs["hand"][2, :, 0].astype(int)) <= set(blue)
    assert set(obs["priv_hand"][2].astype(int)) <= set(red)


def test_focus_decks_reach_one_seat():
    env = cfbot_arena.VecEnv(num_envs=64, workers=2, seed=9)
    spec = env.spec()
    focus = tokens(spec, ARENA_ARCHETYPES["guide_example"])
    obs = env.reset(9)
    assert (obs["focus_seat"] == -1).all()
    env.set_focus([focus], 1.0)
    obs = env.reset(9)
    decks = obs["own_deck"][:, :, 0].astype(int).reshape(64, 2, 8)
    holds = (np.sort(decks, -1) == np.sort(focus)).all(-1)
    assert holds.any(-1).all() and 20 < holds[:, 0].sum() < 44
    # The bridge names the seat, also for episodes that start inside a step.
    rng = np.random.default_rng(9)
    finished = 0
    for _ in range(400):
        seat = obs["focus_seat"]
        assert set(np.unique(seat)) <= {0, 1}
        decks = obs["own_deck"][:, :, 0].astype(int).reshape(64, 2, 8)
        assert (np.sort(decks[np.arange(64), seat], -1) == np.sort(focus)).all()
        obs = env.step(random_actions(obs, rng, 0.6))
        finished += int((obs["outcome"] >= 0).sum())
    assert finished > 20


def test_learner_takes_the_focus_seat(tmp_path):
    from cfbot.config import ModelConfig, TrainConfig
    from cfbot.league import League
    from cfbot.model import Policy
    from cfbot.rollout import Collector

    config = TrainConfig(run_dir=str(tmp_path), simulator="arena", num_envs=24, rollout_steps=40, device="cpu",
                         self_play_frac=0.0, baseline_frac=1.0, baseline_frac_final=1.0,
                         model=ModelConfig(d_model=32, n_heads=2, d_card=16, d_board=16, d_trunk=32,
                                           d_critic=16, critic_hidden=32))
    env = make_env(config.num_envs, simulator="arena", seed=4, max_skip=config.max_skip)
    env.set_focus([ARENA_ARCHETYPES["guide_example"]], 1.0)
    model = Policy(env.spec, config.model)
    collector = Collector(config, env, model, League(config, env.spec, torch.device("cpu")), seed=4)
    assert (collector.opponent != 0).all()
    for _ in range(3):
        collector.collect(0, 0.0)
        assert (collector.learner_side == env.focus_seat).all()


def parity_game(deck0, deck1, seed, rng, play_prob):
    """One battle in the bridge and the same battle in the stock simulator,
    driven as the server drives it, with the client's featurizer per seat."""
    cr_sim = pytest.importorskip("cr_sim")
    env = cfbot_arena.VecEnv(num_envs=1, workers=1, decision_ticks=DT, max_skip=SKIP)
    spec = env.spec()
    env.queue_match(0, tokens(spec, deck0), tokens(spec, deck1), seed)
    obs = env.reset()
    battle = cr_sim.Battle()
    battle.reset(seed, deck0, deck1)
    feats = [cfbot_arena.Featurizer(deck0, DT), cfbot_arena.Featurizer(deck1, DT)]
    for seat in (0, 1):
        feats[seat].update(battle.view(seat).observe())
    plays = reveals = 0
    for decision in range(2000):
        for seat in (0, 1):
            mine = feats[seat].observe()
            for key in POLICY_KEYS:
                assert np.array_equal(obs[key][seat], mine[key][0]), (seed, decision, seat, key)
            assert bool(obs["choice"][seat]) == feats[seat].has_choice()
        reveals = max(reveals, int(obs["reveal_mask"].sum()))
        acts = random_actions(obs, rng, play_prob)
        sent = [feats[seat].act(int(acts[seat])) for seat in (0, 1)]
        obs = env.step(acts)
        plays += int((acts != 0).sum())
        assert (obs["verdict"][acts != 0] == 1).all()
        out = 0
        for interval in range(1 + SKIP):
            for tick in range(DT):
                first = interval == 0 and tick == 0
                out = battle.step(sent[0] if first else None, sent[1] if first else None)
                for seat in (0, 1):
                    feats[seat].update(battle.view(seat).observe())
                if out:
                    break
            if out or feats[0].has_choice() or feats[1].has_choice():
                break
        if out:
            assert obs["done"][0] and obs["outcome"][0] == {1: 0, 2: 1, 3: 2}[out]
            assert obs["episode_ticks"][0] == battle.tick
            return plays, reveals
        assert not obs["done"][0]
    raise AssertionError("the battle did not end")


def test_bridge_rows_equal_the_client_rows():
    rng = np.random.default_rng(0)
    probe = cfbot_arena.VecEnv(num_envs=1, workers=1)
    names = probe.spec()["card_names"]
    plays = reveals = 0
    for index in range(8):
        if index < len(SPECIAL):
            deck0, deck1 = SPECIAL[index]
        else:
            decks = probe.random_decks(2, 100 + index)
            deck0, deck1 = [names[t] for t in decks[0]], [names[t] for t in decks[1]]
        got = parity_game(deck0, deck1, 1000 + index, rng, (0.15, 0.35, 0.8)[index % 3])
        plays += got[0]
        reveals = max(reveals, got[1])
    # The games were real ones: cards were played and opponent cards were inferred.
    assert plays > 150 and reveals >= 8


def test_a_spawner_elsewhere_does_not_hide_a_spell():
    """A Tombstone in one lane used to explain every new skeleton, so a
    Graveyard cast in the other lane was never seen."""
    cr_sim = pytest.importorskip("cr_sim")
    mine = ["knight", "archers", "giant", "musketeer", "skeletons", "fireball", "arrows", "zap"]
    theirs = ["tombstone", "graveyard", "knight", "archers", "giant", "musketeer", "ice_spirit", "zap"]
    battle = cr_sim.Battle()
    battle.reset(35, mine, theirs)
    feat = cfbot_arena.Featurizer(mine, DT)
    names = feat.card_names()
    done = set()
    for tick in range(2400):
        feat.update(battle.view(0).observe())
        action = None
        view = battle.view(1)
        hand, elixir = view.hand(), view.elixir()
        if "tombstone" not in done and "tombstone" in hand and elixir >= 3:
            action = (hand.index("tombstone"), 14.5, 5.5)
            done.add("tombstone")
        elif "tombstone" in done and "graveyard" not in done and "graveyard" in hand and elixir >= 5:
            action = (hand.index("graveyard"), 3.5, 25.5)
            done.add("graveyard")
            cast = tick
        elif "graveyard" not in done and elixir >= 9.5:
            slot = next(i for i, card in enumerate(hand) if card not in ("tombstone", "graveyard"))
            action = (slot, 9.5, 2.5)
        if battle.step(None, action):
            break
        if "graveyard" in done and tick > cast + 120:
            break
    assert done == {"tombstone", "graveyard"}
    rows = feat.observe()
    seen = {names[int(token)] for token in rows["revealed_deck"][0, :, 0] if token}
    assert {"tombstone", "graveyard"} <= seen


def test_ground_troops_are_not_offered_the_water():
    env = cfbot_arena.VecEnv(num_envs=48, workers=4, seed=6)
    spec = env.spec()
    units = spec["unit_table"]
    cards = spec["card_table"]
    obs = env.reset(6)
    rng = np.random.default_rng(6)
    river = np.zeros((A.HEIGHT, A.WIDTH), dtype=bool)
    river[15:17] = True
    off_bridge = river.copy()
    off_bridge[:, 2:5] = False
    off_bridge[:, 13:16] = False
    checked = wet = 0
    for _ in range(900):
        legal = obs["legal"][:, 1:A.PLAY_ACTION_COUNT].reshape(-1, 4, A.HEIGHT, A.WIDTH).astype(bool)
        played = obs["hand"][:, :, 1].astype(int)
        for row, slot in zip(*np.nonzero(legal[:, :, 15:21].any((-1, -2)) & ~legal[:, :, 21:].any((-1, -2)))):
            token = played[row, slot]
            walks = not (units[token, 8] or units[token, 22] or units[token, 14])
            checked += 1
            if walks:
                assert not (legal[row, slot] & off_bridge).any()
                assert (legal[row, slot] & river).any()
            else:
                wet += int((legal[row, slot] & off_bridge).any())
        obs = env.step(random_actions(obs, rng, 0.5))
    assert checked > 50 and wet > 0
    assert cards.shape[0] == len(spec["card_names"])


def test_make_env_and_scripted_bots():
    env = make_env(8, simulator="arena", seed=3, exclude_cards=("MergeMaiden",), pool_prob=0.5,
                   deck_pool=list(ARENA_ARCHETYPES.values()))
    obs = env.reset()
    assert env.spec["simulator"] == "arena" and obs["legal"].shape == (16, A.NUM_ACTIONS)
    bot = RushBot(1.0, seed=1)
    played = 0
    for _ in range(60):
        acts = bot.act(obs)
        cells = (acts[acts > 0] - 1) % A.CELLS
        assert ((cells // A.WIDTH >= 9) & (cells // A.WIDTH < 15)).all()
        played += int((acts > 0).sum())
        obs, info = env.step(acts)
        assert (info["verdict"][acts.numpy() > 0] == 1).all()
    assert played > 50
    type_mask, cell_mask = A.split_legal(obs["legal"])
    assert not type_mask[:, 5:9].any() and cell_mask.shape == (16, 8, A.CELLS)


def test_policy_runs_on_arena_rows():
    from cfbot.config import ModelConfig
    from cfbot.model import Policy

    env = make_env(4, simulator="arena", seed=1)
    obs = env.reset()
    model = Policy(env.spec, ModelConfig()).eval()
    with torch.no_grad():
        out = model.act({key: obs[key] for key in POLICY_KEYS}, generator=torch.Generator().manual_seed(0),
                        with_value=False)
    legal = obs["legal"].bool()
    assert legal.gather(1, out["flat"].view(-1, 1)).all()
    env.step(out["flat"])
