import copy
import random

import numpy as np
import pytest

import rainbow_engine as re


def play_random(game, rng):
    while not game.is_over:
        game.step(rng.choice(game.legal_actions()))
    return game


def test_constants():
    assert re.NUM_ACTIONS == 75
    assert re.OBS_SIZE > 0
    assert (re.MIN_PLAYERS, re.MAX_PLAYERS) == (2, 6)


@pytest.mark.parametrize("n", range(2, 7))
def test_random_games_finish(n):
    rng = random.Random(n)
    for seed in range(50):
        g = play_random(re.Game(n, seed=seed), rng)
        assert g.legal_actions() == []
        assert not g.legal_action_mask().any()
        assert abs(sum(g.returns()) - 1.0) < 1e-6
        assert set(g.winners()) == {i for i, s in enumerate(g.scores()) if s == max(g.scores())}


def test_seed_is_deterministic():
    assert re.Game(4, seed=7) == re.Game(4, seed=7)
    assert re.Game(4, seed=7) != re.Game(4, seed=8)


def test_mask_matches_legal_actions():
    g = re.Game(4, seed=1)
    mask = g.legal_action_mask()
    assert mask.shape == (re.NUM_ACTIONS,) and mask.dtype == np.bool_
    assert sorted(np.flatnonzero(mask).tolist()) == sorted(g.legal_actions())
    assert all(g.is_legal(a) == mask[a] for a in range(re.NUM_ACTIONS))


def test_illegal_actions_raise():
    g = re.Game(3, seed=0)
    illegal = next(a for a in range(re.NUM_ACTIONS) if not g.is_legal(a))
    with pytest.raises(ValueError):
        g.step(illegal)
    with pytest.raises(ValueError):
        g.step(re.NUM_ACTIONS)
    with pytest.raises(ValueError):
        re.Game(7)
    with pytest.raises(IndexError):
        g.hand(3)


def test_clone_is_independent():
    g = re.Game(4, seed=3)
    for c in (g.clone(), copy.copy(g), copy.deepcopy(g)):
        assert c == g
        c.step(c.legal_actions()[0])
        assert c != g


def test_observation_shape_and_hidden_info():
    g = re.Game(4, seed=5)
    obs = g.observation(0)
    assert obs.shape == (re.OBS_SIZE,) and obs.dtype == np.float32
    assert ((obs >= 0) & (obs <= 1)).all()
    # first block is the observer's own hand
    assert np.allclose(obs[:6] * 10, g.hand(0))


def test_determinize_preserves_observer_view():
    rng = random.Random(0)
    g = re.Game(4, seed=11)
    for _ in range(6):
        g.step(rng.choice(g.legal_actions()))
    d = g.determinize(1, seed=2)
    assert d.hand(1) == g.hand(1)
    assert np.array_equal(d.observation(1), g.observation(1))
    assert d.scores() == g.scores() and d.trick() == g.trick()
    assert sum(map(sum, (d.hand(p) for p in range(4)))) == sum(map(sum, (g.hand(p) for p in range(4))))


def test_rollout_returns_distribution():
    g = re.Game(3, seed=4)
    r = g.rollout(seed=1)
    assert len(r) == 3 and abs(sum(r) - 1.0) < 1e-6
    assert not g.is_over  # rollout works on a copy


def test_batch_observations():
    games = [re.Game(n, seed=n) for n in range(2, 7)]
    obs, masks = re.batch_observations(games)
    assert obs.shape == (5, re.OBS_SIZE) and masks.shape == (5, re.NUM_ACTIONS)
    for i, g in enumerate(games):
        assert np.array_equal(obs[i], g.observation(g.current_player))
        assert np.array_equal(masks[i], g.legal_action_mask())


def test_action_helpers():
    assert re.action_to_str(0) == "1"
    assert re.action_info(0) == ("single", 1, 1)
    assert re.action_info(6) == ("set", 1, 2)
    assert re.action_info(re.NUM_ACTIONS - 1) == ("run", 5, 6)
    kinds = [re.action_info(a)[0] for a in range(re.NUM_ACTIONS)]
    assert (kinds.count("single"), kinds.count("set"), kinds.count("run")) == (6, 54, 15)


def test_action_features_table():
    table = re.action_features()
    assert table.shape == (re.NUM_ACTIONS, re.ACTION_FEATURES) and table.dtype == np.float32
    assert len({row.tobytes() for row in table}) == re.NUM_ACTIONS
    # first features are card counts per rank / 10
    assert np.allclose(table[6][:6] * 10, [2, 0, 0, 0, 0, 0])  # a pair of 1s


@pytest.mark.parametrize("spec", ["random", "cheapest", "economist", "pimc:k=4", "ismcts:n=50"])
def test_rust_agents_play_legal_moves(spec):
    agent = re.RustAgent(spec, seed=1)
    g = re.Game(4, seed=2)
    while not g.is_over:
        a = agent.act(g)
        assert g.is_legal(a)
        g.step(a)
    with pytest.raises(ValueError):
        agent.act(g)
    with pytest.raises(ValueError):
        re.RustAgent("nonsense")


def test_history_records_every_play():
    rng = random.Random(3)
    g = re.Game(3, seed=3)
    played = []
    while not g.is_over:
        seat, action = g.current_player, rng.choice(g.legal_actions())
        g.step(action)
        played.append((seat, action))
    assert g.history() == played


def test_full_observation_extends_observation():
    g = re.Game(4, seed=8)
    full = g.full_observation(1)
    assert full.shape == (re.FULL_OBS_SIZE,) and re.FULL_OBS_SIZE == re.OBS_SIZE + re.HIDDEN_SIZE
    assert np.array_equal(full[: re.OBS_SIZE], g.observation(1))
    # the first hidden block is the player's own hand
    assert np.allclose(full[re.OBS_SIZE : re.OBS_SIZE + 6] * 10, g.hand(1))
    batch = re.batch_full_observations([g])
    assert np.array_equal(batch[0], g.full_observation(g.current_player))


def test_pickle_round_trip():
    import pickle

    rng = random.Random(5)
    for n in range(2, 7):
        g = re.Game(n, seed=n)
        for _ in range(rng.randrange(40)):
            if g.is_over:
                break
            g.step(rng.choice(g.legal_actions()))
        restored = pickle.loads(pickle.dumps(g))
        assert restored == g
        assert re.Game.from_bytes(g.to_bytes()) == g
    with pytest.raises(ValueError):
        re.Game.from_bytes(b"\x07")


def test_vec_env_runs_and_resets():
    env = re.VecEnv(16, seed=1)
    rng = np.random.default_rng(0)
    finished = 0
    for _ in range(200):
        obs, masks = env.observe()
        assert obs.shape == (16, re.OBS_SIZE) and masks.any(axis=1).all()
        before = [env.game(i) for i in range(len(env))]
        actions = np.array([rng.choice(np.flatnonzero(m)) for m in masks])
        done, returns, scores = env.step(actions)
        for i in np.flatnonzero(done):
            g = before[i]
            g.step(int(actions[i]))
            assert g.is_over
            n = g.num_players
            assert np.allclose(returns[i, :n], g.returns())
            assert list(scores[i, :n]) == g.scores()
            finished += 1
        assert not done[~done].any()
    assert finished > 0
    obs, masks, full = env.observe(full=True)
    assert full.shape == (16, re.FULL_OBS_SIZE)
    with pytest.raises(ValueError):
        env.step(np.zeros(3, dtype=np.int64))


def test_vec_env_with_opponents_only_shows_learner_turns():
    env = re.VecEnv(8, seed=2, players=4, opponents=["cheapest", "economist"], learners=1)
    rng = np.random.default_rng(1)
    finished = 0
    for _ in range(300):
        obs, masks = env.observe()
        seats = env.current_players()
        learners = env.learner_seats()
        assert learners.sum(axis=1).tolist() == [1] * 8
        assert all(learners[i, seats[i]] for i in range(8))
        actions = (rng.random(masks.shape) * masks).argmax(1)
        done, returns, scores = env.step(actions)
        finished += int(done.sum())
        assert np.allclose(returns[done].sum(axis=1), 1.0)
    assert finished > 0
    with pytest.raises(ValueError):
        re.VecEnv(2, opponents=["nonsense"])


def test_vec_env_pfsp_tracks_results_against_each_opponent():
    # a random learner rarely finishes ahead of the Economist, so PFSP should favour it
    env = re.VecEnv(16, seed=3, players=3, opponents=["random", "economist"], learners=1, pfsp=2.0)
    rng = np.random.default_rng(2)
    assert env.opponent_win_rates() == [0.5, 0.5]
    for _ in range(600):
        _, masks = env.observe()
        env.step((rng.random(masks.shape) * masks).argmax(1))
    against_random, against_economist = env.opponent_win_rates()
    assert against_economist < against_random
