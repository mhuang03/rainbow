"""Smoke tests: every trainer runs briefly and exports a network the Rust agents accept."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import rainbow_engine as re  # noqa: E402

PYTHON = Path(__file__).parent.parent
QUICK = ["--minutes", "0.05", "--eval-every", "1000", "--threads", "1", "--actors", "16"]


def run(script: str, *args: str) -> None:
    subprocess.run([sys.executable, str(PYTHON / script), *args], cwd=PYTHON, check=True, capture_output=True)


@pytest.mark.parametrize(
    "script, spec, extra",
    [
        ("train_dmc.py", "dmc", ["--hidden", "16", "8", "--aux-weight", "1"]),
        ("train_dmc.py", "dmc", ["--hidden", "16", "--opponents", "cheapest"]),
        ("train_dqn.py", "dqn", ["--hidden", "16", "--batch", "32"]),
        ("train_ppo.py", "ppo", ["--hidden", "16", "--rollout", "256", "--minibatch", "128"]),
        ("train_ppo.py", "ppo", ["--hidden", "16", "--rollout", "256", "--oracle-fraction", "0.5"]),
        ("train_nfsp.py", "nfsp", ["--hidden", "16", "--batch", "32", "--reservoir", "1000"]),
    ],
)
def test_trainer_exports_a_playable_network(tmp_path, script, spec, extra):
    out = tmp_path / "net.bin"
    run(script, *QUICK, "--out", str(out), *extra)
    agent = re.RustAgent(f"{spec}:path={out}")
    game = re.Game(3, seed=0)
    while not game.is_over:
        game.step(agent.act(game))


def test_belief_and_expert_iteration(tmp_path):
    belief = tmp_path / "belief.bin"
    run("train_belief.py", "--games", "4", "--epochs", "1", "--hidden", "16", "--agents", "cheapest", "--out", str(belief))
    re.RustAgent(f"ismcts:n=10,belief={belief}").act(re.Game(4, seed=1))

    init = tmp_path / "dmc.bin"
    run("train_dmc.py", *QUICK, "--hidden", "16", "--out", str(init))
    data = tmp_path / "selfplay.bin"
    arena = PYTHON.parent / "target/release/rainbow"
    if not arena.exists():
        pytest.skip("needs the release binary")
    subprocess.run(
        [str(arena), "selfplay", "-g", "2", "-t", "1", "-o", str(data), f"n=20,prior={init},leaf={init}"],
        check=True,
        capture_output=True,
    )
    out = tmp_path / "exit.bin"
    run("train_exit.py", "--init", str(init), "--data", str(data), "--out", str(out), "--epochs", "1", "--eval-deals", "1")
    re.RustAgent(f"dmc:path={out}")


def test_gymnasium_env():
    gym = pytest.importorskip("gymnasium")
    from gymnasium.utils.env_checker import check_env

    import rainbow_gym

    env = rainbow_gym.RainbowEnv(players=3, opponents=["cheapest", "random"])
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=1)
    total = 0.0
    for _ in range(200):
        obs, reward, terminated, truncated, info = env.step(int(np.flatnonzero(info["action_mask"])[0]))
        total += reward
        if terminated:
            break
    assert terminated and -1.0 <= total <= 2.0
