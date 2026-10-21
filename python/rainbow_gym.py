"""Gymnasium environment for Rainbow: one learner seat against Rust agents.

Observations are the 172-float `Game.observation` vectors; actions are indices in
`0..NUM_ACTIONS`. The legal-action mask is in `info["action_mask"]` and from `action_masks()`
(the convention of sb3-contrib's MaskablePPO). The reward is 0 until the game ends, then the
learner's win share plus a small score-margin term (as in the other trainers).

    import rainbow_gym
    env = rainbow_gym.RainbowEnv(players=4, opponents=["economist"])
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(info["action_mask"].argmax())
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import rainbow_engine as re
from rl_common import final_returns


class RainbowEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        players: int | None = 4,
        opponents: list[str] | None = None,
        render_mode: str | None = None,
    ):
        """`players` fixes the player count (None: 2 to 6 at random each game). Opponents are Rust
        agent specs, drawn at random for each non-learner seat (default: `economist`)."""
        self.players = players
        self.opponent_specs = opponents or ["economist"]
        self.render_mode = render_mode
        self.observation_space = spaces.Box(0.0, 1.0, shape=(re.OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(re.NUM_ACTIONS)
        self._game: re.Game | None = None
        self._seat = 0
        self._opponents: dict[int, re.RustAgent] = {}

    def _advance(self) -> None:
        """Plays opponents until it is the learner's turn or the game ends."""
        game = self._game
        while not game.is_over and game.current_player != self._seat:
            game.step(self._opponents[game.current_player].act(game))

    def _obs(self) -> tuple[np.ndarray, dict[str, Any]]:
        mask = self._game.legal_action_mask() if not self._game.is_over else np.zeros(re.NUM_ACTIONS, bool)
        return self._game.observation(self._seat), {"action_mask": mask}

    def action_masks(self) -> np.ndarray:
        return self._obs()[1]["action_mask"]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        n = self.players or int(rng.integers(re.MIN_PLAYERS, re.MAX_PLAYERS + 1))
        self._game = re.Game(n, seed=int(rng.integers(1 << 62)), leader=int(rng.integers(n)))
        self._seat = int(rng.integers(n))
        self._opponents = {
            s: re.RustAgent(self.opponent_specs[int(rng.integers(len(self.opponent_specs)))], seed=int(rng.integers(1 << 62)))
            for s in range(n)
            if s != self._seat
        }
        self._advance()
        return self._obs()

    def step(self, action: int):
        if self._game is None or self._game.is_over:
            raise RuntimeError("call reset() first")
        self._game.step(int(action))
        self._advance()
        obs, info = self._obs()
        terminated = self._game.is_over
        reward = float(final_returns(self._game)[self._seat]) if terminated else 0.0
        if terminated:
            info["scores"] = self._game.scores()
        return obs, reward, terminated, False, info

    def render(self) -> str | None:
        if self.render_mode == "ansi" and self._game is not None:
            return str(self._game)
        return None
