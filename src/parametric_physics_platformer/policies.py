"""Scripted policies for automated data collection.

Each policy takes an observation and returns an action dict
compatible with PlatformerEnv's action space.
"""

import numpy as np
from typing import Dict, Any, Optional


class BasePolicy:
    """Base class for scripted policies."""

    name: str = "base"

    def __call__(self, obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        return self.act(obs)

    def act(self, obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        raise NotImplementedError

    def reset(self):
        """Called at the start of each episode."""
        pass

    def _make_action(self, move_x: float, jump: int) -> Dict[str, Any]:
        return {
            "move_x": np.array([np.clip(move_x, -1.0, 1.0)], dtype=np.float32),
            "jump": int(jump),
        }


class RandomPolicy(BasePolicy):
    """Uniform random actions each step.

    Broad state coverage, many deaths, good baseline.
    """

    name = "random"

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()

    def act(self, obs):
        move_x = self.rng.uniform(-1.0, 1.0)
        jump = int(self.rng.random() < 0.15)  # 15% jump chance per step
        return self._make_action(move_x, jump)


class RushPolicy(BasePolicy):
    """Always move right, jump when velocity stalls or on a timer.

    Fast completions, misses collectibles.
    """

    name = "rush"

    def __init__(self, jump_interval: int = 25):
        self.jump_interval = jump_interval
        self._step = 0

    def reset(self):
        self._step = 0

    def act(self, obs):
        state = obs["state"]
        vx = state[2]
        vy = state[3]
        grounded = state[4] > 0.5

        self._step += 1

        # Jump if: on ground AND (periodic timer OR horizontal speed stalled)
        should_jump = grounded and (
            self._step % self.jump_interval == 0
            or abs(vx) < 10.0
        )

        return self._make_action(1.0, int(should_jump))


class CautiousPolicy(BasePolicy):
    """Slow movement, careful jumps. Waits when velocity is high.

    Safe play, slow, fewer deaths.
    """

    name = "cautious"

    def __init__(self):
        self._step = 0

    def reset(self):
        self._step = 0

    def act(self, obs):
        state = obs["state"]
        vx = state[2]
        vy = state[3]
        grounded = state[4] > 0.5

        self._step += 1

        # Move right at half speed
        move_x = 0.5

        # If falling fast, slow down horizontal
        if vy < -100:
            move_x = 0.2

        # Jump periodically when grounded, less often than rush
        should_jump = grounded and self._step % 40 == 0

        return self._make_action(move_x, int(should_jump))


class ExplorerPolicy(BasePolicy):
    """Seeks collectibles by varying movement direction.

    High collectible score, thorough coverage.
    """

    name = "explorer"

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()
        self._step = 0
        self._direction = 1.0  # Start moving right
        self._direction_timer = 0

    def reset(self):
        self._step = 0
        self._direction = 1.0
        self._direction_timer = 0

    def act(self, obs):
        state = obs["state"]
        vx = state[2]
        grounded = state[4] > 0.5

        self._step += 1
        self._direction_timer += 1

        # Periodically change direction (explore both ways)
        if self._direction_timer > 60 + self.rng.integers(0, 40):
            self._direction *= -1
            self._direction_timer = 0

        # Mostly move in current direction but with some randomness
        move_x = self._direction * 0.7 + self.rng.uniform(-0.3, 0.3)

        # Jump frequently to reach platforms with collectibles
        should_jump = grounded and (
            self._step % 20 == 0
            or abs(vx) < 5.0
        )

        return self._make_action(move_x, int(should_jump))


POLICIES = {
    "random": RandomPolicy,
    "rush": RushPolicy,
    "cautious": CautiousPolicy,
    "explorer": ExplorerPolicy,
}
