"""Gymnasium wrappers for the Parametric Physics Platformer.

These wrappers adapt the platformer's Dict observation space and hybrid
action space for standard RL libraries, which require flat Box spaces.
They also handle domain randomization (sampling new physics configs per episode).

Wrapper stack order:
  1. DomainRandomizationWrapper — sample new physics config per reset
  2. StateOnlyWrapper — extract flat state vector from Dict obs
  3. DynamicsBlindWrapper (optional) — remove dynamics type indices
  4. ContinuousActionWrapper — convert hybrid Dict action to Box(2,)
  5. HistoryStackWrapper (optional) — stack last K observations
"""

from collections import deque
from typing import Callable, Optional

import numpy as np
import gymnasium
from gymnasium import spaces

from .config import GameConfig, DynamicsConfig
from .dynamics import VerticalModel, HorizontalModel


class StateOnlyWrapper(gymnasium.ObservationWrapper):
    """Extract the flat state vector from the platformer's Dict observation.

    The platformer env returns {'rgb': (H,W,3), 'state': (16,)}. For
    state-based RL (no pixels), we only need the state vector. This
    wrapper discards RGB and returns a flat Box(16,) observation.
    """

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        # The original obs space is Dict with 'state' key
        assert isinstance(env.observation_space, spaces.Dict), (
            f"StateOnlyWrapper expects Dict obs space, got {type(env.observation_space)}"
        )
        assert "state" in env.observation_space.spaces, (
            "StateOnlyWrapper expects 'state' key in obs Dict"
        )
        self.observation_space = env.observation_space["state"]

    def observation(self, obs):
        return obs["state"]


class ContinuousActionWrapper(gymnasium.ActionWrapper):
    """Convert the platformer's hybrid action space to a continuous Box(2,).

    The platformer expects {'move_x': Box(1,), 'jump': Discrete(2)}.
    SB3 needs a single Box space. We map:
      - action[0] -> move_x (pass-through, continuous in [-1, 1])
      - action[1] -> jump (thresholded: jump=1 if action[1] > threshold, else 0)

    The SB3 policy outputs continuous values for both dimensions. For PPO,
    values are sampled from a Gaussian; for SAC, from a squashed Gaussian
    in [-1, 1]. The jump dimension naturally learns a bimodal distribution —
    values cluster near 0 (no jump) or 1 (jump).

    Args:
        env: Wrapped platformer env (after StateOnlyWrapper if used).
        jump_threshold: Threshold for converting continuous jump value
            to discrete. Default 0.5 — agent outputs above this trigger jump.
    """

    def __init__(self, env: gymnasium.Env, jump_threshold: float = 0.5):
        super().__init__(env)
        self.jump_threshold = jump_threshold
        # New action space: [move_x, jump_continuous]
        # move_x in [-1, 1], jump_continuous in [0, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

    def action(self, action):
        """Convert Box(2,) action to the platformer's Dict action format."""
        return {
            "move_x": np.array([action[0]], dtype=np.float32),
            "jump": int(action[1] > self.jump_threshold),
        }


class DynamicsBlindWrapper(gymnasium.ObservationWrapper):
    """Remove dynamics type indices from the state vector.

    The full state vector is (16,) with indices 14-15 encoding dynamics
    type information (type_id 0-15 and vertical_model 0-3). For the
    "blind" generalist agent variant, we strip these so the agent must
    infer dynamics from behavioral cues alone.

    This tests whether the agent can discover dynamics type from observation
    without being told — the key ablation for RQ4/RQ6.
    """

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box), (
            f"DynamicsBlindWrapper expects Box obs, got {type(env.observation_space)}"
        )
        orig_dim = env.observation_space.shape[0]
        assert orig_dim >= 16, (
            f"Expected at least 16-dim obs (full state vector), got {orig_dim}"
        )
        # Remove last 2 dimensions (indices 14-15)
        self.observation_space = spaces.Box(
            low=env.observation_space.low[:14],
            high=env.observation_space.high[:14],
            dtype=env.observation_space.dtype,
        )

    def observation(self, obs):
        return obs[:14]


class HistoryStackWrapper(gymnasium.ObservationWrapper):
    """Stack the last K observations into a single flat vector.

    Gives the agent a temporal context window without requiring a recurrent
    architecture. Useful for inferring dynamics from recent trajectory.

    On reset(), the deque is filled with K copies of the initial observation.
    On step() the new observation is appended and the concatenated stack returned.

    Args:
        env: A Gymnasium env with a flat Box observation space.
        k: Number of observations to stack (default 8).
    """

    def __init__(self, env: gymnasium.Env, k: int = 8):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box)
        assert len(env.observation_space.shape) == 1

        self.k = k
        self._obs_dim = env.observation_space.shape[0]

        low = np.tile(env.observation_space.low, k)
        high = np.tile(env.observation_space.high, k)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype,
        )
        self._history: deque = deque(maxlen=k)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._history.clear()
        for _ in range(self.k):
            self._history.append(obs)
        return self._stack(), info

    def observation(self, obs):
        self._history.append(obs)
        return self._stack()

    def _stack(self) -> np.ndarray:
        return np.concatenate(list(self._history), axis=0)


class DomainRandomizationWrapper(gymnasium.Wrapper):
    """Sample a new physics configuration on each episode reset.

    This is the core domain randomization mechanism for RL training.
    On each reset(), the wrapper calls the provided config_sampler to
    generate a new GameConfig, then mutates the underlying env's config
    before delegating to the env's reset (which rebuilds everything from
    config: physics world, level, player, dynamics model, calibration).

    For specialists: the sampler fixes dynamics type (vertical + horizontal
    model) while randomizing all other parameters.

    For generalists: the sampler randomizes everything including dynamics type.

    Args:
        env: A PlatformerEnv instance.
        config_sampler: A callable that returns a new GameConfig each time.
    """

    def __init__(self, env: gymnasium.Env, config_sampler: Callable[[], GameConfig]):
        super().__init__(env)
        self.config_sampler = config_sampler

    def reset(self, **kwargs):
        # Sample a fresh config and apply it to the underlying env.
        # PlatformerEnv.reset() reads self.config to rebuild the entire
        # game state, so mutating it before reset is sufficient.
        self.env.unwrapped.config = self.config_sampler()
        return self.env.reset(**kwargs)


# ---------------------------------------------------------------------------
# Config sampler factories
# ---------------------------------------------------------------------------

def make_specialist_sampler(
    vertical_model: str, horizontal_model: str
) -> Callable[[], GameConfig]:
    """Create a config sampler that fixes dynamics type, randomizes everything else.

    The specialist agent trains on one dynamics type (e.g., parabolic + force)
    with randomized physics parameters, layout, and objectives. This tests
    whether an agent can master a single dynamics regime across varied configs.

    Args:
        vertical_model: One of 'parabolic', 'cubic', 'floaty', 'asymmetric'.
        horizontal_model: One of 'force', 'velocity', 'impulse', 'drag_limited'.

    Returns:
        A callable that produces a GameConfig with fixed dynamics type.
    """
    # Validate model names at construction time
    assert vertical_model in DynamicsConfig.VERTICAL_MODELS, (
        f"Unknown vertical model: {vertical_model}"
    )
    assert horizontal_model in DynamicsConfig.HORIZONTAL_MODELS, (
        f"Unknown horizontal model: {horizontal_model}"
    )

    def sampler() -> GameConfig:
        # Sample a fully random config with ensure_features=True
        # (guarantees hazards, springs, collectibles for richer trajectories)
        config = GameConfig.sample_full(ensure_features=True)
        # Override dynamics type to the specialist's fixed type
        config.dynamics.vertical_model = vertical_model
        config.dynamics.horizontal_model = horizontal_model
        return config

    return sampler


def make_generalist_sampler() -> Callable[[], GameConfig]:
    """Create a config sampler that randomizes everything including dynamics type.

    The generalist agent trains across all 16 dynamics types with random
    physics parameters. This tests whether a single agent can handle
    diverse physics regimes.

    Returns:
        A callable that produces a fully randomized GameConfig.
    """
    def sampler() -> GameConfig:
        return GameConfig.sample_full(ensure_features=True)
    return sampler


def make_platformer_env(
    variant: str,
    dynamics_type: Optional[str] = None,
    seed: int = 0,
    history_k: int = 8,
    max_episode_steps: int = 1000,
) -> gymnasium.Env:
    """Build a fully wrapped platformer env for RL training.

    Applies the correct wrapper stack for each agent variant. This is the
    factory function passed to SB3's SubprocVecEnv (one call per worker).

    Args:
        variant: One of 'specialist', 'labeled', 'blind', 'history'.
        dynamics_type: For specialist variant: 'parabolic_force' format
            (vertical_horizontal). None for generalist variants.
        seed: Random seed for this env instance.
        history_k: Number of observations to stack for 'history' variant.
        max_episode_steps: Max steps per episode before truncation.

    Returns:
        A wrapped Gymnasium env ready for SB3.
    """
    from .gym_env import PlatformerEnv

    # Build base env
    env = PlatformerEnv(max_episode_steps=max_episode_steps)

    # 1. Domain randomization wrapper
    if variant == "specialist":
        assert dynamics_type is not None, "specialist variant requires dynamics_type"
        v_model, h_model = dynamics_type.split("_", 1)
        sampler = make_specialist_sampler(v_model, h_model)
    else:
        sampler = make_generalist_sampler()
    env = DomainRandomizationWrapper(env, sampler)

    # 2. State-only (drop RGB, keep state vector)
    env = StateOnlyWrapper(env)

    # 3. Dynamics blinding (for blind and history variants)
    if variant in ("blind", "history"):
        env = DynamicsBlindWrapper(env)

    # 4. Continuous action space (convert hybrid to Box(2,))
    env = ContinuousActionWrapper(env)

    # 5. History stacking (for history variant only)
    if variant == "history":
        env = HistoryStackWrapper(env, k=history_k)

    # Seed the env (SB3's SubprocVecEnv will call reset with its own seeds,
    # but seeding here ensures reproducible config sampling)
    env.unwrapped.np_random = np.random.default_rng(seed)

    return env
