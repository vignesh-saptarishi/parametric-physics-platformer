"""Tests for platformer RL wrappers."""

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import numpy as np
import pytest
import gymnasium
from gymnasium import spaces

from parametric_physics_platformer.gym_env import PlatformerEnv
from parametric_physics_platformer.config import GameConfig, PhysicsConfig, DynamicsConfig
from parametric_physics_platformer.rl_wrappers import (
    StateOnlyWrapper,
    ContinuousActionWrapper,
    DynamicsBlindWrapper,
    DomainRandomizationWrapper,
    make_specialist_sampler,
    make_generalist_sampler,
    make_platformer_env,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_env():
    """A fresh PlatformerEnv for wrapping."""
    env = PlatformerEnv()
    yield env
    env.close()


@pytest.fixture
def reset_env():
    """A PlatformerEnv that has been reset (has valid state)."""
    env = PlatformerEnv()
    env.reset(seed=42)
    yield env
    env.close()


# ---------------------------------------------------------------------------
# StateOnlyWrapper
# ---------------------------------------------------------------------------

class TestStateOnlyWrapper:
    def test_obs_space_is_box_16(self, base_env):
        wrapped = StateOnlyWrapper(base_env)
        assert isinstance(wrapped.observation_space, spaces.Box)
        assert wrapped.observation_space.shape == (16,)
        wrapped.close()

    def test_obs_values_match_state(self, base_env):
        wrapped = StateOnlyWrapper(base_env)
        obs, _ = wrapped.reset(seed=42)
        # Get raw obs from the underlying env for comparison
        raw = base_env._get_state_vector()
        np.testing.assert_array_almost_equal(obs, raw)
        wrapped.close()

    def test_step_returns_state_vector(self, base_env):
        wrapped = StateOnlyWrapper(base_env)
        wrapped.reset(seed=42)
        action = {"move_x": np.array([0.5], dtype=np.float32), "jump": 0}
        obs, reward, terminated, truncated, info = wrapped.step(action)
        assert obs.shape == (16,)
        assert obs.dtype == np.float32
        wrapped.close()


# ---------------------------------------------------------------------------
# ContinuousActionWrapper
# ---------------------------------------------------------------------------

class TestContinuousActionWrapper:
    def test_action_space_is_box_2(self, base_env):
        wrapped = ContinuousActionWrapper(base_env)
        assert isinstance(wrapped.action_space, spaces.Box)
        assert wrapped.action_space.shape == (2,)
        wrapped.close()

    def test_move_x_passthrough(self, base_env):
        wrapped = ContinuousActionWrapper(base_env)
        # Test that action[0] passes through to move_x
        converted = wrapped.action(np.array([0.7, 0.0], dtype=np.float32))
        assert converted["move_x"][0] == pytest.approx(0.7)
        wrapped.close()

    def test_jump_threshold_above(self, base_env):
        wrapped = ContinuousActionWrapper(base_env)
        converted = wrapped.action(np.array([0.0, 0.6], dtype=np.float32))
        assert converted["jump"] == 1
        wrapped.close()

    def test_jump_threshold_below(self, base_env):
        wrapped = ContinuousActionWrapper(base_env)
        converted = wrapped.action(np.array([0.0, 0.4], dtype=np.float32))
        assert converted["jump"] == 0
        wrapped.close()

    def test_jump_threshold_exact(self, base_env):
        """action[1] == 0.5 exactly should NOT trigger jump."""
        wrapped = ContinuousActionWrapper(base_env)
        converted = wrapped.action(np.array([0.0, 0.5], dtype=np.float32))
        assert converted["jump"] == 0
        wrapped.close()

    def test_jump_threshold_just_above(self, base_env):
        """action[1] = 0.501 should trigger jump."""
        wrapped = ContinuousActionWrapper(base_env)
        converted = wrapped.action(np.array([0.0, 0.501], dtype=np.float32))
        assert converted["jump"] == 1
        wrapped.close()

    def test_step_with_continuous_action(self, base_env):
        """Full step through the wrapper should work without error."""
        wrapped = StateOnlyWrapper(base_env)
        wrapped = ContinuousActionWrapper(wrapped)
        wrapped.reset(seed=42)
        action = np.array([0.5, 0.8], dtype=np.float32)
        obs, reward, terminated, truncated, info = wrapped.step(action)
        assert obs.shape == (16,)
        wrapped.close()


# ---------------------------------------------------------------------------
# DynamicsBlindWrapper
# ---------------------------------------------------------------------------

class TestDynamicsBlindWrapper:
    def test_obs_shape_is_14(self, base_env):
        wrapped = StateOnlyWrapper(base_env)
        wrapped = DynamicsBlindWrapper(wrapped)
        assert wrapped.observation_space.shape == (14,)
        wrapped.close()

    def test_dynamics_indices_removed(self, base_env):
        wrapped = StateOnlyWrapper(base_env)
        wrapped = DynamicsBlindWrapper(wrapped)
        obs, _ = wrapped.reset(seed=42)
        # Get full state for comparison
        full_state = base_env._get_state_vector()
        # Blind obs should match first 14 elements
        np.testing.assert_array_almost_equal(obs, full_state[:14])
        wrapped.close()

    def test_step_returns_14_dim(self, base_env):
        wrapped = StateOnlyWrapper(base_env)
        wrapped = DynamicsBlindWrapper(wrapped)
        wrapped = ContinuousActionWrapper(wrapped)
        wrapped.reset(seed=42)
        obs, _, _, _, _ = wrapped.step(np.array([0.5, 0.0], dtype=np.float32))
        assert obs.shape == (14,)
        wrapped.close()


# ---------------------------------------------------------------------------
# DomainRandomizationWrapper
# ---------------------------------------------------------------------------

class TestDomainRandomizationWrapper:
    def test_specialist_keeps_dynamics_type(self):
        """Specialist sampler should fix dynamics type across resets."""
        env = PlatformerEnv()
        sampler = make_specialist_sampler("cubic", "velocity")
        wrapped = DomainRandomizationWrapper(env, sampler)

        for _ in range(5):
            _, info = wrapped.reset(seed=None)
            assert info["dynamics_type"]["vertical"] == "cubic"
            assert info["dynamics_type"]["horizontal"] == "velocity"
        wrapped.close()

    def test_specialist_randomizes_physics(self):
        """Specialist should vary physics params across resets."""
        env = PlatformerEnv()
        sampler = make_specialist_sampler("parabolic", "force")
        wrapped = DomainRandomizationWrapper(env, sampler)

        jump_heights = []
        for _ in range(10):
            wrapped.reset(seed=None)
            jump_heights.append(wrapped.unwrapped.config.physics.jump_height)

        # Physics params should vary (very unlikely to get 10 identical values)
        assert len(set(jump_heights)) > 1
        wrapped.close()

    def test_generalist_varies_dynamics_type(self):
        """Generalist sampler should produce varying dynamics types."""
        env = PlatformerEnv()
        sampler = make_generalist_sampler()
        wrapped = DomainRandomizationWrapper(env, sampler)

        vertical_types = set()
        for _ in range(50):  # Enough resets to see variation
            _, info = wrapped.reset(seed=None)
            vertical_types.add(info["dynamics_type"]["vertical"])

        # Should see at least 2 different vertical types in 50 resets
        assert len(vertical_types) >= 2
        wrapped.close()

    def test_env_functions_after_config_mutation(self):
        """Env should work correctly after domain randomization changes config."""
        env = PlatformerEnv()
        sampler = make_generalist_sampler()
        wrapped = DomainRandomizationWrapper(env, sampler)
        wrapped = StateOnlyWrapper(wrapped)
        wrapped = ContinuousActionWrapper(wrapped)

        wrapped.reset(seed=42)
        action = np.array([0.5, 0.0], dtype=np.float32)
        for _ in range(10):
            obs, reward, terminated, truncated, info = wrapped.step(action)
            if terminated or truncated:
                wrapped.reset(seed=42)
        wrapped.close()


# ---------------------------------------------------------------------------
# Full wrapper stacks
# ---------------------------------------------------------------------------

class TestFullStackSpecialist:
    def test_obs_and_action_shapes(self):
        env = make_platformer_env(variant="specialist", dynamics_type="parabolic_force")
        obs, _ = env.reset(seed=42)
        assert obs.shape == (16,)
        assert env.action_space.shape == (2,)
        env.close()

    def test_100_steps_no_crash(self):
        env = make_platformer_env(variant="specialist", dynamics_type="cubic_drag_limited")
        env.reset(seed=42)
        for _ in range(100):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()
        env.close()


class TestFullStackLabeled:
    def test_obs_shape_16(self):
        env = make_platformer_env(variant="labeled")
        obs, _ = env.reset(seed=42)
        assert obs.shape == (16,)
        env.close()


class TestFullStackBlind:
    def test_obs_shape_14(self):
        env = make_platformer_env(variant="blind")
        obs, _ = env.reset(seed=42)
        assert obs.shape == (14,)
        assert env.action_space.shape == (2,)
        env.close()

    def test_100_steps_no_crash(self):
        env = make_platformer_env(variant="blind")
        env.reset(seed=42)
        for _ in range(100):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()
        env.close()


class TestFullStackHistory:
    def test_obs_shape_k_times_14(self):
        env = make_platformer_env(variant="history", history_k=8)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (8 * 14,)  # 112
        assert env.action_space.shape == (2,)
        env.close()

    def test_custom_k(self):
        env = make_platformer_env(variant="history", history_k=4)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (4 * 14,)  # 56
        env.close()

    def test_first_obs_is_uniform(self):
        """On reset, all K history slots should contain the same observation."""
        env = make_platformer_env(variant="history", history_k=4)
        obs, _ = env.reset(seed=42)
        # Each slot is 14-dim, all slots should be identical
        slots = obs.reshape(4, 14)
        for i in range(1, 4):
            np.testing.assert_array_almost_equal(slots[0], slots[i])
        env.close()

    def test_100_steps_no_crash(self):
        env = make_platformer_env(variant="history", history_k=8)
        env.reset(seed=42)
        for _ in range(100):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()
        env.close()


# ---------------------------------------------------------------------------
# Sampler validation
# ---------------------------------------------------------------------------

class TestSamplerValidation:
    def test_invalid_vertical_model_raises(self):
        with pytest.raises(AssertionError, match="Unknown vertical model"):
            make_specialist_sampler("nonexistent", "force")

    def test_invalid_horizontal_model_raises(self):
        with pytest.raises(AssertionError, match="Unknown horizontal model"):
            make_specialist_sampler("parabolic", "nonexistent")

    def test_specialist_without_dynamics_type_raises(self):
        with pytest.raises(AssertionError, match="requires dynamics_type"):
            make_platformer_env(variant="specialist", dynamics_type=None)
