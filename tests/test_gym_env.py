"""Tests for Gymnasium environment wrapper."""

import os
import numpy as np
import pytest

# Headless rendering
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from parametric_physics_platformer.gym_env import PlatformerEnv
from parametric_physics_platformer.config import GameConfig, PhysicsConfig, DynamicsConfig


class TestPlatformerEnvCreation:
    def test_create_default(self):
        env = PlatformerEnv()
        assert env.observation_space is not None
        assert env.action_space is not None
        env.close()

    def test_create_with_config(self):
        config = GameConfig(physics=PhysicsConfig(jump_height=150))
        env = PlatformerEnv(config=config)
        assert env.config.physics.jump_height == 150
        env.close()

    def test_custom_resolution(self):
        env = PlatformerEnv(obs_resolution=(64, 64))
        obs, _ = env.reset(seed=42)
        assert obs["rgb"].shape == (64, 64, 3)
        env.close()

    def test_custom_max_steps(self):
        env = PlatformerEnv(max_episode_steps=500)
        assert env.max_episode_steps == 500
        env.close()


class TestPlatformerEnvReset:
    def test_reset_returns_obs_and_info(self):
        env = PlatformerEnv()
        obs, info = env.reset(seed=42)
        assert "rgb" in obs
        assert "state" in obs
        assert "score" in info
        assert "episode_steps" in info
        env.close()

    def test_obs_shapes(self):
        env = PlatformerEnv(obs_resolution=(84, 84))
        obs, _ = env.reset(seed=42)
        assert obs["rgb"].shape == (84, 84, 3)
        assert obs["rgb"].dtype == np.uint8
        assert obs["state"].shape == (16,)
        assert obs["state"].dtype == np.float32
        env.close()

    def test_obs_in_observation_space(self):
        env = PlatformerEnv()
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        env.close()

    def test_reset_with_seed_reproducible(self):
        env = PlatformerEnv()
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1["state"], obs2["state"])
        env.close()

    def test_initial_state_vector(self):
        env = PlatformerEnv()
        obs, _ = env.reset(seed=42)
        state = obs["state"]
        # Player position should be nonzero
        assert state[0] != 0.0 or state[1] != 0.0
        # Physics config should be populated
        assert state[5] == pytest.approx(env.config.physics.jump_height, rel=1e-5)
        assert state[6] == pytest.approx(env.config.physics.jump_duration, rel=1e-5)
        # Score should be 0, episode progress 0
        assert state[10] == 0.0
        assert state[11] == 0.0
        # Not dead, not complete
        assert state[12] == 0.0
        assert state[13] == 0.0
        env.close()


class TestPlatformerEnvStep:
    def test_step_returns_five_values(self):
        env = PlatformerEnv()
        env.reset(seed=42)
        action = {"move_x": np.array([0.5], dtype=np.float32), "jump": 0}
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        env.close()

    def test_step_obs_in_space(self):
        env = PlatformerEnv()
        env.reset(seed=42)
        action = {"move_x": np.array([0.0], dtype=np.float32), "jump": 0}
        obs, _, _, _, _ = env.step(action)
        assert env.observation_space.contains(obs)
        env.close()

    def test_reward_signals_in_info(self):
        env = PlatformerEnv()
        env.reset(seed=42)
        action = {"move_x": np.array([1.0], dtype=np.float32), "jump": 0}
        _, _, _, _, info = env.step(action)
        assert "reward_signals" in info
        signals = info["reward_signals"]
        assert "goal" in signals
        assert "progress" in signals
        assert "death" in signals
        assert "step" in signals
        env.close()

    def test_episode_steps_increment(self):
        env = PlatformerEnv()
        env.reset(seed=42)
        action = {"move_x": np.array([0.0], dtype=np.float32), "jump": 0}
        env.step(action)
        env.step(action)
        env.step(action)
        assert env._episode_steps == 3
        env.close()

    def test_truncation_at_max_steps(self):
        env = PlatformerEnv(max_episode_steps=5)
        env.reset(seed=42)
        action = {"move_x": np.array([0.0], dtype=np.float32), "jump": 0}
        for _ in range(4):
            _, _, terminated, truncated, _ = env.step(action)
            assert not truncated
        _, _, terminated, truncated, _ = env.step(action)
        assert truncated
        env.close()

    def test_moving_right_gives_positive_progress(self):
        env = PlatformerEnv()
        env.reset(seed=42)
        # Move right for several steps
        action = {"move_x": np.array([1.0], dtype=np.float32), "jump": 0}
        total_progress = 0.0
        for _ in range(10):
            _, _, _, _, info = env.step(action)
            total_progress += info["reward_signals"]["progress"]
        assert total_progress > 0.0
        env.close()

    def test_no_action_no_crash(self):
        """Neutral action should not crash."""
        env = PlatformerEnv()
        env.reset(seed=42)
        action = {"move_x": np.array([0.0], dtype=np.float32), "jump": 0}
        for _ in range(20):
            env.step(action)
        env.close()


class TestPlatformerEnvActionSpace:
    def test_sample_action(self):
        env = PlatformerEnv()
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        env.close()

    def test_many_random_steps(self):
        """Run 100 random steps without crashing."""
        env = PlatformerEnv()
        env.reset(seed=42)
        for _ in range(100):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset(seed=42)
        env.close()


class TestPlatformerEnvRendering:
    def test_rgb_zeros_without_render_mode(self):
        # Without render_mode, RGB is zeros (skip rendering optimization)
        env = PlatformerEnv()
        obs, _ = env.reset(seed=42)
        assert obs["rgb"].sum() == 0
        env.close()

    def test_rgb_not_all_zeros_with_render_mode(self):
        # With render_mode="rgb_array", obs RGB has actual content
        env = PlatformerEnv(render_mode="rgb_array")
        obs, _ = env.reset(seed=42)
        assert obs["rgb"].sum() > 0
        env.close()

    def test_rgb_array_render_mode(self):
        env = PlatformerEnv(render_mode="rgb_array")
        env.reset(seed=42)
        frame = env.render()
        assert frame.shape == (256, 256, 3)
        assert frame.dtype == np.uint8
        env.close()


class TestPlatformerEnvRewardWeights:
    def test_custom_reward_weights(self):
        env = PlatformerEnv(reward_weights={
            "goal": 1000.0,
            "progress": 0.0,
            "death": -100.0,
            "step": 0.0,
        })
        env.reset(seed=42)
        action = {"move_x": np.array([1.0], dtype=np.float32), "jump": 0}
        _, reward, _, _, info = env.step(action)
        # With progress=0 and step=0, reward comes only from goal/death
        signals = info["reward_signals"]
        expected = (
            1000.0 * signals["goal"]
            + 0.0 * signals["progress"]
            + (-100.0) * signals["death"]
            + 0.0 * signals["step"]
        )
        assert reward == pytest.approx(expected)
        env.close()


class TestGymEnvDynamicsMetadata:
    def test_state_vector_includes_dynamics_type(self):
        """State vector should include dynamics type id."""
        env = PlatformerEnv()
        obs, info = env.reset(seed=42)
        # State vector now has dynamics type id at index 14, vertical type at 15
        assert obs["state"].shape[0] == 16
        assert obs["state"][14] >= 0  # type_id is non-negative
        assert obs["state"][14] < 16  # max 16 combos
        assert obs["state"][15] >= 0  # vertical type
        assert obs["state"][15] < 4  # 4 vertical models
        env.close()

    def test_info_includes_dynamics_metadata(self):
        """Info dict should include full dynamics metadata."""
        env = PlatformerEnv()
        obs, info = env.reset(seed=42)
        assert "dynamics_type" in info
        assert "vertical" in info["dynamics_type"]
        assert "horizontal" in info["dynamics_type"]
        env.close()

    def test_dynamics_type_from_config(self):
        """Env should create dynamics model from config."""
        config = GameConfig()
        config.dynamics.vertical_model = "cubic"
        env = PlatformerEnv(config=config)
        obs, info = env.reset(seed=42)
        assert info["dynamics_type"]["vertical"] == "cubic"
        # State vector type_id should reflect cubic
        # Cubic = index 1, force = index 0 -> type_id = 1*4 + 0 = 4
        assert obs["state"][14] == 4.0
        env.close()

    def test_default_dynamics_is_standard(self):
        """Default config should use parabolic + force (type_id=0)."""
        env = PlatformerEnv()
        obs, info = env.reset(seed=42)
        assert info["dynamics_type"]["vertical"] == "parabolic"
        assert info["dynamics_type"]["horizontal"] == "force"
        assert obs["state"][14] == 0.0
        assert obs["state"][15] == 0.0
        env.close()

    def test_dynamics_metadata_persists_across_steps(self):
        """Dynamics metadata should be in info on every step."""
        env = PlatformerEnv()
        env.reset(seed=42)
        action = {"move_x": np.array([1.0], dtype=np.float32), "jump": 0}
        _, _, _, _, info = env.step(action)
        assert "dynamics_type" in info
        env.close()
