"""Tests for trajectory data collector."""

import json
import os
import numpy as np
import pytest
from pathlib import Path

os.environ['SDL_VIDEODRIVER'] = 'dummy'

from parametric_physics_platformer.data_collector import TrajectoryCollector
from parametric_physics_platformer.gym_env import PlatformerEnv


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path / "trajectories"


@pytest.fixture
def env():
    e = PlatformerEnv(max_episode_steps=50)
    yield e
    e.close()


class TestTrajectoryCollector:
    def test_create(self, tmp_dir):
        collector = TrajectoryCollector(output_dir=str(tmp_dir))
        assert collector.episode_count == 0
        assert not collector.recording

    def test_record_episode(self, tmp_dir, env):
        collector = TrajectoryCollector(output_dir=str(tmp_dir))
        obs, info = env.reset(seed=42)
        collector.begin_episode(obs, info)
        assert collector.recording

        action = {"move_x": np.array([1.0], dtype=np.float32), "jump": 0}
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(action)
            collector.record_step(action, obs, reward, terminated, truncated, info)
            if terminated or truncated:
                break

        path = collector.end_episode()
        assert path is not None
        assert path.exists()
        assert collector.episode_count == 1
        assert not collector.recording

    def test_saved_file_contents(self, tmp_dir, env):
        collector = TrajectoryCollector(output_dir=str(tmp_dir))
        obs, info = env.reset(seed=42)
        collector.begin_episode(obs, info)

        action = {"move_x": np.array([0.5], dtype=np.float32), "jump": 1}
        n_steps = 10
        for _ in range(n_steps):
            obs, reward, terminated, truncated, info = env.step(action)
            collector.record_step(action, obs, reward, terminated, truncated, info)
            if terminated or truncated:
                break

        path = collector.end_episode()
        data = np.load(path)

        # States: initial + n_steps = n_steps + 1
        assert data["states"].shape[0] == n_steps + 1
        assert data["states"].shape[1] == 16
        # Actions: n_steps
        assert data["actions_move_x"].shape == (n_steps,)
        assert data["actions_jump"].shape == (n_steps,)
        # Rewards
        assert data["rewards"].shape == (n_steps,)
        # Done flags
        assert data["terminated"].shape == (n_steps,)
        assert data["truncated"].shape == (n_steps,)
        # RGB frames: initial + n_steps
        assert "rgb_frames" in data
        assert data["rgb_frames"].shape[0] == n_steps + 1

    def test_no_rgb(self, tmp_dir, env):
        collector = TrajectoryCollector(output_dir=str(tmp_dir), save_rgb=False)
        obs, info = env.reset(seed=42)
        collector.begin_episode(obs, info)

        action = {"move_x": np.array([0.0], dtype=np.float32), "jump": 0}
        obs, reward, terminated, truncated, info = env.step(action)
        collector.record_step(action, obs, reward, terminated, truncated, info)

        path = collector.end_episode()
        data = np.load(path)
        assert "rgb_frames" not in data
        assert "states" in data

    def test_reward_signals_saved(self, tmp_dir, env):
        collector = TrajectoryCollector(output_dir=str(tmp_dir))
        obs, info = env.reset(seed=42)
        collector.begin_episode(obs, info)

        action = {"move_x": np.array([1.0], dtype=np.float32), "jump": 0}
        obs, reward, terminated, truncated, info = env.step(action)
        collector.record_step(action, obs, reward, terminated, truncated, info)

        path = collector.end_episode()
        data = np.load(path)
        assert "reward_goal" in data
        assert "reward_progress" in data
        assert "reward_death" in data
        assert "reward_step" in data

    def test_multiple_episodes_increment(self, tmp_dir, env):
        collector = TrajectoryCollector(output_dir=str(tmp_dir))

        for i in range(3):
            obs, info = env.reset(seed=i)
            collector.begin_episode(obs, info)
            action = {"move_x": np.array([0.0], dtype=np.float32), "jump": 0}
            obs, reward, terminated, truncated, info = env.step(action)
            collector.record_step(action, obs, reward, terminated, truncated, info)
            collector.end_episode()

        assert collector.episode_count == 3
        assert (tmp_dir / "episode_0000.npz").exists()
        assert (tmp_dir / "episode_0001.npz").exists()
        assert (tmp_dir / "episode_0002.npz").exists()

    def test_end_without_begin_returns_none(self, tmp_dir):
        collector = TrajectoryCollector(output_dir=str(tmp_dir))
        assert collector.end_episode() is None

    def test_metadata_passthrough(self, tmp_dir, env):
        collector = TrajectoryCollector(output_dir=str(tmp_dir))
        obs, info = env.reset(seed=42)
        collector.begin_episode(obs, info, metadata={"policy": "rush"})
        assert collector._episode_metadata["policy"] == "rush"


class TestDataCollectorDynamicsMetadata:
    def test_episode_saves_dynamics_metadata(self, tmp_dir, env):
        """Saved episode should include dynamics type info from state vector."""
        collector = TrajectoryCollector(output_dir=str(tmp_dir))
        obs, info = env.reset(seed=42)
        collector.begin_episode(obs, info)

        action = {"move_x": np.array([1.0], dtype=np.float32), "jump": 0}
        obs, reward, terminated, truncated, info = env.step(action)
        collector.record_step(action, obs, reward, terminated, truncated, info)

        path = collector.end_episode()
        data = np.load(path)
        # State vector should now be 16-dim with dynamics type info
        assert data["states"].shape[1] == 16

    def test_dynamics_metadata_in_episode_metadata(self, tmp_dir, env):
        """Dynamics metadata should be saved when passed via begin_episode."""
        collector = TrajectoryCollector(output_dir=str(tmp_dir))
        obs, info = env.reset(seed=42)
        # Pass dynamics metadata explicitly
        dynamics_meta = info.get("dynamics_type", {})
        collector.begin_episode(obs, info, metadata={"dynamics_type": dynamics_meta})

        action = {"move_x": np.array([0.0], dtype=np.float32), "jump": 0}
        obs, reward, terminated, truncated, info = env.step(action)
        collector.record_step(action, obs, reward, terminated, truncated, info)

        collector.end_episode()
        assert collector._episode_count == 1


class TestMetadataPipeline:
    """Tests for metadata serialization to .npz files."""

    def test_metadata_json_saved_to_npz(self, tmp_dir, env):
        """Metadata should be saved as JSON string in .npz."""
        collector = TrajectoryCollector(output_dir=str(tmp_dir))
        obs, info = env.reset(seed=42)
        collector.begin_episode(obs, info, metadata={"policy": "rush"})

        action = {"move_x": np.array([1.0], dtype=np.float32), "jump": 0}
        obs, reward, terminated, truncated, info = env.step(action)
        collector.record_step(action, obs, reward, terminated, truncated, info)

        path = collector.end_episode()
        data = np.load(path, allow_pickle=True)
        assert "metadata_json" in data

        metadata = json.loads(str(data["metadata_json"]))
        assert metadata["policy"] == "rush"

    def test_metadata_contains_behavioral_profile(self, tmp_dir, env):
        """Metadata should contain behavioral profile from env info."""
        collector = TrajectoryCollector(output_dir=str(tmp_dir))
        obs, info = env.reset(seed=42)
        collector.begin_episode(obs, info, metadata={"policy": "test"})

        action = {"move_x": np.array([0.0], dtype=np.float32), "jump": 0}
        obs, reward, terminated, truncated, info = env.step(action)
        collector.record_step(action, obs, reward, terminated, truncated, info)

        path = collector.end_episode()
        data = np.load(path, allow_pickle=True)
        metadata = json.loads(str(data["metadata_json"]))

        # initial_info should contain behavioral profile from gym env
        initial_info = metadata["initial_info"]
        assert "behavioral_profile" in initial_info
        assert "actual_apex_height" in initial_info["behavioral_profile"]

    def test_metadata_contains_level_geometry(self, tmp_dir, env):
        """Metadata should contain level geometry from env info."""
        collector = TrajectoryCollector(output_dir=str(tmp_dir))
        obs, info = env.reset(seed=42)
        collector.begin_episode(obs, info, metadata={"policy": "test"})

        action = {"move_x": np.array([0.0], dtype=np.float32), "jump": 0}
        obs, reward, terminated, truncated, info = env.step(action)
        collector.record_step(action, obs, reward, terminated, truncated, info)

        path = collector.end_episode()
        data = np.load(path, allow_pickle=True)
        metadata = json.loads(str(data["metadata_json"]))

        initial_info = metadata["initial_info"]
        assert "level_geometry" in initial_info
        assert "platforms" in initial_info["level_geometry"]
        assert "collectibles" in initial_info["level_geometry"]

    def test_metadata_contains_dynamics_type(self, tmp_dir, env):
        """Metadata should contain dynamics type from env info."""
        collector = TrajectoryCollector(output_dir=str(tmp_dir))
        obs, info = env.reset(seed=42)
        collector.begin_episode(obs, info)

        action = {"move_x": np.array([0.0], dtype=np.float32), "jump": 0}
        obs, reward, terminated, truncated, info = env.step(action)
        collector.record_step(action, obs, reward, terminated, truncated, info)

        path = collector.end_episode()
        data = np.load(path, allow_pickle=True)
        metadata = json.loads(str(data["metadata_json"]))

        initial_info = metadata["initial_info"]
        assert "dynamics_type" in initial_info
        assert "level_seed" in initial_info
