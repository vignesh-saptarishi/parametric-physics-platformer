"""Trajectory data collector for the platformer environment.

Records per-step data (observations, actions, rewards, dones) during episodes
and saves to .npz files for later training.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List


def _json_default(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class TrajectoryCollector:
    """Records episode trajectories from PlatformerEnv.

    Stores per-step: rgb obs, state vector, action, reward, reward signals,
    terminated, truncated. Saves each episode as a compressed .npz file.

    Usage:
        collector = TrajectoryCollector(output_dir="data/trajectories")
        obs, info = env.reset()
        collector.begin_episode(obs, info)

        while True:
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            collector.record_step(action, obs, reward, terminated, truncated, info)
            if terminated or truncated:
                collector.end_episode()
                break
    """

    def __init__(
        self,
        output_dir: str = "data/trajectories",
        save_rgb: bool = True,
        compress: bool = True,
    ):
        """Initialize collector.

        Args:
            output_dir: Directory to save episode .npz files.
            save_rgb: Whether to include RGB frames (large).
            compress: Whether to use compressed npz format.
        """
        self.output_dir = Path(output_dir)
        self.save_rgb = save_rgb
        self.compress = compress

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Episode buffer
        self._reset_buffer()

        # Track episode count
        self._episode_count = self._count_existing_episodes()

    def _count_existing_episodes(self) -> int:
        """Count existing episode files to continue numbering."""
        existing = list(self.output_dir.glob("episode_*.npz"))
        if not existing:
            return 0
        nums = []
        for f in existing:
            try:
                nums.append(int(f.stem.split("_")[1]))
            except (IndexError, ValueError):
                pass
        return max(nums) + 1 if nums else 0

    def _reset_buffer(self):
        self._rgb_frames: List[np.ndarray] = []
        self._states: List[np.ndarray] = []
        self._actions_move_x: List[float] = []
        self._actions_jump: List[int] = []
        self._rewards: List[float] = []
        self._reward_signals: List[Dict[str, float]] = []
        self._terminated: List[bool] = []
        self._truncated: List[bool] = []
        self._episode_metadata: Dict[str, Any] = {}
        self._recording = False

    def begin_episode(self, obs: Dict[str, np.ndarray], info: Dict[str, Any],
                      metadata: Optional[Dict[str, Any]] = None):
        """Start recording a new episode.

        Args:
            obs: Initial observation from env.reset().
            info: Initial info from env.reset().
            metadata: Optional metadata (policy name, config, seed, etc.).
        """
        self._reset_buffer()
        self._recording = True

        # Store initial observation
        if self.save_rgb:
            self._rgb_frames.append(obs["rgb"])
        self._states.append(obs["state"])

        self._episode_metadata = metadata or {}
        self._episode_metadata["initial_info"] = info

    def record_step(
        self,
        action: Dict[str, Any],
        obs: Dict[str, np.ndarray],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ):
        """Record one step of the episode.

        Args:
            action: Action dict with 'move_x' and 'jump'.
            obs: Observation returned by env.step().
            reward: Reward from env.step().
            terminated: Whether episode terminated.
            truncated: Whether episode was truncated.
            info: Info dict from env.step().
        """
        if not self._recording:
            return

        # Actions
        move_x = action["move_x"]
        if isinstance(move_x, np.ndarray):
            move_x = float(move_x.item())
        self._actions_move_x.append(float(move_x))

        jump = action["jump"]
        if isinstance(jump, np.ndarray):
            jump = int(jump.item())
        self._actions_jump.append(int(jump))

        # Observation
        if self.save_rgb:
            self._rgb_frames.append(obs["rgb"])
        self._states.append(obs["state"])

        # Reward
        self._rewards.append(float(reward))
        if "reward_signals" in info:
            self._reward_signals.append(info["reward_signals"])

        # Done flags
        self._terminated.append(terminated)
        self._truncated.append(truncated)

    def end_episode(self, filename: Optional[str] = None) -> Optional[Path]:
        """Finish recording and save to disk.

        Args:
            filename: Optional custom filename (e.g. "rush_0001.npz").
                      If None, uses auto-generated "episode_NNNN.npz".

        Returns:
            Path to saved .npz file, or None if not recording.
        """
        if not self._recording:
            return None

        self._recording = False

        # Build arrays
        data = {
            "states": np.array(self._states, dtype=np.float32),
            "actions_move_x": np.array(self._actions_move_x, dtype=np.float32),
            "actions_jump": np.array(self._actions_jump, dtype=np.int8),
            "rewards": np.array(self._rewards, dtype=np.float32),
            "terminated": np.array(self._terminated, dtype=np.bool_),
            "truncated": np.array(self._truncated, dtype=np.bool_),
        }

        if self.save_rgb and self._rgb_frames:
            data["rgb_frames"] = np.array(self._rgb_frames, dtype=np.uint8)

        # Flatten reward signals into arrays
        if self._reward_signals:
            signal_keys = self._reward_signals[0].keys()
            for key in signal_keys:
                data[f"reward_{key}"] = np.array(
                    [s[key] for s in self._reward_signals], dtype=np.float32
                )

        # Serialize metadata as JSON string for .npz storage
        if self._episode_metadata:
            data["metadata_json"] = np.array(
                json.dumps(self._episode_metadata, default=_json_default)
            )

        # Save — use custom filename if provided, else auto-number
        if filename:
            filepath = self.output_dir / filename
        else:
            filepath = self.output_dir / f"episode_{self._episode_count:04d}.npz"

        if self.compress:
            np.savez_compressed(filepath, **data)
        else:
            np.savez(filepath, **data)

        self._episode_count += 1
        self._reset_buffer()

        return filepath

    @property
    def episode_count(self) -> int:
        return self._episode_count

    @property
    def recording(self) -> bool:
        return self._recording
