"""Per-trajectory metric computation for platformer data collections.

Loads .npz trajectory files and computes a comprehensive set of metrics
for diversity analysis, coverage reports, and data quality validation.

State vector layout (16 floats):
    [0]  player x position (px)
    [1]  player y position (px)
    [2]  player vx (px/s)
    [3]  player vy (px/s)
    [4]  player grounded (0/1)
    [5]  jump_height (physics config)
    [6]  jump_duration (physics config)
    [7]  move_speed (physics config)
    [8]  air_control (physics config)
    [9]  ground_friction (physics config)
    [10] score
    [11] episode progress (steps/max_steps, 0-1)
    [12] level_complete (0/1)
    [13] player_dead (0/1)
    [14] dynamics type ID (0-15)
    [15] vertical model index (0-3)

Usage:
    # Single episode
    metrics = compute_metrics("path/to/episode.npz")

    # Full collection -> DataFrame
    df = compute_collection_metrics("data/collections/grid-256-20260207/")

    # Save as parquet
    df.to_parquet("data/collections/grid-256-20260207/metrics.parquet")
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor


# ---------------------------------------------------------------------------
# State vector index constants — keeps the rest of the code readable
# and decoupled from any future layout changes.
# ---------------------------------------------------------------------------
_POS_X, _POS_Y = 0, 1
_VEL_X, _VEL_Y = 2, 3
_GROUNDED = 4
_JUMP_HEIGHT = 5
_JUMP_DURATION = 6
_MOVE_SPEED = 7
_AIR_CONTROL = 8
_GROUND_FRICTION = 9
_SCORE = 10
_PROGRESS = 11
_LEVEL_COMPLETE = 12
_PLAYER_DEAD = 13
_DYNAMICS_TYPE_ID = 14

# Dynamics type decoding — maps type ID (0-15) to (vertical, horizontal)
_VERTICAL_NAMES = ["parabolic", "cubic", "floaty", "asymmetric"]
_HORIZONTAL_NAMES = ["force", "velocity", "impulse", "drag_limited"]


def _decode_dynamics_type(type_id: int) -> tuple[str, str]:
    """Decode dynamics type ID into (vertical_name, horizontal_name)."""
    v_idx = int(type_id) // 4
    h_idx = int(type_id) % 4
    return _VERTICAL_NAMES[v_idx], _HORIZONTAL_NAMES[h_idx]


def compute_metrics(npz_path: str | Path) -> dict:
    """Compute all trajectory metrics from a single .npz episode file.

    Args:
        npz_path: Path to a trajectory .npz file.

    Returns:
        Dict of metric name -> value. Flat structure (no nesting)
        suitable for direct insertion into a DataFrame row.
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    states = data["states"]            # (T+1, 16)
    actions_mx = data["actions_move_x"]  # (T,)
    actions_j = data["actions_jump"]     # (T,)
    rewards = data["rewards"]            # (T,)
    terminated = data["terminated"]      # (T,)
    truncated = data["truncated"]        # (T,)

    T = len(rewards)  # number of steps (states has T+1 rows)
    fps = 30  # platformer default

    # Parse metadata if available
    meta = {}
    if "metadata_json" in data:
        meta = json.loads(str(data["metadata_json"]))

    # ------------------------------------------------------------------
    # File identity — lets us join metrics back to source files
    # ------------------------------------------------------------------
    metrics = {
        "file": npz_path.name,
        "dynamics_dir": npz_path.parent.name,
    }

    # ------------------------------------------------------------------
    # Metadata fields — policy, dynamics, physics params
    # ------------------------------------------------------------------
    metrics["policy"] = meta.get("policy", "unknown")
    metrics["dynamics_type"] = meta.get("dynamics_type", npz_path.parent.name)

    # Decode dynamics from state vector (more reliable than metadata)
    type_id = int(states[0, _DYNAMICS_TYPE_ID])
    v_name, h_name = _decode_dynamics_type(type_id)
    metrics["vertical_model"] = v_name
    metrics["horizontal_model"] = h_name
    metrics["dynamics_type_id"] = type_id

    # Physics config from state vector (constant across episode)
    metrics["jump_height"] = float(states[0, _JUMP_HEIGHT])
    metrics["jump_duration"] = float(states[0, _JUMP_DURATION])
    metrics["move_speed"] = float(states[0, _MOVE_SPEED])
    metrics["air_control"] = float(states[0, _AIR_CONTROL])
    metrics["ground_friction"] = float(states[0, _GROUND_FRICTION])

    # Layout config from metadata
    layout = meta.get("layout_config", {})
    metrics["difficulty_sigma"] = layout.get("difficulty_sigma", np.nan)
    metrics["platform_density"] = layout.get("platform_density", np.nan)

    # Behavioral profile from initial_info (ground truth from calibration)
    profile = meta.get("initial_info", {}).get("behavioral_profile", {})
    metrics["actual_apex_height"] = profile.get("actual_apex_height", np.nan)
    metrics["actual_max_speed"] = profile.get("actual_max_speed", np.nan)
    metrics["horizontal_jump_reach"] = profile.get("horizontal_jump_reach", np.nan)
    metrics["trajectory_asymmetry"] = profile.get("trajectory_asymmetry", np.nan)
    metrics["apex_dwell_fraction"] = profile.get("apex_dwell_fraction", np.nan)

    # ------------------------------------------------------------------
    # Outcome metrics
    # ------------------------------------------------------------------
    metrics["episode_steps"] = T
    metrics["episode_time"] = T / fps

    # Check final state flags for outcome
    final_complete = bool(states[-1, _LEVEL_COMPLETE] > 0.5)
    final_dead = bool(states[-1, _PLAYER_DEAD] > 0.5)
    final_truncated = bool(truncated[-1]) if len(truncated) > 0 else False

    metrics["completed"] = final_complete
    metrics["died"] = final_dead and not final_complete
    metrics["timed_out"] = final_truncated and not final_complete and not final_dead

    outcome = "timeout"
    if final_complete:
        outcome = "goal"
    elif final_dead:
        outcome = "death"
    metrics["outcome"] = outcome

    # ------------------------------------------------------------------
    # Temporal metrics
    # ------------------------------------------------------------------
    if final_complete:
        metrics["time_to_goal"] = T / fps
    else:
        metrics["time_to_goal"] = np.nan

    metrics["survival_time"] = T / fps  # same as episode_time, but semantically different

    # ------------------------------------------------------------------
    # Reward metrics
    # ------------------------------------------------------------------
    metrics["total_reward"] = float(rewards.sum())
    metrics["final_score"] = float(states[-1, _SCORE])

    # Per-signal rewards if available
    for signal in ["goal", "progress", "death", "step"]:
        key = f"reward_{signal}"
        if key in data:
            metrics[f"total_reward_{signal}"] = float(data[key].sum())

    # ------------------------------------------------------------------
    # Spatial metrics — derived from position trace
    # ------------------------------------------------------------------
    pos_x = states[:, _POS_X]  # (T+1,)
    pos_y = states[:, _POS_Y]

    metrics["max_x"] = float(pos_x.max())
    metrics["min_x"] = float(pos_x.min())
    metrics["max_y"] = float(pos_y.max())
    metrics["x_range"] = float(pos_x.max() - pos_x.min())

    # Net horizontal displacement (how far right did we get?)
    metrics["net_x_displacement"] = float(pos_x[-1] - pos_x[0])

    # Backtrack ratio: total distance traveled / net displacement
    # Measures how "direct" the trajectory is. 1.0 = perfectly direct.
    dx = np.abs(np.diff(pos_x))
    total_distance_x = float(dx.sum())
    net_distance_x = abs(float(pos_x[-1] - pos_x[0]))
    if total_distance_x > 1e-3:
        metrics["backtrack_ratio"] = net_distance_x / total_distance_x
    else:
        metrics["backtrack_ratio"] = 1.0

    # Spatial grid occupancy: discretize (x, y) into cells and count unique
    # Use 50px cells — coarse enough to be meaningful, fine enough for coverage
    cell_size = 50.0
    cells_x = (pos_x / cell_size).astype(int)
    cells_y = (pos_y / cell_size).astype(int)
    unique_cells = len(set(zip(cells_x.tolist(), cells_y.tolist())))
    metrics["unique_cells_visited"] = unique_cells

    # ------------------------------------------------------------------
    # Physics metrics — derived from velocity traces
    # ------------------------------------------------------------------
    vel_x = states[:, _VEL_X]
    vel_y = states[:, _VEL_Y]
    grounded = states[:, _GROUNDED]

    metrics["mean_abs_vx"] = float(np.abs(vel_x).mean())
    metrics["mean_vy"] = float(vel_y.mean())
    metrics["max_abs_vx"] = float(np.abs(vel_x).max())
    metrics["max_vy"] = float(vel_y.max())  # max upward velocity

    # Airborne fraction: what fraction of timesteps is the player in the air?
    airborne_mask = grounded < 0.5
    metrics["airborne_fraction"] = float(airborne_mask.mean())

    # Jump count: transitions from grounded to airborne
    grounded_bool = grounded > 0.5
    jumps = np.diff(grounded_bool.astype(int))
    metrics["jump_count"] = int((jumps == -1).sum())  # grounded->airborne

    # ------------------------------------------------------------------
    # Action metrics — derived from action sequences
    # ------------------------------------------------------------------
    move_x = actions_mx.astype(float)
    jump = actions_j.astype(float)

    metrics["mean_move_x"] = float(move_x.mean())
    metrics["mean_abs_move_x"] = float(np.abs(move_x).mean())
    metrics["jump_frequency"] = float(jump.mean())

    # Direction changes: how often does the agent switch left/right?
    # Count sign changes in move_x (ignoring near-zero)
    active_mask = np.abs(move_x) > 0.05
    active_signs = np.sign(move_x[active_mask])
    if len(active_signs) > 1:
        direction_changes = int((np.diff(active_signs) != 0).sum())
    else:
        direction_changes = 0
    metrics["direction_changes"] = direction_changes

    # Action entropy: how diverse are the actions?
    # Discretize move_x into bins and compute entropy
    move_bins = np.digitize(move_x, bins=[-0.75, -0.25, 0.25, 0.75])
    # Combine with jump to get joint action bins (5 move bins x 2 jump = 10)
    joint_actions = move_bins * 2 + jump.astype(int)
    _, counts = np.unique(joint_actions, return_counts=True)
    probs = counts / counts.sum()
    metrics["action_entropy"] = float(-np.sum(probs * np.log2(probs + 1e-10)))

    # ------------------------------------------------------------------
    # Dynamics response metrics — how the agent's state responds to actions
    # ------------------------------------------------------------------
    if T > 1:
        # Velocity changes between consecutive steps
        dvx = np.diff(vel_x[:T])  # use T states (aligned with T-1 action pairs)
        dvy = np.diff(vel_y[:T])

        # Move responsiveness: correlation between action and velocity change
        # High = actions translate directly into movement, low = sluggish/indirect
        if len(move_x[:-1]) > 1 and np.std(move_x[:-1]) > 1e-5:
            corr = np.corrcoef(move_x[:-1], dvx)[0, 1]
            metrics["move_responsiveness"] = float(corr) if not np.isnan(corr) else 0.0
        else:
            metrics["move_responsiveness"] = 0.0

        # Gravity variance: variance in vy changes when airborne
        # Different dynamics models produce different vy change patterns
        airborne_steps = airborne_mask[:T-1]
        if airborne_steps.sum() > 2:
            metrics["gravity_variance"] = float(dvy[airborne_steps].var())
        else:
            metrics["gravity_variance"] = np.nan
    else:
        metrics["move_responsiveness"] = 0.0
        metrics["gravity_variance"] = np.nan

    # ------------------------------------------------------------------
    # Collection / objective metrics
    # ------------------------------------------------------------------
    score_trace = states[:, _SCORE]
    score_gained = float(score_trace[-1] - score_trace[0])
    metrics["collectibles_gathered"] = int(round(score_gained))

    # Collectibles available from metadata
    obj_config = meta.get("objectives_config", {})
    level_geom = meta.get("initial_info", {}).get("level_geometry", {})
    n_collectibles = len(level_geom.get("collectibles", []))
    metrics["collectibles_available"] = n_collectibles
    if n_collectibles > 0:
        metrics["collection_ratio"] = score_gained / n_collectibles
    else:
        metrics["collection_ratio"] = np.nan

    return metrics


def _compute_metrics_safe(npz_path: str) -> Optional[dict]:
    """Wrapper that catches errors for parallel execution."""
    try:
        return compute_metrics(npz_path)
    except Exception as e:
        return {"file": Path(npz_path).name, "_error": str(e)}


def compute_collection_metrics(
    collection_dir: str | Path,
    workers: int = 10,
) -> pd.DataFrame:
    """Compute metrics for all episodes in a collection directory.

    Scans for .npz files under {collection_dir}/trajectories/, computes
    metrics in parallel, and returns a DataFrame with one row per episode.

    Args:
        collection_dir: Path to collection root (contains trajectories/).
        workers: Number of parallel workers.

    Returns:
        DataFrame with one row per episode and all metrics as columns.
    """
    collection_dir = Path(collection_dir)
    traj_dir = collection_dir / "trajectories"

    if not traj_dir.exists():
        raise FileNotFoundError(f"No trajectories/ dir in {collection_dir}")

    # Find all .npz files
    npz_files = sorted(traj_dir.rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {traj_dir}")

    print(f"Computing metrics for {len(npz_files)} episodes ({workers} workers)...")

    # Parallel metric computation
    paths_str = [str(p) for p in npz_files]

    if workers <= 1:
        rows = []
        for i, p in enumerate(paths_str):
            row = _compute_metrics_safe(p)
            rows.append(row)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(paths_str)}...")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_compute_metrics_safe, paths_str, chunksize=50))

    # Separate successes from errors
    good_rows = [r for r in rows if r and "_error" not in r]
    bad_rows = [r for r in rows if r and "_error" in r]

    if bad_rows:
        print(f"WARNING: {len(bad_rows)} episodes failed to compute metrics:")
        for r in bad_rows[:5]:
            print(f"  {r['file']}: {r['_error']}")
        if len(bad_rows) > 5:
            print(f"  ... and {len(bad_rows) - 5} more")

    df = pd.DataFrame(good_rows)
    print(f"Computed metrics for {len(df)} episodes successfully.")

    return df
