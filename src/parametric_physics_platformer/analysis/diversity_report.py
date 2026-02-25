"""Aggregate diversity analysis for platformer data collections.

Takes a metrics DataFrame (from trajectory_metrics.py) and produces:
1. Coverage heatmap: dynamics_type x policy matrix
2. State-space occupancy analysis
3. Policy divergence matrix (per dynamics type)
4. Gap report: flags under-sampled or under-performing cells
5. Calibration summary: behavioral profiles across dynamics types

Outputs: JSON summary + matplotlib figures saved to an analysis/ subdir.

Usage:
    from parametric_physics_platformer.analysis.trajectory_metrics import compute_collection_metrics
    from parametric_physics_platformer.analysis.diversity_report import generate_report

    df = compute_collection_metrics("data/collections/grid-256-20260207/")
    generate_report(df, output_dir="data/collections/grid-256-20260207/analysis")
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless rendering — no display needed
import matplotlib.pyplot as plt
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


# ---------------------------------------------------------------------------
# Dynamics type ordering for consistent axes across all plots
# ---------------------------------------------------------------------------
_VERTICAL_ORDER = ["parabolic", "cubic", "floaty", "asymmetric"]
_HORIZONTAL_ORDER = ["force", "velocity", "impulse", "drag_limited"]
_DYNAMICS_ORDER = [
    f"{v}_{h}" for v in _VERTICAL_ORDER for h in _HORIZONTAL_ORDER
]
_POLICY_ORDER = ["random", "rush", "cautious", "explorer"]


def _pivot_grid(df, value_col, aggfunc="mean"):
    """Create dynamics_type x policy pivot table with consistent ordering."""
    pivot = df.pivot_table(
        index="dynamics_type", columns="policy",
        values=value_col, aggfunc=aggfunc,
    )
    # Reindex to canonical order, filling missing cells with NaN
    pivot = pivot.reindex(index=_DYNAMICS_ORDER, columns=_POLICY_ORDER)
    return pivot


def _save_fig(fig, output_dir, name):
    """Save figure and close it."""
    path = Path(output_dir) / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


# ===================================================================
# 1. Coverage heatmap
# ===================================================================

def plot_coverage_heatmap(df, output_dir):
    """Plot 3-panel heatmap: episode count, completion rate, mean episode length.

    Gives a quick overview of data balance and policy effectiveness
    across the dynamics grid.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    # Panel 1: Episode counts
    counts = _pivot_grid(df, "episode_steps", aggfunc="count")
    im1 = axes[0].imshow(counts.values, cmap="YlGn", aspect="auto")
    axes[0].set_title("Episode Count")
    _annotate_heatmap(axes[0], counts)
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # Panel 2: Completion rate
    completion = _pivot_grid(df, "completed", aggfunc="mean")
    im2 = axes[1].imshow(completion.values, cmap="RdYlGn", aspect="auto",
                          vmin=0, vmax=1)
    axes[1].set_title("Completion Rate")
    _annotate_heatmap(axes[1], completion, fmt=".0%")
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    # Panel 3: Mean episode length (steps)
    length = _pivot_grid(df, "episode_steps", aggfunc="mean")
    im3 = axes[2].imshow(length.values, cmap="YlOrRd", aspect="auto")
    axes[2].set_title("Mean Episode Steps")
    _annotate_heatmap(axes[2], length, fmt=".0f")
    plt.colorbar(im3, ax=axes[2], shrink=0.8)

    # Shared axis labels
    for ax in axes:
        ax.set_xticks(range(len(_POLICY_ORDER)))
        ax.set_xticklabels(_POLICY_ORDER, rotation=45, ha="right")
        ax.set_yticks(range(len(_DYNAMICS_ORDER)))
        ax.set_yticklabels(_DYNAMICS_ORDER, fontsize=8)

    fig.suptitle("Coverage Heatmap: Dynamics Type x Policy", fontsize=14)
    fig.tight_layout()
    return _save_fig(fig, output_dir, "coverage_heatmap")


def _annotate_heatmap(ax, data, fmt=".0f"):
    """Add text annotations to each cell of a heatmap."""
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.values[i, j]
            if pd.isna(val):
                text = "-"
            elif fmt == ".0%":
                text = f"{val:.0%}"
            else:
                text = f"{val:{fmt}}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if val and not pd.isna(val) and val > data.values[~np.isnan(data.values)].mean() else "black")


# ===================================================================
# 2. State-space occupancy
# ===================================================================

def plot_state_space_occupancy(df, output_dir):
    """Plot state-space coverage metrics per dynamics type.

    Shows how much of the reachable state space each dynamics type
    actually covers. Sparse coverage = agents aren't exploring enough.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Group by dynamics type
    grouped = df.groupby("dynamics_type")

    # Panel 1: Unique cells visited distribution
    ax = axes[0, 0]
    dynamics_data = [grouped.get_group(dt)["unique_cells_visited"].values
                     for dt in _DYNAMICS_ORDER if dt in grouped.groups]
    dynamics_labels = [dt for dt in _DYNAMICS_ORDER if dt in grouped.groups]
    bp = ax.boxplot(dynamics_data, vert=True, patch_artist=True)
    ax.set_xticklabels(dynamics_labels, rotation=90, fontsize=7)
    ax.set_ylabel("Unique cells visited")
    ax.set_title("Spatial Coverage by Dynamics Type")

    # Panel 2: Airborne fraction distribution
    ax = axes[0, 1]
    data = [grouped.get_group(dt)["airborne_fraction"].values
            for dt in _DYNAMICS_ORDER if dt in grouped.groups]
    bp = ax.boxplot(data, vert=True, patch_artist=True)
    ax.set_xticklabels(dynamics_labels, rotation=90, fontsize=7)
    ax.set_ylabel("Airborne fraction")
    ax.set_title("Airborne Fraction by Dynamics Type")

    # Panel 3: Max X reached distribution
    ax = axes[1, 0]
    data = [grouped.get_group(dt)["max_x"].values
            for dt in _DYNAMICS_ORDER if dt in grouped.groups]
    bp = ax.boxplot(data, vert=True, patch_artist=True)
    ax.set_xticklabels(dynamics_labels, rotation=90, fontsize=7)
    ax.set_ylabel("Max X position (px)")
    ax.set_title("Horizontal Reach by Dynamics Type")

    # Panel 4: Action entropy distribution
    ax = axes[1, 1]
    data = [grouped.get_group(dt)["action_entropy"].values
            for dt in _DYNAMICS_ORDER if dt in grouped.groups]
    bp = ax.boxplot(data, vert=True, patch_artist=True)
    ax.set_xticklabels(dynamics_labels, rotation=90, fontsize=7)
    ax.set_ylabel("Action entropy (bits)")
    ax.set_title("Action Diversity by Dynamics Type")

    fig.suptitle("State-Space Occupancy", fontsize=14)
    fig.tight_layout()
    return _save_fig(fig, output_dir, "state_space_occupancy")


# ===================================================================
# 3. Policy divergence matrix
# ===================================================================

def plot_policy_divergence(df, output_dir):
    """Plot policy divergence: how differently do policies behave on each dynamics type?

    For each dynamics type, compute the pairwise distributional distance
    between policies using a set of key behavioral metrics. Higher distance
    means more diverse data from that dynamics type.

    We use mean absolute difference on standardized metrics as a simple,
    interpretable divergence measure (not KL — distributions are too small
    for reliable density estimation at validation-tier sizes).
    """
    # Key metrics that characterize trajectory behavior
    behavior_cols = [
        "episode_steps", "max_x", "airborne_fraction", "mean_abs_vx",
        "jump_count", "action_entropy", "backtrack_ratio", "direction_changes",
    ]

    # Filter to columns that exist
    behavior_cols = [c for c in behavior_cols if c in df.columns]

    # Standardize within the full dataset so all metrics are comparable
    df_std = df[behavior_cols].copy()
    for col in behavior_cols:
        std = df_std[col].std()
        if std > 1e-10:
            df_std[col] = (df_std[col] - df_std[col].mean()) / std
        else:
            df_std[col] = 0.0

    df_std["dynamics_type"] = df["dynamics_type"].values
    df_std["policy"] = df["policy"].values

    # Compute mean standardized profile per (dynamics_type, policy) cell
    cell_means = df_std.groupby(["dynamics_type", "policy"])[behavior_cols].mean()

    # For each dynamics type, compute pairwise policy distances
    n_policies = len(_POLICY_ORDER)
    divergence_matrices = {}

    for dt in _DYNAMICS_ORDER:
        mat = np.zeros((n_policies, n_policies))
        for i, p1 in enumerate(_POLICY_ORDER):
            for j, p2 in enumerate(_POLICY_ORDER):
                if (dt, p1) in cell_means.index and (dt, p2) in cell_means.index:
                    v1 = cell_means.loc[(dt, p1)].values
                    v2 = cell_means.loc[(dt, p2)].values
                    mat[i, j] = float(np.mean(np.abs(v1 - v2)))
        divergence_matrices[dt] = mat

    # Plot as a 4x4 grid of small divergence matrices
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))

    for idx, dt in enumerate(_DYNAMICS_ORDER):
        ax = axes[idx // 4, idx % 4]
        mat = divergence_matrices[dt]
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0,
                        vmax=max(m.max() for m in divergence_matrices.values()))
        ax.set_title(dt.replace("_", "\n"), fontsize=8)
        ax.set_xticks(range(n_policies))
        ax.set_yticks(range(n_policies))
        if idx // 4 == 3:
            ax.set_xticklabels([p[:3] for p in _POLICY_ORDER], fontsize=7)
        else:
            ax.set_xticklabels([])
        if idx % 4 == 0:
            ax.set_yticklabels([p[:3] for p in _POLICY_ORDER], fontsize=7)
        else:
            ax.set_yticklabels([])

    fig.suptitle("Policy Divergence by Dynamics Type\n(mean abs diff on standardized behavior metrics)",
                 fontsize=13)
    fig.tight_layout()
    return _save_fig(fig, output_dir, "policy_divergence")


# ===================================================================
# 4. Gap report
# ===================================================================

def compute_gap_report(df, min_episodes=10, min_completion_rate=0.05):
    """Identify gaps in the collection: under-sampled or problematic cells.

    Args:
        df: Metrics DataFrame.
        min_episodes: Flag cells with fewer episodes than this.
        min_completion_rate: Flag cells with lower completion rate.

    Returns:
        Dict with gap analysis results.
    """
    gaps = {
        "low_episode_count": [],
        "zero_completions": [],
        "low_completion_rate": [],
        "low_action_diversity": [],
        "sparse_coverage": [],
    }

    for dt in _DYNAMICS_ORDER:
        for pol in _POLICY_ORDER:
            cell = df[(df["dynamics_type"] == dt) & (df["policy"] == pol)]

            if len(cell) == 0:
                gaps["low_episode_count"].append({
                    "dynamics_type": dt, "policy": pol,
                    "count": 0, "issue": "missing"
                })
                continue

            if len(cell) < min_episodes:
                gaps["low_episode_count"].append({
                    "dynamics_type": dt, "policy": pol,
                    "count": len(cell), "target": min_episodes,
                })

            completion_rate = cell["completed"].mean()
            if completion_rate == 0:
                gaps["zero_completions"].append({
                    "dynamics_type": dt, "policy": pol,
                    "episodes": len(cell),
                })
            elif completion_rate < min_completion_rate:
                gaps["low_completion_rate"].append({
                    "dynamics_type": dt, "policy": pol,
                    "rate": float(completion_rate),
                    "threshold": min_completion_rate,
                })

            # Low action diversity: action entropy < 1 bit
            mean_entropy = cell["action_entropy"].mean()
            if mean_entropy < 1.0:
                gaps["low_action_diversity"].append({
                    "dynamics_type": dt, "policy": pol,
                    "mean_entropy": float(mean_entropy),
                })

            # Sparse spatial coverage: median unique cells < 5
            median_cells = cell["unique_cells_visited"].median()
            if median_cells < 5:
                gaps["sparse_coverage"].append({
                    "dynamics_type": dt, "policy": pol,
                    "median_cells": float(median_cells),
                })

    # Remove empty gap categories
    gaps = {k: v for k, v in gaps.items() if v}

    return gaps


# ===================================================================
# 5. Calibration summary
# ===================================================================

def plot_calibration_summary(df, output_dir):
    """Plot behavioral profile stats across dynamics types.

    Shows how calibrated behavioral properties (apex height, max speed,
    jump reach) vary across physics configs and dynamics types. This is
    the ground truth that probes will try to recover from latent space.
    """
    profile_cols = [
        ("actual_apex_height", "Apex Height (px)"),
        ("actual_max_speed", "Max Speed (px/s)"),
        ("horizontal_jump_reach", "Jump Reach (px)"),
        ("trajectory_asymmetry", "Trajectory Asymmetry"),
        ("apex_dwell_fraction", "Apex Dwell Fraction"),
    ]

    # Filter to cols that exist and have data
    profile_cols = [(c, label) for c, label in profile_cols
                    if c in df.columns and df[c].notna().sum() > 0]

    n_panels = len(profile_cols)
    if n_panels == 0:
        return None

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    grouped = df.groupby("dynamics_type")

    for ax, (col, label) in zip(axes, profile_cols):
        data = [grouped.get_group(dt)[col].dropna().values
                for dt in _DYNAMICS_ORDER if dt in grouped.groups]
        labels = [dt for dt in _DYNAMICS_ORDER if dt in grouped.groups]
        bp = ax.boxplot(data, vert=True, patch_artist=True)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylabel(label)
        ax.set_title(label)

    fig.suptitle("Behavioral Calibration Across Dynamics Types", fontsize=14)
    fig.tight_layout()
    return _save_fig(fig, output_dir, "calibration_summary")


# ===================================================================
# 6. Physics parameter distributions
# ===================================================================

def plot_physics_distributions(df, output_dir):
    """Plot distributions of physics parameters in the collection.

    Verifies that the random sampling covers the intended parameter ranges
    and isn't accidentally clustered.
    """
    physics_cols = [
        ("jump_height", "Jump Height (px)"),
        ("jump_duration", "Jump Duration (s)"),
        ("move_speed", "Move Speed (px/s)"),
        ("air_control", "Air Control"),
        ("ground_friction", "Ground Friction"),
        ("difficulty_sigma", "Difficulty Sigma"),
    ]

    physics_cols = [(c, label) for c, label in physics_cols
                    if c in df.columns and df[c].notna().sum() > 0]

    n_panels = len(physics_cols)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, (col, label) in enumerate(physics_cols):
        if i < len(axes):
            ax = axes[i]
            ax.hist(df[col].dropna(), bins=30, edgecolor="black",
                    alpha=0.7, color="#4C72B0")
            ax.set_xlabel(label)
            ax.set_ylabel("Count")
            ax.set_title(label)

    # Hide unused axes
    for i in range(len(physics_cols), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Physics Parameter Distributions", fontsize=14)
    fig.tight_layout()
    return _save_fig(fig, output_dir, "physics_distributions")


# ===================================================================
# 7. Visitation heatmap (x, y)
# ===================================================================

def _load_positions_for_file(npz_path_str):
    """Load (x, y, vx, vy, grounded) and dynamics_type from one .npz file.

    Returns (dynamics_type, policy, positions, velocities, grounded) or None on error.
    Runs in a subprocess for parallel loading.
    """
    try:
        npz_path = Path(npz_path_str)
        data = np.load(npz_path, allow_pickle=True)
        states = data["states"]  # (T+1, 16)
        pos = states[:, 0:2].copy()       # (T+1, 2) — x, y
        vel = states[:, 2:4].copy()       # (T+1, 2) — vx, vy
        grounded = states[:, 4].copy()    # (T+1,) — 0/1

        # Get dynamics type from metadata or directory name
        meta = {}
        if "metadata_json" in data:
            meta = json.loads(str(data["metadata_json"]))
        dynamics_type = meta.get("dynamics_type", npz_path.parent.name)
        policy = meta.get("policy", "unknown")

        return dynamics_type, policy, pos, vel, grounded
    except Exception:
        return None


def _load_raw_trajectories(collection_dir, workers=10, max_per_cell=50):
    """Load raw position/velocity data from a collection.

    Loads up to max_per_cell episodes per (dynamics_type, policy) cell
    to keep memory bounded. Groups data by dynamics_type and policy.

    Args:
        collection_dir: Path to collection root.
        workers: Parallel workers for loading.
        max_per_cell: Cap episodes per cell to limit memory.

    Returns:
        by_dynamics: dict[str, list of (pos, vel, grounded) arrays]
        by_policy: dict[str, list of (pos, vel, grounded) arrays]
        by_policy_dynamics: dict[str, dict[str, list of (pos, vel, grounded)]]
            Cross-indexed: by_policy_dynamics[policy][dynamics_type]
    """
    traj_dir = Path(collection_dir) / "trajectories"
    all_npz = sorted(traj_dir.rglob("*.npz"))

    # Sample up to max_per_cell per cell to bound memory
    # Group files by parent dir (dynamics type) and filename prefix (policy)
    cell_files = defaultdict(list)
    for f in all_npz:
        dt = f.parent.name
        policy = f.stem.rsplit("_", 1)[0]  # "rush_0001" -> "rush"
        cell_files[(dt, policy)].append(f)

    sampled = []
    rng = np.random.default_rng(42)
    for key, files in cell_files.items():
        if len(files) > max_per_cell:
            indices = rng.choice(len(files), size=max_per_cell, replace=False)
            sampled.extend(files[i] for i in indices)
        else:
            sampled.extend(files)

    # Load in parallel
    paths_str = [str(p) for p in sampled]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_load_positions_for_file, paths_str, chunksize=50))
    else:
        results = [_load_positions_for_file(p) for p in paths_str]

    # Group by dynamics type, by policy, and cross-indexed (policy, dynamics)
    by_dynamics = defaultdict(list)
    by_policy = defaultdict(list)
    by_policy_dynamics = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r is None:
            continue
        dynamics_type, policy, pos, vel, grounded = r
        by_dynamics[dynamics_type].append((pos, vel, grounded))
        by_policy[policy].append((pos, vel, grounded))
        by_policy_dynamics[policy][dynamics_type].append((pos, vel, grounded))

    return by_dynamics, by_policy, by_policy_dynamics


def plot_visitation_heatmap_by_dynamics(by_dynamics, output_dir):
    """Plot (x, y) visitation heatmaps — one panel per dynamics type.

    Aggregates all position samples into a 2D histogram. Shows where
    agents spend time in physical space for each dynamics type.
    """
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))

    # Determine global x/y range across all dynamics types for consistent axes
    all_x, all_y = [], []
    for trajs in by_dynamics.values():
        for pos, _, _ in trajs:
            all_x.append(pos[:, 0])
            all_y.append(pos[:, 1])
    if not all_x:
        plt.close(fig)
        return None
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    # Use percentile-based ranges to avoid outlier stretching
    x_range = (np.percentile(all_x, 1), np.percentile(all_x, 99))
    y_range = (np.percentile(all_y, 1), np.percentile(all_y, 99))
    bins_x = np.linspace(x_range[0], x_range[1], 60)
    bins_y = np.linspace(y_range[0], y_range[1], 40)

    for idx, dt in enumerate(_DYNAMICS_ORDER):
        ax = axes[idx // 4, idx % 4]

        if dt in by_dynamics:
            # Concatenate all positions for this dynamics type
            xs = np.concatenate([pos[:, 0] for pos, _, _ in by_dynamics[dt]])
            ys = np.concatenate([pos[:, 1] for pos, _, _ in by_dynamics[dt]])

            h, _, _ = np.histogram2d(xs, ys, bins=[bins_x, bins_y])
            # Log scale to show low-density regions
            h_log = np.log1p(h.T)  # transpose so y is vertical axis
            ax.imshow(h_log, origin="lower", aspect="auto", cmap="hot",
                      extent=[x_range[0], x_range[1], y_range[0], y_range[1]])
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")

        ax.set_title(dt.replace("_", "\n"), fontsize=8)
        if idx // 4 == 3:
            ax.set_xlabel("x (px)", fontsize=7)
        if idx % 4 == 0:
            ax.set_ylabel("y (px)", fontsize=7)
        ax.tick_params(labelsize=6)

    fig.suptitle("Visitation Heatmap (x, y) by Dynamics Type", fontsize=14)
    fig.tight_layout()
    return _save_fig(fig, output_dir, "visitation_heatmap_by_dynamics")


def plot_visitation_heatmap_by_policy(by_policy, output_dir):
    """Plot (x, y) visitation heatmaps — one panel per policy.

    Same as the dynamics version but grouped by policy to show how
    different behavioral strategies explore space differently.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    # Global ranges
    all_x, all_y = [], []
    for trajs in by_policy.values():
        for pos, _, _ in trajs:
            all_x.append(pos[:, 0])
            all_y.append(pos[:, 1])
    if not all_x:
        plt.close(fig)
        return None
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    x_range = (np.percentile(all_x, 1), np.percentile(all_x, 99))
    y_range = (np.percentile(all_y, 1), np.percentile(all_y, 99))
    bins_x = np.linspace(x_range[0], x_range[1], 60)
    bins_y = np.linspace(y_range[0], y_range[1], 40)

    for i, pol in enumerate(_POLICY_ORDER):
        ax = axes[i]
        if pol in by_policy:
            xs = np.concatenate([pos[:, 0] for pos, _, _ in by_policy[pol]])
            ys = np.concatenate([pos[:, 1] for pos, _, _ in by_policy[pol]])
            h, _, _ = np.histogram2d(xs, ys, bins=[bins_x, bins_y])
            h_log = np.log1p(h.T)
            ax.imshow(h_log, origin="lower", aspect="auto", cmap="hot",
                      extent=[x_range[0], x_range[1], y_range[0], y_range[1]])
        ax.set_title(pol, fontsize=11)
        ax.set_xlabel("x (px)")
        if i == 0:
            ax.set_ylabel("y (px)")

    fig.suptitle("Visitation Heatmap (x, y) by Policy", fontsize=14)
    fig.tight_layout()
    return _save_fig(fig, output_dir, "visitation_heatmap_by_policy")


# ===================================================================
# 9. Single-axis acceleration portraits (grouped by relevant model)
# ===================================================================

def _split_by_model_axis(by_dynamics):
    """Split trajectory data by vertical and horizontal model independently.

    Groups all trajectories that share the same vertical model (regardless
    of horizontal) and vice versa. This allows plotting each model axis
    in isolation.

    Returns:
        by_vertical: dict[str, list of (pos, vel, grounded)]
        by_horizontal: dict[str, list of (pos, vel, grounded)]
    """
    by_vertical = defaultdict(list)
    by_horizontal = defaultdict(list)
    for dt, trajs in by_dynamics.items():
        for v in _VERTICAL_ORDER:
            if dt.startswith(v + "_"):
                by_vertical[v].extend(trajs)
                h = dt[len(v) + 1:]
                by_horizontal[h].extend(trajs)
                break
    return by_vertical, by_horizontal


def _simulate_vertical_theoretical(v_name, vy_range, n_samples=20):
    """Simulate theoretical (vy, dvy) curves for a vertical dynamics model.

    Sweeps over the sampled physics parameter ranges (jump_height, jump_duration)
    and simulates a single jump for each combo. Returns arrays of (vy, dvy) points
    from the simulated trajectories.

    The equations (from src/dynamics.py):
      parabolic:  gravity(t) = base_gravity                    (constant)
      cubic:      gravity(t) = -|base_gravity| * alpha * t / jump_duration
      floaty:     gravity(t) = base_gravity * tanh(k * t)
      asymmetric: gravity = base_gravity * rise_mult (vy>0), * fall_mult (vy<=0)

    Where base_gravity = -2 * jump_height / jump_duration^2, dt = 1/60.
    """
    import math

    dt = 1.0 / 60  # physics timestep
    rng = np.random.default_rng(123)

    # Fixed model-specific params (not yet exposed to random sampling)
    cubic_alpha = 3.0
    floaty_k = 4.0
    rise_mult = 0.5
    fall_mult = 2.0

    # Sweep over physics config ranges
    jump_heights = rng.uniform(60.0, 200.0, size=n_samples)
    jump_durations = rng.uniform(0.25, 0.6, size=n_samples)

    all_vy, all_dvy = [], []
    for jh, jd in zip(jump_heights, jump_durations):
        base_gravity = -2 * jh / (jd ** 2)
        jump_impulse = -base_gravity * jd  # upward impulse (positive vy)

        # Simulate one jump
        vy = jump_impulse
        airtime = 0.0
        for _ in range(200):  # max steps
            # Compute gravity for this model
            if v_name == "parabolic":
                g = base_gravity
            elif v_name == "cubic":
                scaled_alpha = abs(base_gravity) * cubic_alpha / max(jd, 0.1)
                g = -scaled_alpha * airtime
            elif v_name == "floaty":
                g = base_gravity * math.tanh(floaty_k * airtime)
            elif v_name == "asymmetric":
                if vy > 0:
                    g = base_gravity * rise_mult
                else:
                    g = base_gravity * fall_mult
            else:
                break

            dvy = g * dt
            # Only record if in vy display range (avoid plotting outside axes)
            if vy_range[0] <= vy <= vy_range[1]:
                all_vy.append(vy)
                all_dvy.append(dvy)

            vy += dvy
            airtime += dt

            # Stop if fallen well past launch (trajectory complete)
            if vy < vy_range[0] * 1.5:
                break

    return np.array(all_vy), np.array(all_dvy)


def plot_vertical_acceleration(by_dynamics, output_dir):
    """Plot (vy, dvy) — 4 empirical panels + 1 theoretical overlay panel per vertical model.

    Row 1: empirical 2D histograms from trajectory data (airborne-only).
    Row 2: theoretical curves from the dynamics equations swept over param ranges.

    This verifies the vertical dynamics equations:
      - Parabolic: constant dvy (horizontal band) — gravity is constant
      - Cubic: nonlinear dvy(vy) — acceleration depends on airtime/velocity
      - Floaty: compressed dvy near vy=0 — reduced gravity at apex
      - Asymmetric: different dvy for vy>0 vs vy<0 — rise/fall asymmetry
    """
    by_vertical, _ = _split_by_model_axis(by_dynamics)

    # Compute airborne (vy, dvy) per vertical model
    accel_data = {}
    for v_name, trajs in by_vertical.items():
        vys, dvys = [], []
        for _, vel, grounded in trajs:
            vy = vel[:, 1]
            if len(vy) < 3:
                continue
            dvy = np.diff(vy)
            airborne_curr = grounded[:-1] < 0.5
            airborne_next = grounded[1:] < 0.5
            mask = airborne_curr & airborne_next
            if mask.sum() > 0:
                vys.append(vy[:-1][mask])
                dvys.append(dvy[mask])
        if vys:
            accel_data[v_name] = (np.concatenate(vys), np.concatenate(dvys))

    if not accel_data:
        return None

    # Global ranges
    all_vy = np.concatenate([v for v, _ in accel_data.values()])
    all_dvy = np.concatenate([d for _, d in accel_data.values()])
    vy_range = (np.percentile(all_vy, 5), np.percentile(all_vy, 95))
    dvy_range = (np.percentile(all_dvy, 5), np.percentile(all_dvy, 95))
    bins_vy = np.linspace(vy_range[0], vy_range[1], 60)
    bins_dvy = np.linspace(dvy_range[0], dvy_range[1], 60)

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    for i, v_name in enumerate(_VERTICAL_ORDER):
        # Row 1: empirical
        ax = axes[0, i]
        if v_name in accel_data:
            vy_arr, dvy_arr = accel_data[v_name]
            h, _, _ = np.histogram2d(vy_arr, dvy_arr, bins=[bins_vy, bins_dvy])
            ax.imshow(np.log1p(h.T), origin="lower", aspect="auto",
                      cmap="inferno",
                      extent=[vy_range[0], vy_range[1],
                              dvy_range[0], dvy_range[1]])
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title(f"{v_name} (empirical)", fontsize=10)
        if i == 0:
            ax.set_ylabel("dvy (px/s per step)")

        # Row 2: theoretical
        ax_t = axes[1, i]
        theo_vy, theo_dvy = _simulate_vertical_theoretical(v_name, vy_range)
        if len(theo_vy) > 0:
            h_t, _, _ = np.histogram2d(theo_vy, theo_dvy, bins=[bins_vy, bins_dvy])
            ax_t.imshow(np.log1p(h_t.T), origin="lower", aspect="auto",
                        cmap="inferno",
                        extent=[vy_range[0], vy_range[1],
                                dvy_range[0], dvy_range[1]])
        ax_t.set_title(f"{v_name} (theoretical)", fontsize=10)
        ax_t.set_xlabel("vy (px/s)")
        if i == 0:
            ax_t.set_ylabel("dvy (px/s per step)")

    fig.suptitle("Vertical Dynamics Verification (vy, dvy) — Airborne Only\n"
                 "Top: empirical (from trajectories) | Bottom: theoretical (from equations + param sweep)",
                 fontsize=12)
    fig.tight_layout()
    return _save_fig(fig, output_dir, "vertical_acceleration")


def _simulate_horizontal_theoretical(h_name, n_samples=30, steps_per_sample=300):
    """Simulate horizontal dynamics using minimal pymunk physics.

    The vertical theoretical uses pure math because gravity goes through
    pymunk's _velocity_func callback (fires every substep, so dvy = g*dt).

    Horizontal can't do that because:
    1. body.apply_force is consumed after the first of 3 substeps, so only
       dt/3 of the force integrates.
    2. pymunk's contact friction between player and platform creates grounded
       damping that depends on the constraint solver.

    This function creates a bare pymunk Space (body + ground segment), mirrors
    the exact force/damping/step sequence from entities.py, and measures dvx.
    No level generation, no pygame, no env overhead — just physics.
    """
    import pymunk

    dt = 1.0 / 60
    substeps = 3
    rng = np.random.default_rng(456)

    # Fixed model params (from dynamics.py defaults)
    velocity_scale = 1.0
    impulse_strength = 50.0
    drag_coefficient = 0.005

    all_vx, all_dvx = [], []

    for _ in range(n_samples):
        # Randomize physics params (same ranges as PhysicsConfig.sample_full)
        move_speed = rng.uniform(150, 400)
        accel_time = rng.uniform(0.05, 0.3)
        air_control = rng.uniform(0.1, 0.9)
        ground_friction = rng.uniform(0.0, 0.8)
        jump_height = rng.uniform(60, 200)
        jump_duration = rng.uniform(0.25, 0.6)
        move_accel = move_speed / accel_time

        # Gravity derived same way as PhysicsConfig
        gravity = -2.0 * jump_height / (jump_duration ** 2)
        jump_impulse = -gravity * jump_duration

        # --- Build minimal pymunk world ---
        space = pymunk.Space()
        space.gravity = (0, gravity)

        # Ground: wide static segment (mirrors Platform entity, friction=1.0)
        ground = pymunk.Segment(space.static_body, (-500, 0), (2000, 0), 5)
        ground.friction = 1.0
        space.add(ground)

        # Player body (mirrors Player entity: mass=1, no rotation)
        body = pymunk.Body(1.0, float("inf"))
        body.position = (100, 30)
        shape = pymunk.Poly.create_box(body, (20, 30))
        shape.friction = ground_friction
        space.add(body, shape)

        # Settle onto ground
        for _ in range(30):
            space.step(dt / substeps)
            if abs(body.velocity.y) < 0.1:
                break

        # --- Run simulation with varied actions ---
        airborne = False
        for step in range(steps_per_sample):
            vx_before = body.velocity.x

            # Action pattern: long right runs, some left, some idle, jumps.
            # Builds up high velocities to cover the full vx range.
            cycle = step % 60
            if cycle < 30:
                direction = 1.0
            elif cycle < 45:
                direction = -1.0
            elif cycle < 55:
                direction = 1.0
            else:
                direction = 0.0
            do_jump = (step % 11 == 0)

            is_grounded = not airborne
            control = 1.0 if is_grounded else air_control

            # Apply horizontal dynamics (mirrors Player._apply_horizontal_force)
            if h_name == "impulse":
                dv = direction * impulse_strength * control
                new_vx = max(-move_speed, min(move_speed, body.velocity.x + dv))
                body.velocity = (new_vx, body.velocity.y)
            else:
                if h_name == "force":
                    if direction > 0 and body.velocity.x >= move_speed:
                        force = 0.0
                    elif direction < 0 and body.velocity.x <= -move_speed:
                        force = 0.0
                    else:
                        force = direction * move_accel * control
                elif h_name == "velocity":
                    target = direction * move_speed * control * velocity_scale
                    force = (target - body.velocity.x) * 30.0
                elif h_name == "drag_limited":
                    drive = direction * move_accel * control
                    drag = -drag_coefficient * body.velocity.x * abs(body.velocity.x)
                    force = drive + drag
                else:
                    force = 0.0
                if force != 0.0:
                    body.apply_force_at_local_point((force, 0), (0, 0))

            # Jump (mirrors Player.update)
            if do_jump and is_grounded:
                body.apply_impulse_at_local_point((0, jump_impulse), (0, 0))

            # Damping (mirrors DynamicsModel.get_damping + Player.update)
            if is_grounded:
                damping = 1.0 - 0.05 * ground_friction
                body.velocity = (body.velocity.x * damping, body.velocity.y)

            # Physics step (mirrors PhysicsWorld.step: 3 substeps)
            for _ in range(substeps):
                space.step(dt / substeps)

            vx_after = body.velocity.x
            airborne = body.position.y > 35  # above ground level

            all_vx.append(vx_before)
            all_dvx.append(vx_after - vx_before)

    return np.array(all_vx), np.array(all_dvx)


def plot_horizontal_acceleration(by_dynamics, output_dir):
    """Plot (vx, dvx) — 4 empirical panels + 4 theoretical panels per horizontal model.

    Row 1: empirical 2D histograms from trajectory data (all timesteps).
    Row 2: theoretical curves from the dynamics equations swept over param ranges.

    Uses ALL timesteps (ground + airborne) because the horizontal model is active
    in both states — on the ground modulated by ground_friction, in air by air_control.

    This verifies the horizontal dynamics equations:
      - Force: smooth dvx spread — acceleration proportional to input, capped at move_speed
      - Velocity: sharp dvx peaks — spring toward target velocity
      - Impulse: discrete dvx clusters — burst-like velocity delta
      - Drag-limited: dvx compressed at high |vx| — quadratic drag caps acceleration
    """
    _, by_horizontal = _split_by_model_axis(by_dynamics)

    # Compute (vx, dvx) per horizontal model — all timesteps
    accel_data = {}
    for h_name, trajs in by_horizontal.items():
        vxs, dvxs = [], []
        for _, vel, _ in trajs:
            vx = vel[:, 0]
            if len(vx) < 3:
                continue
            dvx = np.diff(vx)
            vxs.append(vx[:-1])
            dvxs.append(dvx)
        if vxs:
            accel_data[h_name] = (np.concatenate(vxs), np.concatenate(dvxs))

    if not accel_data:
        return None

    # Global ranges
    all_vx = np.concatenate([v for v, _ in accel_data.values()])
    all_dvx = np.concatenate([d for _, d in accel_data.values()])
    vx_range = (np.percentile(all_vx, 5), np.percentile(all_vx, 95))
    dvx_range = (np.percentile(all_dvx, 5), np.percentile(all_dvx, 95))
    bins_vx = np.linspace(vx_range[0], vx_range[1], 60)
    bins_dvx = np.linspace(dvx_range[0], dvx_range[1], 60)

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    for i, h_name in enumerate(_HORIZONTAL_ORDER):
        # Row 1: empirical
        ax = axes[0, i]
        if h_name in accel_data:
            vx_arr, dvx_arr = accel_data[h_name]
            h, _, _ = np.histogram2d(vx_arr, dvx_arr, bins=[bins_vx, bins_dvx])
            ax.imshow(np.log1p(h.T), origin="lower", aspect="auto",
                      cmap="inferno",
                      extent=[vx_range[0], vx_range[1],
                              dvx_range[0], dvx_range[1]])
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title(f"{h_name} (empirical)", fontsize=10)
        if i == 0:
            ax.set_ylabel("dvx (px/s per step)")

        # Row 2: theoretical (simulated through actual physics engine)
        ax_t = axes[1, i]
        theo_vx, theo_dvx = _simulate_horizontal_theoretical(h_name)
        if len(theo_vx) > 0:
            h_t, _, _ = np.histogram2d(theo_vx, theo_dvx, bins=[bins_vx, bins_dvx])
            ax_t.imshow(np.log1p(h_t.T), origin="lower", aspect="auto",
                        cmap="inferno",
                        extent=[vx_range[0], vx_range[1],
                                dvx_range[0], dvx_range[1]])
        ax_t.set_title(f"{h_name} (simulated)", fontsize=10)
        ax_t.set_xlabel("vx (px/s)")
        if i == 0:
            ax_t.set_ylabel("dvx (px/s per step)")

    fig.suptitle("Horizontal Dynamics Verification (vx, dvx) — All Timesteps\n"
                 "Top: empirical (from trajectories) | Bottom: simulated (engine + param sweep)",
                 fontsize=12)
    fig.tight_layout()
    return _save_fig(fig, output_dir, "horizontal_acceleration")


# ===================================================================
# 10. Joint acceleration portrait (dvx, dvy) — the 16-type fingerprint
# ===================================================================

def plot_joint_acceleration_portrait_by_dynamics(by_dynamics, output_dir):
    """Plot (dvx, dvy) joint acceleration portraits — one panel per dynamics type.

    This is the key diagnostic plot for dynamics diversity. The vertical
    model controls dvy and the horizontal model controls dvx, so each of
    the 16 (vertical x horizontal) combinations should produce a distinct
    (dvx, dvy) distribution:
      - Rows differ vertically (dvy axis): parabolic=tight band,
        cubic=spread, floaty=compressed near 0, asymmetric=skewed
      - Columns differ horizontally (dvx axis): force=smooth spread,
        velocity=sharp peaks, impulse=discrete clusters,
        drag_limited=compressed at high speed

    Airborne-only: filters to timesteps where both current and next state
    are airborne, removing ground friction effects and jump impulse spikes.
    """
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))

    # Compute airborne (dvx, dvy) for each dynamics type
    accel_data = {}  # dynamics_type -> (all_dvx, all_dvy)
    for dt, trajs in by_dynamics.items():
        dvxs, dvys = [], []
        for _, vel, grounded in trajs:
            vx, vy = vel[:, 0], vel[:, 1]
            if len(vx) < 3:
                continue
            dvx = np.diff(vx)
            dvy = np.diff(vy)
            # Both current and next timestep must be airborne
            airborne_curr = grounded[:-1] < 0.5
            airborne_next = grounded[1:] < 0.5
            mask = airborne_curr & airborne_next
            if mask.sum() > 0:
                dvxs.append(dvx[mask])
                dvys.append(dvy[mask])
        if dvxs:
            accel_data[dt] = (np.concatenate(dvxs), np.concatenate(dvys))

    if not accel_data:
        plt.close(fig)
        return None

    # Global ranges — 5th-95th percentile for tight focus
    all_dvx = np.concatenate([d for d, _ in accel_data.values()])
    all_dvy = np.concatenate([d for _, d in accel_data.values()])
    dvx_range = (np.percentile(all_dvx, 5), np.percentile(all_dvx, 95))
    dvy_range = (np.percentile(all_dvy, 5), np.percentile(all_dvy, 95))
    bins_dvx = np.linspace(dvx_range[0], dvx_range[1], 60)
    bins_dvy = np.linspace(dvy_range[0], dvy_range[1], 60)

    for idx, dt in enumerate(_DYNAMICS_ORDER):
        ax = axes[idx // 4, idx % 4]

        if dt in accel_data:
            dvx_arr, dvy_arr = accel_data[dt]
            h, _, _ = np.histogram2d(dvx_arr, dvy_arr, bins=[bins_dvx, bins_dvy])
            ax.imshow(np.log1p(h.T), origin="lower", aspect="auto",
                      cmap="inferno",
                      extent=[dvx_range[0], dvx_range[1],
                              dvy_range[0], dvy_range[1]])
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")

        ax.set_title(dt.replace("_", "\n"), fontsize=8)
        if idx // 4 == 3:
            ax.set_xlabel("dvx (px/s per step)", fontsize=7)
        if idx % 4 == 0:
            ax.set_ylabel("dvy (px/s per step)", fontsize=7)
        ax.tick_params(labelsize=6)

    fig.suptitle("Joint Acceleration Portrait (dvx, dvy) by Dynamics Type — Airborne Only\n"
                 "Rows = vertical model (dvy axis) | Columns = horizontal model (dvx axis) | "
                 "Each cell should be unique",
                 fontsize=11)
    fig.tight_layout()
    return _save_fig(fig, output_dir, "joint_acceleration_by_dynamics")


def plot_joint_acceleration_per_policy(by_policy_dynamics, output_dir):
    """Plot (dvx, dvy) joint acceleration — one 4x4 dynamics grid per policy.

    Same as joint_acceleration_by_dynamics but with policy held constant.
    Differences between panels are purely due to dynamics, not behavior.
    """
    # Compute airborne (dvx, dvy) per (policy, dynamics) cell + global ranges
    all_dvx, all_dvy = [], []
    cell_accel = defaultdict(dict)

    for pol, dt_data in by_policy_dynamics.items():
        for dt, trajs in dt_data.items():
            dvxs, dvys = [], []
            for _, vel, grounded in trajs:
                vx, vy = vel[:, 0], vel[:, 1]
                if len(vx) < 3:
                    continue
                dvx = np.diff(vx)
                dvy = np.diff(vy)
                airborne_curr = grounded[:-1] < 0.5
                airborne_next = grounded[1:] < 0.5
                mask = airborne_curr & airborne_next
                if mask.sum() > 0:
                    dvxs.append(dvx[mask])
                    dvys.append(dvy[mask])
            if dvxs:
                dvx_cat = np.concatenate(dvxs)
                dvy_cat = np.concatenate(dvys)
                cell_accel[pol][dt] = (dvx_cat, dvy_cat)
                all_dvx.append(dvx_cat)
                all_dvy.append(dvy_cat)

    if not all_dvx:
        return {}

    all_dvx = np.concatenate(all_dvx)
    all_dvy = np.concatenate(all_dvy)
    dvx_range = (np.percentile(all_dvx, 5), np.percentile(all_dvx, 95))
    dvy_range = (np.percentile(all_dvy, 5), np.percentile(all_dvy, 95))
    bins_dvx = np.linspace(dvx_range[0], dvx_range[1], 60)
    bins_dvy = np.linspace(dvy_range[0], dvy_range[1], 60)

    figures = {}
    for pol in _POLICY_ORDER:
        if pol not in cell_accel:
            continue

        fig, axes = plt.subplots(4, 4, figsize=(16, 14))

        for idx, dt in enumerate(_DYNAMICS_ORDER):
            ax = axes[idx // 4, idx % 4]

            if dt in cell_accel[pol]:
                dvx_arr, dvy_arr = cell_accel[pol][dt]
                h, _, _ = np.histogram2d(dvx_arr, dvy_arr,
                                         bins=[bins_dvx, bins_dvy])
                ax.imshow(np.log1p(h.T), origin="lower", aspect="auto",
                          cmap="inferno",
                          extent=[dvx_range[0], dvx_range[1],
                                  dvy_range[0], dvy_range[1]])
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="gray")

            ax.set_title(dt.replace("_", "\n"), fontsize=8)
            if idx // 4 == 3:
                ax.set_xlabel("dvx (px/s per step)", fontsize=7)
            if idx % 4 == 0:
                ax.set_ylabel("dvy (px/s per step)", fontsize=7)
            ax.tick_params(labelsize=6)

        fig.suptitle(f"Joint Acceleration (dvx, dvy) — Policy: {pol} — Airborne Only\n"
                     "Each panel = one dynamics type, policy held constant → "
                     "16 distinct acceleration fingerprints",
                     fontsize=11)
        fig.tight_layout()
        name = f"joint_acceleration_{pol}"
        figures[name] = _save_fig(fig, output_dir, name)

    return figures


# ===================================================================
# Main report generator
# ===================================================================

def generate_report(
    df: pd.DataFrame,
    output_dir: str | Path,
    collection_dir: str | Path | None = None,
    min_episodes_per_cell: int = 10,
    workers: int = 10,
) -> dict:
    """Generate full diversity analysis report.

    Creates figures and a JSON summary in the output directory.

    Args:
        df: Metrics DataFrame from compute_collection_metrics().
        output_dir: Directory to write analysis outputs.
        collection_dir: Path to collection root (for loading raw trajectories).
                        If None, skips visitation/phase portrait plots.
        min_episodes_per_cell: Threshold for gap report.
        workers: Parallel workers for loading raw trajectories.

    Returns:
        Summary dict (also saved as JSON).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating diversity report in {output_dir}/")

    # ------------------------------------------------------------------
    # Generate figures from metrics DataFrame
    # ------------------------------------------------------------------
    figures = {}

    print("  Plotting coverage heatmap...")
    figures["coverage_heatmap"] = plot_coverage_heatmap(df, output_dir)

    print("  Plotting state-space occupancy...")
    figures["state_space_occupancy"] = plot_state_space_occupancy(df, output_dir)

    print("  Plotting policy divergence...")
    figures["policy_divergence"] = plot_policy_divergence(df, output_dir)

    print("  Plotting calibration summary...")
    cal_path = plot_calibration_summary(df, output_dir)
    if cal_path:
        figures["calibration_summary"] = cal_path

    print("  Plotting physics distributions...")
    figures["physics_distributions"] = plot_physics_distributions(df, output_dir)

    # ------------------------------------------------------------------
    # Generate figures from raw trajectories (position/velocity data)
    # ------------------------------------------------------------------
    if collection_dir is not None:
        print("  Loading raw trajectories for heatmaps...")
        by_dynamics, by_policy, by_policy_dynamics = _load_raw_trajectories(
            collection_dir, workers=workers, max_per_cell=50,
        )

        print("  Plotting visitation heatmaps (x, y)...")
        path = plot_visitation_heatmap_by_dynamics(by_dynamics, output_dir)
        if path:
            figures["visitation_heatmap_by_dynamics"] = path
        path = plot_visitation_heatmap_by_policy(by_policy, output_dir)
        if path:
            figures["visitation_heatmap_by_policy"] = path

        print("  Plotting vertical dynamics verification (vy, dvy)...")
        path = plot_vertical_acceleration(by_dynamics, output_dir)
        if path:
            figures["vertical_acceleration"] = path

        print("  Plotting horizontal dynamics verification (vx, dvx)...")
        path = plot_horizontal_acceleration(by_dynamics, output_dir)
        if path:
            figures["horizontal_acceleration"] = path

        print("  Plotting joint acceleration portraits (dvx, dvy)...")
        path = plot_joint_acceleration_portrait_by_dynamics(by_dynamics, output_dir)
        if path:
            figures["joint_acceleration_by_dynamics"] = path

        print("  Plotting per-policy joint acceleration portraits (dvx, dvy)...")
        paths = plot_joint_acceleration_per_policy(by_policy_dynamics, output_dir)
        figures.update(paths)
    else:
        print("  Skipping visitation/phase plots (no collection_dir provided)")

    # ------------------------------------------------------------------
    # Compute summary statistics
    # ------------------------------------------------------------------
    print("  Computing gap report...")
    gaps = compute_gap_report(df, min_episodes=min_episodes_per_cell)

    summary = {
        "total_episodes": len(df),
        "dynamics_types": sorted(df["dynamics_type"].unique().tolist()),
        "policies": sorted(df["policy"].unique().tolist()),
        "n_dynamics_types": df["dynamics_type"].nunique(),
        "n_policies": df["policy"].nunique(),

        # Outcome breakdown
        "outcomes": df["outcome"].value_counts().to_dict(),
        "completion_rate": float(df["completed"].mean()),

        # Per-dynamics-type completion rates
        "completion_by_dynamics": df.groupby("dynamics_type")["completed"].mean().to_dict(),

        # Per-policy completion rates
        "completion_by_policy": df.groupby("policy")["completed"].mean().to_dict(),

        # Trajectory length stats
        "episode_steps_mean": float(df["episode_steps"].mean()),
        "episode_steps_std": float(df["episode_steps"].std()),
        "episode_steps_min": int(df["episode_steps"].min()),
        "episode_steps_max": int(df["episode_steps"].max()),

        # Spatial coverage
        "unique_cells_mean": float(df["unique_cells_visited"].mean()),
        "max_x_mean": float(df["max_x"].mean()),

        # Action diversity
        "action_entropy_mean": float(df["action_entropy"].mean()),

        # Physics parameter ranges
        "physics_ranges": {
            col: {"min": float(df[col].min()), "max": float(df[col].max()),
                   "mean": float(df[col].mean()), "std": float(df[col].std())}
            for col in ["jump_height", "jump_duration", "move_speed",
                        "air_control", "ground_friction"]
            if col in df.columns
        },

        # Gap report
        "gaps": gaps,
        "n_gap_categories": len(gaps),
        "has_gaps": len(gaps) > 0,

        # Figures
        "figures": figures,
    }

    # Save JSON summary
    summary_path = output_dir / "diversity_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary saved to {summary_path}")

    # Save metrics DataFrame — prefer parquet, fall back to CSV
    try:
        metrics_path = output_dir / "metrics.parquet"
        df.to_parquet(metrics_path, index=False)
    except ImportError:
        metrics_path = output_dir / "metrics.csv"
        df.to_csv(metrics_path, index=False)
    print(f"  Metrics saved to {metrics_path}")

    # Print highlights
    print()
    print(f"=== Diversity Report Summary ===")
    print(f"Episodes: {len(df)}")
    print(f"Grid: {df['dynamics_type'].nunique()} dynamics types x {df['policy'].nunique()} policies")
    print(f"Completion rate: {df['completed'].mean():.1%}")
    print(f"Mean episode length: {df['episode_steps'].mean():.0f} steps")
    print(f"Action entropy: {df['action_entropy'].mean():.2f} bits")
    print(f"Spatial coverage: {df['unique_cells_visited'].mean():.1f} cells (mean)")

    if gaps:
        print(f"\nGaps found ({len(gaps)} categories):")
        for cat, items in gaps.items():
            print(f"  {cat}: {len(items)} cells")
    else:
        print(f"\nNo gaps detected.")

    print(f"\nFigures: {output_dir}/")

    return summary
