"""Behavioral calibration system for measuring actual physics outcomes.

PhysicsConfig parameters (jump_height, move_speed, etc.) are player attributes
that parameterize the equations of motion. Under parabolic dynamics, they map
directly to behavioral outcomes. Under non-standard dynamics (cubic, floaty,
asymmetric, velocity, impulse, drag), the actual behavior differs.

This module measures what the player ACTUALLY does for any given config +
dynamics combination by running canonical test actions in a headless
side-simulation. The results provide ground truth for:
- Level generation (use measured reachability, not declared params)
- Episode metadata (three-layer ground truth for probes)
- Scientific analysis (behavioral vs equation-level probes)

Usage:
    profile = calibrate(physics_config, dynamics_model)
    print(profile.actual_apex_height)  # measured, not declared
"""

from dataclasses import dataclass, asdict
from typing import Dict, Tuple

from .config import PhysicsConfig
from .physics import PhysicsWorld, PhysicsParams
from .entities import Player, Platform
from .dynamics import DynamicsModel


@dataclass
class BehavioralProfile:
    """Measured behavioral outcomes for a specific config + dynamics combination.

    These are empirical measurements from canonical test actions, not
    theoretical predictions. They provide ground truth for what the
    player can actually do under the given physics.

    Fields are grouped by which canonical test produces them:
    - Vertical: from a standing jump with no horizontal input
    - Horizontal: from running on flat ground
    - Combined: from jumping while running at speed
    """

    # --- Vertical (canonical jump test) ---
    actual_apex_height: float     # px - true max height reached above launch point
    actual_apex_time: float       # s - time from launch to apex
    actual_total_airtime: float   # s - time from launch to landing
    apex_dwell_fraction: float    # 0-1 - fraction of airtime spent in top 20% of apex
    trajectory_asymmetry: float   # rise_time / fall_time (1.0 = symmetric parabola)

    # --- Horizontal (canonical movement test) ---
    actual_max_speed: float       # px/s - measured top speed
    time_to_max_speed: float      # s - time from rest to 90% of max speed
    stopping_distance: float      # px - distance to stop from max speed, no input

    # --- Combined (jump + movement test) ---
    horizontal_jump_reach: float  # px - max x displacement during a full jump at speed

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "BehavioralProfile":
        """Create from dictionary."""
        return cls(**d)


# ---------------------------------------------------------------------------
# Calibration simulation
# ---------------------------------------------------------------------------

# Simulation parameters
_DT = 1.0 / 60.0          # 60fps timestep, matches game
_MAX_JUMP_FRAMES = 600     # 10s max for a single jump (safety cutoff)
_MAX_MOVEMENT_FRAMES = 600 # 10s max for movement test
_SPEED_STABLE_FRAMES = 10  # frames of stable speed to declare "at max"
_SPEED_STABLE_THRESHOLD = 0.5  # px/s change threshold for "stable"
_STOPPED_THRESHOLD = 1.0   # px/s below which player is "stopped"


def _create_test_world(
    physics_config: PhysicsConfig,
    dynamics_model: DynamicsModel,
) -> Tuple[PhysicsWorld, Player]:
    """Create a minimal physics world for calibration testing.

    Sets up: flat ground platform spanning 10000px, player standing at x=500.
    No hazards, no goals, no collectibles -- just physics.

    Returns:
        (physics_world, player) tuple ready for testing.
    """
    physics_params = PhysicsParams(gravity=physics_config.gravity)
    physics = PhysicsWorld(physics_params)

    # Wide flat ground so player can't fall off during horizontal tests
    ground_y = 50.0
    ground_width = 10000.0
    Platform(physics, ground_width / 2, ground_y, ground_width)

    # Create player at rest on ground
    player_x = 500.0
    player_start_y = ground_y + 30.0  # above ground surface
    player = Player(
        physics,
        (player_x, player_start_y),
        physics_config=physics_config,
        dynamics_model=dynamics_model,
    )

    # Settle: let player land on ground before testing
    for _ in range(30):
        player.update(_DT)
        physics.step(_DT)
        if player.is_grounded:
            break
    player._airtime = 0.0

    return physics, player


def _run_jump_test(
    physics_config: PhysicsConfig,
    dynamics_model: DynamicsModel,
) -> Tuple[float, float, float, float, float]:
    """Run canonical jump test: standing jump, no horizontal input.

    Returns:
        (apex_height, apex_time, total_airtime, dwell_fraction, asymmetry)
    """
    physics, player = _create_test_world(physics_config, dynamics_model)

    # Record ground position before jump
    _, ground_y = player.position

    # Trigger jump
    player.jump()
    player.update(_DT)
    physics.step(_DT)

    # Track trajectory
    max_height = 0.0
    apex_time = 0.0
    launch_frame = 0
    apex_frame = 0
    landed_frame = 0
    heights = []

    for frame in range(_MAX_JUMP_FRAMES):
        player.update(_DT)
        physics.step(_DT)

        _, py = player.position
        height = py - ground_y
        heights.append(height)

        if height > max_height:
            max_height = height
            apex_frame = frame

        # Detect landing: player was airborne and is now grounded
        if frame > 5 and player.is_grounded:
            landed_frame = frame
            break
    else:
        # Safety: didn't land within max frames
        landed_frame = len(heights) - 1

    # Compute metrics
    total_frames = landed_frame - launch_frame
    apex_height = max_height
    apex_time = (apex_frame + 1) * _DT  # +1 because frame 0 is first step after jump
    total_airtime = (total_frames + 1) * _DT

    # Rise time = frames to apex, fall time = frames from apex to landing
    rise_frames = apex_frame - launch_frame
    fall_frames = landed_frame - apex_frame
    trajectory_asymmetry = (rise_frames / max(fall_frames, 1))

    # Dwell fraction: time spent in top 20% of apex height
    dwell_threshold = max_height * 0.8
    dwell_frames = sum(1 for h in heights[:landed_frame] if h >= dwell_threshold)
    apex_dwell_fraction = dwell_frames / max(total_frames, 1)

    return apex_height, apex_time, total_airtime, apex_dwell_fraction, trajectory_asymmetry


def _run_movement_test(
    physics_config: PhysicsConfig,
    dynamics_model: DynamicsModel,
) -> Tuple[float, float, float]:
    """Run canonical movement test: hold right on flat ground, then release.

    Returns:
        (max_speed, time_to_max_speed, stopping_distance)
    """
    physics, player = _create_test_world(physics_config, dynamics_model)

    # Phase 1: Accelerate (hold right)
    max_speed = 0.0
    time_to_max = 0.0
    stable_count = 0
    prev_speed = 0.0
    reached_90_pct = False
    last_accel_frame = 0
    speed_90_time = 0.0

    for frame in range(_MAX_MOVEMENT_FRAMES):
        # Apply rightward movement
        player._apply_horizontal_force(1.0)
        player.update(_DT)
        physics.step(_DT)

        vx, _ = player.velocity
        speed = abs(vx)

        if speed > max_speed:
            max_speed = speed

        # Check if speed has stabilized
        if abs(speed - prev_speed) < _SPEED_STABLE_THRESHOLD:
            stable_count += 1
        else:
            stable_count = 0

        # Track time to 90% of eventual max speed
        if not reached_90_pct and max_speed > 0 and speed >= max_speed * 0.9:
            speed_90_time = (frame + 1) * _DT
            reached_90_pct = True

        if stable_count >= _SPEED_STABLE_FRAMES:
            last_accel_frame = frame
            break

        prev_speed = speed
        last_accel_frame = frame

    time_to_max = speed_90_time if reached_90_pct else (last_accel_frame + 1) * _DT

    # Phase 2: Decelerate (release input, measure stopping distance)
    start_x, _ = player.position
    stop_distance = 0.0

    for frame in range(_MAX_MOVEMENT_FRAMES):
        # No input -- just let physics + friction slow the player
        player.update(_DT)
        physics.step(_DT)

        vx, _ = player.velocity
        if abs(vx) < _STOPPED_THRESHOLD:
            break

    end_x, _ = player.position
    stop_distance = abs(end_x - start_x)

    return max_speed, time_to_max, stop_distance


def _run_combined_test(
    physics_config: PhysicsConfig,
    dynamics_model: DynamicsModel,
) -> float:
    """Run combined test: jump while moving right at speed.

    Accelerates to near max speed first, then jumps while continuing
    to hold right. Measures total horizontal displacement during the jump.

    Returns:
        horizontal_jump_reach in pixels
    """
    physics, player = _create_test_world(physics_config, dynamics_model)

    # First accelerate to near max speed
    for frame in range(120):  # 2 seconds of acceleration
        player._apply_horizontal_force(1.0)
        player.update(_DT)
        physics.step(_DT)

    # Record position at jump start
    start_x, _ = player.position

    # Jump while continuing to move right
    player.jump()

    for frame in range(_MAX_JUMP_FRAMES):
        player._apply_horizontal_force(1.0)
        player.update(_DT)
        physics.step(_DT)

        # Detect landing
        if frame > 5 and player.is_grounded:
            break

    end_x, _ = player.position
    return abs(end_x - start_x)


# ---------------------------------------------------------------------------
# Main calibration API
# ---------------------------------------------------------------------------

# Cache: maps config hash -> BehavioralProfile
_calibration_cache: Dict[str, BehavioralProfile] = {}


def _config_hash(
    physics_config: PhysicsConfig,
    dynamics_model: DynamicsModel,
) -> str:
    """Create a hashable key from config + dynamics for caching.

    Includes all parameters that affect behavior: PhysicsConfig fields,
    dynamics type (vertical + horizontal), and dynamics-specific params.
    """
    parts = [
        f"jh={physics_config.jump_height}",
        f"jd={physics_config.jump_duration}",
        f"ms={physics_config.move_speed}",
        f"at={physics_config.accel_time}",
        f"ac={physics_config.air_control}",
        f"gf={physics_config.ground_friction}",
        f"v={dynamics_model.vertical.name}",
        f"h={dynamics_model.horizontal.name}",
        f"bg={dynamics_model.vertical_params.base_gravity}",
        f"ca={dynamics_model.vertical_params.cubic_alpha}",
        f"fk={dynamics_model.vertical_params.floaty_k}",
        f"rm={dynamics_model.vertical_params.rise_multiplier}",
        f"fm={dynamics_model.vertical_params.fall_multiplier}",
        f"vs={dynamics_model.horizontal_params.velocity_scale}",
        f"is={dynamics_model.horizontal_params.impulse_strength}",
        f"dc={dynamics_model.horizontal_params.drag_coefficient}",
    ]
    return "|".join(parts)


def calibrate(
    physics_config: PhysicsConfig,
    dynamics_model: DynamicsModel,
    use_cache: bool = True,
) -> BehavioralProfile:
    """Measure actual behavioral outcomes for a config + dynamics combination.

    Runs three headless canonical tests:
    1. Standing jump (no horizontal input) -> vertical metrics
    2. Run on flat ground -> horizontal metrics
    3. Jump while running -> combined reach

    Results are cached by config hash so repeated calls are free.

    Args:
        physics_config: Player attribute configuration.
        dynamics_model: Active dynamics model (determines equations of motion).
        use_cache: Whether to use cached results. Set False for testing.

    Returns:
        BehavioralProfile with all measured outcomes.
    """
    key = _config_hash(physics_config, dynamics_model)
    if use_cache and key in _calibration_cache:
        return _calibration_cache[key]

    # Run the three canonical tests
    apex_height, apex_time, total_airtime, dwell_fraction, asymmetry = \
        _run_jump_test(physics_config, dynamics_model)

    max_speed, time_to_max, stop_distance = \
        _run_movement_test(physics_config, dynamics_model)

    jump_reach = _run_combined_test(physics_config, dynamics_model)

    profile = BehavioralProfile(
        actual_apex_height=apex_height,
        actual_apex_time=apex_time,
        actual_total_airtime=total_airtime,
        apex_dwell_fraction=dwell_fraction,
        trajectory_asymmetry=asymmetry,
        actual_max_speed=max_speed,
        time_to_max_speed=time_to_max,
        stopping_distance=stop_distance,
        horizontal_jump_reach=jump_reach,
    )

    if use_cache:
        _calibration_cache[key] = profile

    return profile


def clear_cache() -> None:
    """Clear the calibration cache. Useful for testing."""
    _calibration_cache.clear()
