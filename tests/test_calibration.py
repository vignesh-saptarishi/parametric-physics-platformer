"""Tests for the behavioral calibration system.

Verifies that calibrate() produces correct BehavioralProfile measurements
for different PhysicsConfig + DynamicsModel combinations. Tests cover:
- Parabolic accuracy (measured vs declared params)
- Non-standard dynamics deviations (cubic, floaty, asymmetric)
- Horizontal model differentiation
- Caching behavior
- All 16 dynamics combinations
- Serialization round-trip
"""

import math
import pytest

from parametric_physics_platformer.calibration import calibrate, clear_cache, BehavioralProfile
from parametric_physics_platformer.config import PhysicsConfig
from parametric_physics_platformer.dynamics import (
    create_dynamics,
    VerticalModel,
    HorizontalModel,
    VerticalParams,
)


@pytest.fixture(autouse=True)
def _clear_calibration_cache():
    """Clear cache before each test so results are independent."""
    clear_cache()
    yield
    clear_cache()


def test_parabolic_matches_declared_params():
    """Parabolic dynamics should produce apex height and time close to declared values.

    Under constant gravity, the equations predict:
      apex_height = jump_height (by construction of PhysicsConfig)
      apex_time = jump_duration

    We allow 5% tolerance on height, 15% on time (discrete stepping introduces
    small errors, especially for time measurement).
    """
    pc = PhysicsConfig()
    dynamics = create_dynamics(
        vertical=VerticalModel.PARABOLIC,
        horizontal=HorizontalModel.FORCE,
        physics_config=pc,
    )
    profile = calibrate(pc, dynamics, use_cache=False)

    assert profile.actual_apex_height == pytest.approx(
        pc.jump_height, rel=0.05
    ), f"apex height {profile.actual_apex_height} not within 5% of {pc.jump_height}"

    assert profile.actual_apex_time == pytest.approx(
        pc.jump_duration, rel=0.15
    ), f"apex time {profile.actual_apex_time} not within 15% of {pc.jump_duration}"


def test_cubic_differs_from_declared():
    """Cubic dynamics should produce apex HIGHER than declared jump_height.

    Cubic gravity starts at zero (F(t) = -m*alpha*t) so gravity is weak early
    in the jump. The player rises longer/higher than parabolic would predict.
    """
    pc = PhysicsConfig()
    dynamics = create_dynamics(
        vertical=VerticalModel.CUBIC,
        horizontal=HorizontalModel.FORCE,
        physics_config=pc,
    )
    profile = calibrate(pc, dynamics, use_cache=False)

    assert profile.actual_apex_height > pc.jump_height, (
        f"cubic apex {profile.actual_apex_height} should exceed "
        f"declared jump_height {pc.jump_height}"
    )


def test_floaty_differs_from_declared():
    """Floaty dynamics should produce apex HIGHER than declared jump_height.

    Floaty gravity follows tanh(k*t), so gravity is near zero initially.
    The player rises higher before gravity ramps up.
    """
    pc = PhysicsConfig()
    dynamics = create_dynamics(
        vertical=VerticalModel.FLOATY,
        horizontal=HorizontalModel.FORCE,
        physics_config=pc,
    )
    profile = calibrate(pc, dynamics, use_cache=False)

    assert profile.actual_apex_height > pc.jump_height, (
        f"floaty apex {profile.actual_apex_height} should exceed "
        f"declared jump_height {pc.jump_height}"
    )


def test_asymmetric_differs_from_declared():
    """Asymmetric with rise_mult=0.5 should produce higher apex than parabolic.

    With rise_multiplier=0.5, gravity during the rise phase is half of normal.
    The player decelerates more slowly, reaching a higher apex.
    """
    pc = PhysicsConfig()

    parabolic = create_dynamics(
        vertical=VerticalModel.PARABOLIC,
        horizontal=HorizontalModel.FORCE,
        physics_config=pc,
    )
    asymmetric = create_dynamics(
        vertical=VerticalModel.ASYMMETRIC,
        horizontal=HorizontalModel.FORCE,
        physics_config=pc,
        vertical_params=VerticalParams(rise_multiplier=0.5, fall_multiplier=2.0),
    )

    profile_parabolic = calibrate(pc, parabolic, use_cache=False)
    profile_asymmetric = calibrate(pc, asymmetric, use_cache=False)

    assert profile_asymmetric.actual_apex_height > profile_parabolic.actual_apex_height, (
        f"asymmetric apex {profile_asymmetric.actual_apex_height} should exceed "
        f"parabolic apex {profile_parabolic.actual_apex_height}"
    )


def test_horizontal_models_differ():
    """Force, velocity, impulse, drag should produce different max speeds.

    Each horizontal model maps player input to motion differently, so even with
    the same PhysicsConfig, the measured top speed should vary across models.
    """
    pc = PhysicsConfig()
    speeds = {}

    for h_model in HorizontalModel:
        dynamics = create_dynamics(
            vertical=VerticalModel.PARABOLIC,
            horizontal=h_model,
            physics_config=pc,
        )
        profile = calibrate(pc, dynamics, use_cache=False)
        speeds[h_model.name] = profile.actual_max_speed

    # All four speeds should be positive
    for name, speed in speeds.items():
        assert speed > 0, f"{name} max speed should be positive, got {speed}"

    # At least some models must produce different speeds
    unique_speeds = set(round(s, 1) for s in speeds.values())
    assert len(unique_speeds) > 1, (
        f"expected different max speeds across horizontal models, got {speeds}"
    )


def test_cache_returns_same_result():
    """Calibrating twice with the same config should return identical profiles."""
    pc = PhysicsConfig()
    dynamics = create_dynamics(
        vertical=VerticalModel.PARABOLIC,
        horizontal=HorizontalModel.FORCE,
        physics_config=pc,
    )

    profile1 = calibrate(pc, dynamics, use_cache=True)
    profile2 = calibrate(pc, dynamics, use_cache=True)

    assert profile1 is profile2, (
        "second calibrate() call with same config should return cached object"
    )


def test_cache_miss_on_different_config():
    """Two different PhysicsConfigs should produce different profiles."""
    pc_low = PhysicsConfig(jump_height=80.0)
    pc_high = PhysicsConfig(jump_height=180.0)

    dynamics_low = create_dynamics(
        vertical=VerticalModel.PARABOLIC,
        horizontal=HorizontalModel.FORCE,
        physics_config=pc_low,
    )
    dynamics_high = create_dynamics(
        vertical=VerticalModel.PARABOLIC,
        horizontal=HorizontalModel.FORCE,
        physics_config=pc_high,
    )

    profile_low = calibrate(pc_low, dynamics_low, use_cache=True)
    profile_high = calibrate(pc_high, dynamics_high, use_cache=True)

    assert profile_low is not profile_high, (
        "different configs should produce different cached entries"
    )
    assert profile_low.actual_apex_height != pytest.approx(
        profile_high.actual_apex_height, rel=0.01
    ), "different jump_height configs should produce different apex heights"


def test_all_16_types_produce_valid_results():
    """Every vertical x horizontal combination should produce finite, non-negative results.

    This is a smoke test across all 16 dynamics types. Each measurement must be
    a finite number, and physical quantities must be non-negative. The
    apex_dwell_fraction must be between 0 and 1.
    """
    pc = PhysicsConfig()

    for v_model in VerticalModel:
        for h_model in HorizontalModel:
            dynamics = create_dynamics(
                vertical=v_model,
                horizontal=h_model,
                physics_config=pc,
            )
            profile = calibrate(pc, dynamics, use_cache=False)
            label = f"{v_model.name}+{h_model.name}"

            # All fields must be finite
            assert math.isfinite(profile.actual_apex_height), f"{label}: apex_height not finite"
            assert math.isfinite(profile.actual_apex_time), f"{label}: apex_time not finite"
            assert math.isfinite(profile.actual_total_airtime), f"{label}: total_airtime not finite"
            assert math.isfinite(profile.apex_dwell_fraction), f"{label}: dwell_fraction not finite"
            assert math.isfinite(profile.trajectory_asymmetry), f"{label}: asymmetry not finite"
            assert math.isfinite(profile.actual_max_speed), f"{label}: max_speed not finite"
            assert math.isfinite(profile.time_to_max_speed), f"{label}: time_to_max not finite"
            assert math.isfinite(profile.stopping_distance), f"{label}: stopping_distance not finite"
            assert math.isfinite(profile.horizontal_jump_reach), f"{label}: jump_reach not finite"

            # Physical quantities must be non-negative
            assert profile.actual_apex_height >= 0, f"{label}: apex_height negative"
            assert profile.actual_total_airtime >= 0, f"{label}: total_airtime negative"
            assert profile.actual_max_speed >= 0, f"{label}: max_speed negative"
            assert profile.stopping_distance >= 0, f"{label}: stopping_distance negative"
            assert profile.horizontal_jump_reach >= 0, f"{label}: jump_reach negative"

            # Dwell fraction bounded between 0 and 1
            assert 0 <= profile.apex_dwell_fraction <= 1, (
                f"{label}: dwell_fraction {profile.apex_dwell_fraction} not in [0, 1]"
            )


def test_behavioral_profile_serialization():
    """BehavioralProfile to_dict() and from_dict() should round-trip exactly."""
    pc = PhysicsConfig()
    dynamics = create_dynamics(
        vertical=VerticalModel.PARABOLIC,
        horizontal=HorizontalModel.FORCE,
        physics_config=pc,
    )
    profile = calibrate(pc, dynamics, use_cache=False)

    d = profile.to_dict()
    restored = BehavioralProfile.from_dict(d)

    # Every field must survive the round-trip
    assert restored.actual_apex_height == profile.actual_apex_height
    assert restored.actual_apex_time == profile.actual_apex_time
    assert restored.actual_total_airtime == profile.actual_total_airtime
    assert restored.apex_dwell_fraction == profile.apex_dwell_fraction
    assert restored.trajectory_asymmetry == profile.trajectory_asymmetry
    assert restored.actual_max_speed == profile.actual_max_speed
    assert restored.time_to_max_speed == profile.time_to_max_speed
    assert restored.stopping_distance == profile.stopping_distance
    assert restored.horizontal_jump_reach == profile.horizontal_jump_reach

    # Also verify the dict has exactly the expected keys
    expected_keys = {
        "actual_apex_height", "actual_apex_time", "actual_total_airtime",
        "apex_dwell_fraction", "trajectory_asymmetry", "actual_max_speed",
        "time_to_max_speed", "stopping_distance", "horizontal_jump_reach",
    }
    assert set(d.keys()) == expected_keys


def test_different_jump_heights_produce_different_apex():
    """Two PhysicsConfigs with different jump_height should produce different apex heights.

    Higher declared jump_height should result in higher actual apex, since
    gravity and impulse are both derived from jump_height.
    """
    pc_low = PhysicsConfig(jump_height=80.0)
    pc_high = PhysicsConfig(jump_height=180.0)

    dynamics_low = create_dynamics(
        vertical=VerticalModel.PARABOLIC,
        horizontal=HorizontalModel.FORCE,
        physics_config=pc_low,
    )
    dynamics_high = create_dynamics(
        vertical=VerticalModel.PARABOLIC,
        horizontal=HorizontalModel.FORCE,
        physics_config=pc_high,
    )

    profile_low = calibrate(pc_low, dynamics_low, use_cache=False)
    profile_high = calibrate(pc_high, dynamics_high, use_cache=False)

    assert profile_high.actual_apex_height > profile_low.actual_apex_height, (
        f"higher jump_height ({pc_high.jump_height}) should produce higher apex "
        f"({profile_high.actual_apex_height}) than lower jump_height "
        f"({pc_low.jump_height}, apex={profile_low.actual_apex_height})"
    )
