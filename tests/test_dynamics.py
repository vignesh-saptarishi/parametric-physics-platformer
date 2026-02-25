"""Tests for dynamics models (equations of motion)."""

import pytest
import math
from parametric_physics_platformer.dynamics import (
    DynamicsModel,
    StandardDynamics,
    CubicDynamics,
    FloatyDynamics,
    AsymmetricDynamics,
    VelocityDynamics,
    ImpulseDynamics,
    DragDynamics,
    CompositeDynamics,
    create_dynamics,
    VerticalModel,
    HorizontalModel,
    VerticalParams,
    HorizontalParams,
)
from parametric_physics_platformer.config import PhysicsConfig


class TestStandardDynamics:
    """StandardDynamics should reproduce current behavior exactly."""

    def test_creates_with_defaults(self):
        model = StandardDynamics()
        assert model.vertical == VerticalModel.PARABOLIC
        assert model.horizontal == HorizontalModel.FORCE

    def test_creates_from_physics_config(self):
        pc = PhysicsConfig(jump_height=150.0, jump_duration=0.5)
        model = StandardDynamics(physics_config=pc)
        assert model.physics_config.jump_height == 150.0

    def test_get_gravity_returns_constant(self):
        pc = PhysicsConfig()
        model = StandardDynamics(physics_config=pc)
        # Constant gravity regardless of airtime
        g0 = model.get_gravity(airtime=0.0)
        g1 = model.get_gravity(airtime=0.5)
        g2 = model.get_gravity(airtime=1.0)
        assert g0 == g1 == g2
        assert g0 == (0, pc.gravity)

    def test_get_horizontal_force_respects_direction(self):
        pc = PhysicsConfig(move_speed=250.0)
        model = StandardDynamics(physics_config=pc)
        force_right = model.get_horizontal_force(direction=1.0, vx=0.0, is_grounded=True)
        force_left = model.get_horizontal_force(direction=-1.0, vx=0.0, is_grounded=True)
        assert force_right > 0
        assert force_left < 0

    def test_get_horizontal_force_caps_at_max_speed(self):
        pc = PhysicsConfig(move_speed=250.0)
        model = StandardDynamics(physics_config=pc)
        # At max speed, force should be zero
        force = model.get_horizontal_force(direction=1.0, vx=250.0, is_grounded=True)
        assert force == 0.0

    def test_get_horizontal_force_air_control(self):
        pc = PhysicsConfig(air_control=0.5)
        model = StandardDynamics(physics_config=pc)
        ground_force = model.get_horizontal_force(direction=1.0, vx=0.0, is_grounded=True)
        air_force = model.get_horizontal_force(direction=1.0, vx=0.0, is_grounded=False)
        assert abs(air_force) == pytest.approx(abs(ground_force) * 0.5, rel=0.01)

    def test_get_metadata(self):
        model = StandardDynamics()
        meta = model.get_metadata()
        assert meta["vertical"] == "parabolic"
        assert meta["horizontal"] == "force"
        assert "vertical_params" in meta
        assert "horizontal_params" in meta

    def test_get_type_id_returns_int(self):
        model = StandardDynamics()
        type_id = model.get_type_id()
        assert isinstance(type_id, int)
        assert type_id >= 0

    def test_get_damping_grounded(self):
        pc = PhysicsConfig(ground_friction=0.3)
        model = StandardDynamics(physics_config=pc)
        damping = model.get_damping(vx=100.0, is_grounded=True)
        expected = 1.0 - 0.05 * 0.3
        assert damping == pytest.approx(expected)

    def test_get_damping_airborne(self):
        model = StandardDynamics()
        damping = model.get_damping(vx=100.0, is_grounded=False)
        assert damping == 1.0


class TestCubicDynamics:
    """Cubic: gravity increases linearly with airtime. F(t) = -m*alpha*t."""

    def test_gravity_zero_at_launch(self):
        model = CubicDynamics()
        gx, gy = model.get_gravity(airtime=0.0)
        # At t=0, gravity should be zero
        assert gy == 0.0

    def test_gravity_increases_with_airtime(self):
        model = CubicDynamics()
        _, gy_early = model.get_gravity(airtime=0.1)
        _, gy_late = model.get_gravity(airtime=0.5)
        # Later airtime = stronger downward pull (more negative)
        assert gy_late < gy_early

    def test_gravity_at_jump_duration_reasonable(self):
        pc = PhysicsConfig(jump_duration=0.4)
        model = CubicDynamics(physics_config=pc)
        _, gy = model.get_gravity(airtime=pc.jump_duration)
        assert gy < 0  # Pointing down

    def test_horizontal_force_works(self):
        model = CubicDynamics()
        force = model.get_horizontal_force(direction=1.0, vx=0.0, is_grounded=True)
        assert force > 0

    def test_metadata_shows_cubic(self):
        model = CubicDynamics()
        assert model.get_metadata()["vertical"] == "cubic"


class TestFloatyDynamics:
    """Floaty: gravity follows tanh curve. Weak initially, snaps to full."""

    def test_gravity_weak_initially(self):
        pc = PhysicsConfig()
        model = FloatyDynamics(physics_config=pc)
        _, gy_early = model.get_gravity(airtime=0.05)
        # Early gravity should be much weaker than standard
        assert abs(gy_early) < abs(pc.gravity) * 0.3

    def test_gravity_approaches_full(self):
        pc = PhysicsConfig()
        model = FloatyDynamics(physics_config=pc)
        _, gy_late = model.get_gravity(airtime=2.0)
        # After long airtime, should approach full gravity
        assert abs(gy_late) > abs(pc.gravity) * 0.9

    def test_floaty_k_controls_snap_speed(self):
        pc = PhysicsConfig()
        slow = FloatyDynamics(physics_config=pc, vertical_params=VerticalParams(floaty_k=2.0))
        fast = FloatyDynamics(physics_config=pc, vertical_params=VerticalParams(floaty_k=8.0))
        _, gy_slow = slow.get_gravity(airtime=0.3)
        _, gy_fast = fast.get_gravity(airtime=0.3)
        # Higher k = faster approach to full gravity
        assert abs(gy_fast) > abs(gy_slow)

    def test_horizontal_force_works(self):
        model = FloatyDynamics()
        force = model.get_horizontal_force(direction=1.0, vx=0.0, is_grounded=True)
        assert force > 0

    def test_metadata_shows_floaty(self):
        model = FloatyDynamics()
        assert model.get_metadata()["vertical"] == "floaty"


class TestAsymmetricDynamics:
    """Asymmetric: different gravity during rise vs fall."""

    def test_rise_gravity_weaker(self):
        pc = PhysicsConfig()
        model = AsymmetricDynamics(
            physics_config=pc,
            vertical_params=VerticalParams(rise_multiplier=0.5, fall_multiplier=2.0),
        )
        rise_g = model.get_gravity_for_velocity(vy=100.0)
        fall_g = model.get_gravity_for_velocity(vy=-100.0)
        assert abs(rise_g[1]) < abs(fall_g[1])

    def test_multipliers_applied_correctly(self):
        pc = PhysicsConfig()
        model = AsymmetricDynamics(
            physics_config=pc,
            vertical_params=VerticalParams(rise_multiplier=0.5, fall_multiplier=2.0),
        )
        _, gy_rise = model.get_gravity_for_velocity(vy=100.0)
        assert gy_rise == pytest.approx(pc.gravity * 0.5, rel=0.01)
        _, gy_fall = model.get_gravity_for_velocity(vy=-100.0)
        assert gy_fall == pytest.approx(pc.gravity * 2.0, rel=0.01)

    def test_horizontal_force_works(self):
        model = AsymmetricDynamics()
        force = model.get_horizontal_force(direction=1.0, vx=0.0, is_grounded=True)
        assert force > 0

    def test_metadata_shows_asymmetric(self):
        model = AsymmetricDynamics()
        assert model.get_metadata()["vertical"] == "asymmetric"


class TestVelocityDynamics:
    """Velocity model: input maps directly to velocity."""

    def test_positive_input_gives_positive_velocity(self):
        model = VelocityDynamics()
        v = model.get_target_velocity(direction=1.0, is_grounded=True)
        assert v > 0

    def test_zero_input_gives_zero_velocity(self):
        model = VelocityDynamics()
        v = model.get_target_velocity(direction=0.0, is_grounded=True)
        assert v == 0.0

    def test_velocity_proportional_to_input(self):
        model = VelocityDynamics()
        v_half = model.get_target_velocity(direction=0.5, is_grounded=True)
        v_full = model.get_target_velocity(direction=1.0, is_grounded=True)
        assert v_full == pytest.approx(v_half * 2, rel=0.01)

    def test_air_control_reduces_velocity(self):
        pc = PhysicsConfig(air_control=0.5)
        model = VelocityDynamics(physics_config=pc)
        v_ground = model.get_target_velocity(direction=1.0, is_grounded=True)
        v_air = model.get_target_velocity(direction=1.0, is_grounded=False)
        assert v_air == pytest.approx(v_ground * 0.5, rel=0.01)

    def test_gravity_still_works(self):
        model = VelocityDynamics()
        _, gy = model.get_gravity(airtime=0.0)
        assert gy < 0  # Standard parabolic gravity

    def test_metadata_shows_velocity(self):
        model = VelocityDynamics()
        assert model.get_metadata()["horizontal"] == "velocity"


class TestImpulseDynamics:
    """Impulse model: input gives per-frame velocity change."""

    def test_impulse_nonzero_for_nonzero_input(self):
        model = ImpulseDynamics()
        dv = model.get_velocity_impulse(direction=1.0, is_grounded=True)
        assert dv > 0

    def test_impulse_scales_with_input(self):
        model = ImpulseDynamics()
        dv_half = model.get_velocity_impulse(direction=0.5, is_grounded=True)
        dv_full = model.get_velocity_impulse(direction=1.0, is_grounded=True)
        assert dv_full == pytest.approx(dv_half * 2, rel=0.01)

    def test_impulse_air_control(self):
        pc = PhysicsConfig(air_control=0.5)
        model = ImpulseDynamics(physics_config=pc)
        ground = model.get_velocity_impulse(direction=1.0, is_grounded=True)
        air = model.get_velocity_impulse(direction=1.0, is_grounded=False)
        assert air == pytest.approx(ground * 0.5, rel=0.01)

    def test_metadata_shows_impulse(self):
        model = ImpulseDynamics()
        assert model.get_metadata()["horizontal"] == "impulse"


class TestDragDynamics:
    """Drag model: force-based but with v^2 drag instead of speed cap."""

    def test_force_at_zero_velocity(self):
        model = DragDynamics()
        force = model.get_horizontal_force(direction=1.0, vx=0.0, is_grounded=True)
        assert force > 0

    def test_drag_reduces_net_force_at_speed(self):
        model = DragDynamics()
        force_slow = model.get_horizontal_force(direction=1.0, vx=50.0, is_grounded=True)
        force_fast = model.get_horizontal_force(direction=1.0, vx=200.0, is_grounded=True)
        assert force_fast < force_slow

    def test_terminal_velocity_exists(self):
        model = DragDynamics()
        # At high enough speed, drag exceeds drive force
        # drive = move_accel ~ 1667, drag = 0.005 * v^2
        # terminal: v = sqrt(drive/drag_coeff) ~ sqrt(1667/0.005) ~ 577
        force = model.get_horizontal_force(direction=1.0, vx=600.0, is_grounded=True)
        assert force <= 0

    def test_no_hard_speed_cap(self):
        model = DragDynamics()
        # Unlike force model, drag model still applies some drive near move_speed
        force = model.get_horizontal_force(direction=1.0, vx=249.0, is_grounded=True)
        assert force > 0

    def test_metadata_shows_drag(self):
        model = DragDynamics()
        assert model.get_metadata()["horizontal"] == "drag_limited"


class TestCompositeDynamics:
    """Any vertical model can combine with any horizontal model."""

    def test_cubic_with_drag(self):
        model = create_dynamics(
            vertical=VerticalModel.CUBIC,
            horizontal=HorizontalModel.DRAG_LIMITED,
        )
        assert model.vertical == VerticalModel.CUBIC
        assert model.horizontal == HorizontalModel.DRAG_LIMITED
        # Both axes work
        _, gy = model.get_gravity(airtime=0.3)
        assert gy < 0
        force = model.get_horizontal_force(direction=1.0, vx=0.0, is_grounded=True)
        assert force > 0

    def test_floaty_with_velocity(self):
        model = create_dynamics(
            vertical=VerticalModel.FLOATY,
            horizontal=HorizontalModel.VELOCITY,
        )
        assert model.vertical == VerticalModel.FLOATY
        assert model.horizontal == HorizontalModel.VELOCITY

    def test_asymmetric_with_impulse(self):
        model = create_dynamics(
            vertical=VerticalModel.ASYMMETRIC,
            horizontal=HorizontalModel.IMPULSE,
        )
        assert model.vertical == VerticalModel.ASYMMETRIC
        assert model.horizontal == HorizontalModel.IMPULSE
        # Asymmetric should still have get_gravity_for_velocity
        assert hasattr(model, 'get_gravity_for_velocity')

    def test_all_16_combinations_valid(self):
        for v in VerticalModel:
            for h in HorizontalModel:
                model = create_dynamics(vertical=v, horizontal=h)
                assert model.get_type_id() >= 0
                assert model.get_type_id() < 16

    def test_type_ids_unique(self):
        ids = set()
        for v in VerticalModel:
            for h in HorizontalModel:
                model = create_dynamics(vertical=v, horizontal=h)
                ids.add(model.get_type_id())
        assert len(ids) == 16

    def test_standard_combo_matches_standard(self):
        composite = create_dynamics(
            vertical=VerticalModel.PARABOLIC,
            horizontal=HorizontalModel.FORCE,
        )
        standard = StandardDynamics()
        # Same gravity
        assert composite.get_gravity(0.5) == standard.get_gravity(0.5)
        # Same force
        cf = composite.get_horizontal_force(1.0, 0.0, True)
        sf = standard.get_horizontal_force(1.0, 0.0, True)
        assert cf == pytest.approx(sf)

    def test_metadata_reflects_both_axes(self):
        model = create_dynamics(
            vertical=VerticalModel.FLOATY,
            horizontal=HorizontalModel.DRAG_LIMITED,
        )
        meta = model.get_metadata()
        assert meta["vertical"] == "floaty"
        assert meta["horizontal"] == "drag_limited"
