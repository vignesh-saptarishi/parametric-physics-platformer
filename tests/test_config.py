"""Tests for configuration system with behavioral parameters."""

import pytest

from parametric_physics_platformer.config import (
    PhysicsConfig,
    LayoutConfig,
    DynamicsConfig,
    ObjectiveConfig,
    GameConfig,
    CONFIGS,
)


class TestPhysicsConfig:
    def test_defaults(self):
        config = PhysicsConfig()
        assert config.jump_height == 120.0
        assert config.jump_duration == 0.4
        assert config.move_speed == 250.0
        assert config.air_control == 0.5
        assert config.ground_friction == 0.3

    def test_derived_gravity(self):
        """Gravity is derived from jump_height and jump_duration."""
        config = PhysicsConfig(jump_height=120.0, jump_duration=0.4)
        # g = 2h/t² = 2*120/0.16 = 1500
        assert config.gravity == pytest.approx(-1500.0, rel=0.01)

    def test_derived_jump_impulse(self):
        """Jump impulse is derived from gravity and jump_duration."""
        config = PhysicsConfig(jump_height=120.0, jump_duration=0.4)
        # v0 = g*t = 1500*0.4 = 600
        assert config.jump_impulse == pytest.approx(600.0, rel=0.01)

    def test_behavioral_relationship(self):
        """Higher jump = more gravity needed for same duration."""
        low_jump = PhysicsConfig(jump_height=60.0, jump_duration=0.4)
        high_jump = PhysicsConfig(jump_height=180.0, jump_duration=0.4)

        # Higher jump needs stronger gravity (more negative)
        assert high_jump.gravity < low_jump.gravity

    def test_sample_jump_only(self):
        config = PhysicsConfig.sample_jump_only()
        # Jump params should be within range
        assert PhysicsConfig.JUMP_HEIGHT_RANGE[0] <= config.jump_height <= PhysicsConfig.JUMP_HEIGHT_RANGE[1]
        assert PhysicsConfig.JUMP_DURATION_RANGE[0] <= config.jump_duration <= PhysicsConfig.JUMP_DURATION_RANGE[1]
        # Other params should be defaults
        assert config.move_speed == 250.0

    def test_sample_full(self):
        config = PhysicsConfig.sample_full()
        # All params should be within ranges
        assert PhysicsConfig.JUMP_HEIGHT_RANGE[0] <= config.jump_height <= PhysicsConfig.JUMP_HEIGHT_RANGE[1]
        assert PhysicsConfig.JUMP_DURATION_RANGE[0] <= config.jump_duration <= PhysicsConfig.JUMP_DURATION_RANGE[1]
        assert PhysicsConfig.MOVE_SPEED_RANGE[0] <= config.move_speed <= PhysicsConfig.MOVE_SPEED_RANGE[1]
        assert PhysicsConfig.AIR_CONTROL_RANGE[0] <= config.air_control <= PhysicsConfig.AIR_CONTROL_RANGE[1]
        assert PhysicsConfig.GROUND_FRICTION_RANGE[0] <= config.ground_friction <= PhysicsConfig.GROUND_FRICTION_RANGE[1]

    def test_to_dict(self):
        config = PhysicsConfig(jump_height=100.0, jump_duration=0.5, ground_friction=0.4)
        d = config.to_dict()
        assert d["jump_height"] == 100.0
        assert d["jump_duration"] == 0.5
        assert d["ground_friction"] == 0.4
        # Derived values not in basic dict
        assert "gravity" not in d

    def test_to_dict_with_derived(self):
        config = PhysicsConfig(jump_height=100.0, jump_duration=0.5)
        d = config.to_dict_with_derived()
        assert d["jump_height"] == 100.0
        assert "gravity" in d
        assert "jump_impulse" in d

    def test_from_dict(self):
        d = {"jump_height": 150.0, "jump_duration": 0.35, "move_speed": 300.0, "accel_time": 0.1, "air_control": 0.6, "ground_friction": 0.5}
        config = PhysicsConfig.from_dict(d)
        assert config.jump_height == 150.0
        assert config.jump_duration == 0.35
        assert config.move_speed == 300.0
        assert config.ground_friction == 0.5


class TestGameConfig:
    def test_defaults(self):
        config = GameConfig()
        assert config.screen_width == 800
        assert config.screen_height == 600
        assert config.fps == 60
        assert isinstance(config.physics, PhysicsConfig)

    def test_sample_jump_only(self):
        config = GameConfig.sample_jump_only()
        assert isinstance(config.physics, PhysicsConfig)
        # Jump params should vary
        assert PhysicsConfig.JUMP_HEIGHT_RANGE[0] <= config.physics.jump_height <= PhysicsConfig.JUMP_HEIGHT_RANGE[1]

    def test_sample_full(self):
        config = GameConfig.sample_full()
        assert isinstance(config.physics, PhysicsConfig)
        assert isinstance(config.layout, LayoutConfig)
        assert isinstance(config.dynamics, DynamicsConfig)
        assert isinstance(config.objectives, ObjectiveConfig)


class TestPresets:
    def test_all_presets_exist(self):
        expected = ["default", "floaty", "tight", "moon", "heavy"]
        for name in expected:
            assert name in CONFIGS

    def test_preset_behavioral_values(self):
        """Presets should have distinct behavioral characteristics."""
        # Floaty = high jump, long duration
        assert CONFIGS["floaty"].physics.jump_height > CONFIGS["default"].physics.jump_height
        assert CONFIGS["floaty"].physics.jump_duration > CONFIGS["default"].physics.jump_duration

        # Tight = low jump, short duration
        assert CONFIGS["tight"].physics.jump_height < CONFIGS["default"].physics.jump_height
        assert CONFIGS["tight"].physics.jump_duration < CONFIGS["default"].physics.jump_duration

        # Moon = very high, very long (floaty)
        assert CONFIGS["moon"].physics.jump_height >= 200.0
        assert CONFIGS["moon"].physics.jump_duration >= 0.7

        # Heavy = low jump, fast fall
        assert CONFIGS["heavy"].physics.jump_height <= 80.0
        assert CONFIGS["heavy"].physics.jump_duration <= 0.3

    def test_preset_difficulty_ordering(self):
        """Jump height roughly corresponds to difficulty (lower = harder)."""
        heights = {name: cfg.physics.jump_height for name, cfg in CONFIGS.items()}
        # Moon should be easiest (highest jump), heavy hardest (lowest)
        assert heights["moon"] > heights["floaty"] > heights["default"] > heights["tight"] > heights["heavy"]


class TestDynamicsConfigDynamicsType:
    def test_default_dynamics_type_is_standard(self):
        dc = DynamicsConfig()
        assert dc.vertical_model == "parabolic"
        assert dc.horizontal_model == "force"

    def test_custom_dynamics_type(self):
        dc = DynamicsConfig(vertical_model="cubic", horizontal_model="drag_limited")
        assert dc.vertical_model == "cubic"
        assert dc.horizontal_model == "drag_limited"

    def test_sample_includes_dynamics_type(self):
        # Run several samples — at least one should differ from default
        found_non_default = False
        for _ in range(50):
            dc = DynamicsConfig.sample()
            assert dc.vertical_model in ("parabolic", "cubic", "floaty", "asymmetric")
            assert dc.horizontal_model in ("force", "velocity", "impulse", "drag_limited")
            if dc.vertical_model != "parabolic" or dc.horizontal_model != "force":
                found_non_default = True
        assert found_non_default
