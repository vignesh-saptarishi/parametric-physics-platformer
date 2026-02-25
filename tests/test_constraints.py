"""Tests for parameter constraints and validated sampling."""

import os
import pytest

# Use dummy video driver for headless testing
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from parametric_physics_platformer.config import GameConfig, PhysicsConfig, LayoutConfig, DynamicsConfig
from parametric_physics_platformer.constraints import (
    ParameterConstraints,
    ConstrainedSampler,
    ConstraintResult,
    ConstraintViolation,
)


class TestConstraintResult:
    def test_valid_result_is_truthy(self):
        result = ConstraintResult(valid=True, violations=[])
        assert result
        assert bool(result) is True

    def test_invalid_result_is_falsy(self):
        violation = ConstraintViolation("param", "error message", "error")
        result = ConstraintResult(valid=False, violations=[violation])
        assert not result
        assert bool(result) is False


class TestParameterConstraints:
    def test_validate_physics_default_is_valid(self):
        physics = PhysicsConfig()
        result = ParameterConstraints.validate_physics(physics)
        assert result.valid

    def test_validate_physics_jump_too_low(self):
        physics = PhysicsConfig(jump_height=10)  # Below MIN of 40
        result = ParameterConstraints.validate_physics(physics)
        assert not result.valid
        assert any(v.param == "jump_height" for v in result.violations)

    def test_validate_physics_jump_very_high_is_warning(self):
        physics = PhysicsConfig(jump_height=500)  # Above MAX of 300
        result = ParameterConstraints.validate_physics(physics)
        # Warnings don't invalidate, but should be recorded
        violations = [v for v in result.violations if v.param == "jump_height"]
        assert len(violations) > 0
        assert violations[0].severity == "warning"

    def test_validate_physics_duration_too_short(self):
        physics = PhysicsConfig(jump_duration=0.05)  # Below MIN of 0.15
        result = ParameterConstraints.validate_physics(physics)
        assert not result.valid
        assert any(v.param == "jump_duration" for v in result.violations)

    def test_validate_physics_speed_too_slow(self):
        physics = PhysicsConfig(move_speed=50)  # Below MIN of 100
        result = ParameterConstraints.validate_physics(physics)
        assert not result.valid
        assert any(v.param == "move_speed" for v in result.violations)

    def test_validate_physics_air_control_out_of_range(self):
        physics = PhysicsConfig(air_control=1.5)  # Above MAX of 1.0
        result = ParameterConstraints.validate_physics(physics)
        assert not result.valid
        assert any(v.param == "air_control" for v in result.violations)

    def test_validate_layout_gaps_achievable(self):
        physics = PhysicsConfig(jump_height=120, move_speed=250)
        layout = LayoutConfig(gap_size_mean=50)  # Small gap
        result = ParameterConstraints.validate_layout(layout, physics)
        assert result.valid

    def test_validate_layout_gaps_too_large(self):
        physics = PhysicsConfig(jump_height=60, move_speed=100, jump_duration=0.25)
        layout = LayoutConfig(gap_size_mean=500)  # Too large for physics
        result = ParameterConstraints.validate_layout(layout, physics)
        assert not result.valid
        assert any(v.param == "gap_size_mean" for v in result.violations)

    def test_validate_layout_density_out_of_range(self):
        physics = PhysicsConfig()
        layout = LayoutConfig(platform_density=2.0)  # Above MAX of 1.0
        result = ParameterConstraints.validate_layout(layout, physics)
        assert not result.valid
        assert any(v.param == "platform_density" for v in result.violations)

    def test_validate_dynamics_default_is_valid(self):
        dynamics = DynamicsConfig()
        result = ParameterConstraints.validate_dynamics(dynamics)
        assert result.valid

    def test_validate_dynamics_hazard_density_out_of_range(self):
        dynamics = DynamicsConfig(hazard_density=1.5)
        result = ParameterConstraints.validate_dynamics(dynamics)
        assert not result.valid

    def test_validate_dynamics_high_hazard_is_warning(self):
        dynamics = DynamicsConfig(hazard_density=0.8)  # High but valid
        result = ParameterConstraints.validate_dynamics(dynamics)
        # Should be valid but with warning
        assert result.valid
        warnings = [v for v in result.violations if v.severity == "warning"]
        assert len(warnings) > 0

    def test_validate_config_aggregates_all(self):
        config = GameConfig()
        result = ParameterConstraints.validate_config(config)
        assert result.valid

    def test_validate_config_catches_physics_error(self):
        config = GameConfig(physics=PhysicsConfig(jump_height=10))
        result = ParameterConstraints.validate_config(config)
        assert not result.valid


class TestConstrainedSampler:
    def test_sample_physics_respects_bounds(self):
        sampler = ConstrainedSampler()

        for _ in range(10):
            physics = sampler.sample_physics()
            result = ParameterConstraints.validate_physics(physics)
            assert result.valid

    def test_sample_physics_with_custom_ranges(self):
        sampler = ConstrainedSampler()
        physics = sampler.sample_physics(
            jump_height_range=(80, 100),
            move_speed_range=(200, 250),
        )

        assert 80 <= physics.jump_height <= 100
        assert 200 <= physics.move_speed <= 250

    def test_sample_layout_constrained_by_physics(self):
        sampler = ConstrainedSampler()
        physics = sampler.sample_physics()

        for difficulty in [0.0, 0.5, 1.0]:
            layout = sampler.sample_layout(physics, difficulty)
            result = ParameterConstraints.validate_layout(layout, physics)
            # Should be valid - gaps should be achievable given physics
            errors = [v for v in result.violations if v.severity == "error"]
            assert len(errors) == 0

    def test_sample_layout_difficulty_affects_density(self):
        sampler = ConstrainedSampler()
        physics = sampler.sample_physics()

        easy_layout = sampler.sample_layout(physics, difficulty=0.0)
        hard_layout = sampler.sample_layout(physics, difficulty=1.0)

        # Easier = higher density (more platforms)
        assert easy_layout.platform_density > hard_layout.platform_density

    def test_sample_layout_difficulty_affects_gaps(self):
        sampler = ConstrainedSampler()
        physics = sampler.sample_physics()

        easy_layout = sampler.sample_layout(physics, difficulty=0.0)
        hard_layout = sampler.sample_layout(physics, difficulty=1.0)

        # Harder = larger gaps
        assert hard_layout.gap_size_mean > easy_layout.gap_size_mean

    def test_sample_dynamics_difficulty_affects_hazards(self):
        sampler = ConstrainedSampler()

        easy_dynamics = sampler.sample_dynamics(difficulty=0.0)
        hard_dynamics = sampler.sample_dynamics(difficulty=1.0)

        assert hard_dynamics.hazard_density > easy_dynamics.hazard_density

    def test_sample_config_produces_valid_config(self):
        sampler = ConstrainedSampler()

        for _ in range(5):
            config = sampler.sample_config()
            result = ParameterConstraints.validate_config(config)
            assert result.valid

    def test_sample_config_with_physics_overrides(self):
        sampler = ConstrainedSampler()
        config = sampler.sample_config(
            physics_overrides={"jump_height": 150, "move_speed": 300}
        )

        assert config.physics.jump_height == 150
        assert config.physics.move_speed == 300

        result = ParameterConstraints.validate_config(config)
        assert result.valid

    def test_sample_config_difficulty_range(self):
        sampler = ConstrainedSampler()

        easy_config = sampler.sample_config(difficulty=0.0)
        hard_config = sampler.sample_config(difficulty=1.0)

        # Easy should have higher platform density
        assert easy_config.layout.platform_density > hard_config.layout.platform_density

    def test_sample_difficulty_range_produces_valid_configs(self):
        sampler = ConstrainedSampler()
        configs = sampler.sample_difficulty_range(n_samples=5)

        assert len(configs) == 5

        for config in configs:
            result = ParameterConstraints.validate_config(config)
            assert result.valid

    def test_sample_difficulty_range_increases_difficulty(self):
        sampler = ConstrainedSampler()
        configs = sampler.sample_difficulty_range(n_samples=3)

        # Platform density should decrease (harder)
        densities = [c.layout.platform_density for c in configs]
        assert densities[0] > densities[1] > densities[2]

    def test_sample_config_raises_on_invalid_override(self):
        sampler = ConstrainedSampler()

        # This should fail validation (jump too low)
        with pytest.raises(ValueError, match="invalid config"):
            sampler.sample_config(physics_overrides={"jump_height": 10})
