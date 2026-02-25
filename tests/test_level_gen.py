"""Tests for level generation."""

import os
import pytest

# Use dummy video driver for headless testing
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from parametric_physics_platformer.config import GameConfig, PhysicsConfig, LayoutConfig, DynamicsConfig, ObjectiveConfig
from parametric_physics_platformer.level_gen import LevelGenerator, LevelSpec, build_level
from parametric_physics_platformer.physics import PhysicsWorld, PhysicsParams


class TestLevelSpec:
    def test_has_required_fields(self):
        spec = LevelSpec(
            platforms=[(100, 50, 200)],
            goals=[(150, 100)],
            hazards=[],
            player_start=(50, 100),
            total_width=300,
            max_height=100,
            num_gaps=0,
            avg_gap_size=0,
        )
        assert spec.platforms == [(100, 50, 200)]
        assert spec.player_start == (50, 100)


class TestLevelGenerator:
    def test_initialization(self):
        config = GameConfig()
        gen = LevelGenerator.from_config(config)
        assert gen.max_jump_height > 0
        assert gen.max_jump_distance > 0

    def test_generate_returns_spec(self):
        config = GameConfig()
        gen = LevelGenerator.from_config(config)
        spec = gen.generate()

        assert isinstance(spec, LevelSpec)
        assert len(spec.platforms) > 0
        assert len(spec.goals) > 0

    def test_generate_with_seed_is_reproducible(self):
        config = GameConfig()
        gen = LevelGenerator.from_config(config)

        spec1 = gen.generate(seed=42)
        spec2 = gen.generate(seed=42)

        assert spec1.platforms == spec2.platforms
        assert spec1.goals == spec2.goals
        assert spec1.hazards == spec2.hazards

    def test_generate_different_seeds_produce_different_levels(self):
        config = GameConfig()
        gen = LevelGenerator.from_config(config)

        spec1 = gen.generate(seed=1)
        spec2 = gen.generate(seed=2)

        # At least platforms should differ
        assert spec1.platforms != spec2.platforms

    def test_player_start_on_ground(self):
        config = GameConfig()
        gen = LevelGenerator.from_config(config)
        spec = gen.generate(seed=42)

        # Player start should be above ground level
        assert spec.player_start[1] > 0

    def test_comfortable_distances_computed(self):
        """Verify comfortable distances are less than max."""
        physics = PhysicsConfig(jump_height=120, jump_duration=0.4)
        layout = LayoutConfig()
        dynamics = DynamicsConfig()
        objectives = ObjectiveConfig()

        gen = LevelGenerator(physics, layout, dynamics, objectives)

        assert gen.comfortable_height < gen.max_jump_height
        assert gen.comfortable_distance < gen.max_jump_distance

    def test_higher_jump_increases_reachability(self):
        """Higher jump height should increase reachable distance."""
        low_physics = PhysicsConfig(jump_height=60, jump_duration=0.4)
        high_physics = PhysicsConfig(jump_height=200, jump_duration=0.4)

        gen_low = LevelGenerator(low_physics, LayoutConfig(), DynamicsConfig(), ObjectiveConfig())
        gen_high = LevelGenerator(high_physics, LayoutConfig(), DynamicsConfig(), ObjectiveConfig())

        assert gen_high.max_jump_height > gen_low.max_jump_height

    def test_hazards_scale_with_density(self):
        """More hazard density should produce more hazards."""
        low_dynamics = DynamicsConfig(hazard_density=0.0)
        high_dynamics = DynamicsConfig(hazard_density=0.8)

        gen_low = LevelGenerator(
            PhysicsConfig(), LayoutConfig(), low_dynamics, ObjectiveConfig()
        )
        gen_high = LevelGenerator(
            PhysicsConfig(), LayoutConfig(), high_dynamics, ObjectiveConfig()
        )

        spec_low = gen_low.generate(seed=42)
        spec_high = gen_high.generate(seed=42)

        assert len(spec_low.hazards) < len(spec_high.hazards)

    def test_timed_hazard_ratio_splits_hazards(self):
        """timed_hazard_ratio should convert some hazards to timed."""
        dynamics = DynamicsConfig(hazard_density=0.5, timed_hazard_ratio=0.5)
        gen = LevelGenerator(PhysicsConfig(), LayoutConfig(), dynamics, ObjectiveConfig())
        spec = gen.generate(seed=42)

        total_hazards = len(spec.hazards) + len(spec.timed_hazards)
        assert total_hazards > 0
        assert len(spec.timed_hazards) > 0
        assert len(spec.hazards) > 0  # Some should remain static

    def test_timed_hazard_ratio_zero_means_no_timed(self):
        """No timed hazards when ratio is 0."""
        dynamics = DynamicsConfig(hazard_density=0.5, timed_hazard_ratio=0.0)
        gen = LevelGenerator(PhysicsConfig(), LayoutConfig(), dynamics, ObjectiveConfig())
        spec = gen.generate(seed=42)

        assert len(spec.timed_hazards) == 0
        assert len(spec.hazards) > 0

    def test_flashing_zones_placed(self):
        """flashing_zone_count should produce flashing zones."""
        dynamics = DynamicsConfig(flashing_zone_count=2)
        gen = LevelGenerator(PhysicsConfig(), LayoutConfig(), dynamics, ObjectiveConfig())
        spec = gen.generate(seed=42)

        assert len(spec.flashing_zones) == 2
        # Each zone spec: (x, y, width, safe_dur, deadly_dur)
        for zone in spec.flashing_zones:
            assert len(zone) == 5

    def test_springs_scale_with_density(self):
        """spring_density should produce springs on platforms."""
        dynamics_none = DynamicsConfig(spring_density=0.0)
        dynamics_some = DynamicsConfig(spring_density=0.4)

        gen_none = LevelGenerator(PhysicsConfig(), LayoutConfig(), dynamics_none, ObjectiveConfig())
        gen_some = LevelGenerator(PhysicsConfig(), LayoutConfig(), dynamics_some, ObjectiveConfig())

        spec_none = gen_none.generate(seed=42)
        spec_some = gen_some.generate(seed=42)

        assert len(spec_none.springs) == 0
        assert len(spec_some.springs) > 0

    def test_collectibles_placed(self):
        """collectibles count should produce collectibles in spec."""
        objectives = ObjectiveConfig(collectibles=5, airspace_collectibles=0)
        gen = LevelGenerator(PhysicsConfig(), LayoutConfig(), DynamicsConfig(), objectives)
        spec = gen.generate(seed=42)

        assert len(spec.collectibles) == 5
        # Each collectible spec: (x, y, value)
        for c in spec.collectibles:
            assert len(c) == 3

    def test_zero_collectibles(self):
        """No collectibles when count is 0."""
        objectives = ObjectiveConfig(collectibles=0, airspace_collectibles=0)
        gen = LevelGenerator(PhysicsConfig(), LayoutConfig(), DynamicsConfig(), objectives)
        spec = gen.generate(seed=42)

        assert len(spec.collectibles) == 0

    def test_airspace_collectibles(self):
        """Airspace collectibles placed in airspace region."""
        objectives = ObjectiveConfig(collectibles=0, airspace_collectibles=5)
        dynamics = DynamicsConfig()
        gen = LevelGenerator(PhysicsConfig(), LayoutConfig(), dynamics, objectives)
        spec = gen.generate(seed=42)

        assert len(spec.collectibles) == 5
        ground_y = 50  # ground platform y
        spring_reach = gen.max_jump_height * dynamics.spring_multiplier
        max_height = max(gen.max_jump_height * 1.5, spring_reach)
        for cx, cy, value in spec.collectibles:
            assert cy >= ground_y
            assert cy <= ground_y + max_height + 10  # small tolerance
            assert value == 1

    def test_both_collectible_types(self):
        """Platform and airspace collectibles combine."""
        objectives = ObjectiveConfig(collectibles=3, airspace_collectibles=4)
        gen = LevelGenerator(PhysicsConfig(), LayoutConfig(), DynamicsConfig(), objectives)
        spec = gen.generate(seed=42)

        assert len(spec.collectibles) == 7

    def test_goal_position_scales_with_distance(self):
        """Goal distance parameter should affect goal placement."""
        near_objectives = ObjectiveConfig(goal_distance=0.3)
        far_objectives = ObjectiveConfig(goal_distance=1.0)

        gen_near = LevelGenerator(
            PhysicsConfig(), LayoutConfig(), DynamicsConfig(), near_objectives
        )
        gen_far = LevelGenerator(
            PhysicsConfig(), LayoutConfig(), DynamicsConfig(), far_objectives
        )

        spec_near = gen_near.generate(seed=42)
        spec_far = gen_far.generate(seed=42)

        # Far goal should be on a platform that's further along
        # (Goals are placed on platforms based on goal_distance)
        assert spec_far.goals[0][0] >= spec_near.goals[0][0]


class TestBuildLevel:
    def test_build_level_creates_entities(self):
        """build_level should create actual game entities from spec."""
        spec = LevelSpec(
            platforms=[(100, 50, 200), (300, 150, 150)],
            goals=[(300, 200)],
            hazards=[(150, 30)],
            player_start=(50, 100),
            total_width=400,
            max_height=200,
            num_gaps=1,
            avg_gap_size=50,
        )

        physics = PhysicsWorld(PhysicsParams())
        platforms, goals, hazards, *_ = build_level(physics, spec)

        assert len(platforms) == 2
        assert len(goals) == 1
        assert len(hazards) == 1

    def test_build_level_creates_new_entity_types(self):
        """build_level should create timed hazards, flashing zones, springs, collectibles."""
        spec = LevelSpec(
            platforms=[(100, 50, 200)],
            goals=[(100, 100)],
            hazards=[],
            player_start=(50, 100),
            timed_hazards=[(200, 80, 2.0, 1.0)],
            flashing_zones=[(300, 60, 100, 2.0, 1.5)],
            springs=[(150, 70, 800.0)],
            collectibles=[(250, 90, 5)],
            total_width=400,
            max_height=200,
            num_gaps=0,
            avg_gap_size=0,
        )

        physics = PhysicsWorld(PhysicsParams())
        platforms, goals, hazards, timed_hazards, flashing_zones, springs, collectibles = build_level(physics, spec)

        assert len(timed_hazards) == 1
        assert len(flashing_zones) == 1
        assert len(springs) == 1
        assert len(collectibles) == 1
        assert collectibles[0].value == 5

    def test_build_level_positions_correct(self):
        """Entities should be positioned according to spec."""
        spec = LevelSpec(
            platforms=[(100, 50, 200)],
            goals=[(150, 100)],
            hazards=[],
            player_start=(50, 100),
            total_width=300,
            max_height=100,
            num_gaps=0,
            avg_gap_size=0,
        )

        physics = PhysicsWorld(PhysicsParams())
        platforms, goals, *_ = build_level(physics, spec)

        # Platform center should be at specified position
        assert platforms[0].x == pytest.approx(100, abs=1)
        assert platforms[0].y == pytest.approx(50, abs=1)

        # Goal should be at specified position
        assert goals[0].x == pytest.approx(150, abs=1)
        assert goals[0].y == pytest.approx(100, abs=1)
