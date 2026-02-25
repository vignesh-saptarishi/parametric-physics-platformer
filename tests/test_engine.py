"""Tests for game engine."""

import os
import pytest

# Use dummy video driver for headless testing
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from parametric_physics_platformer.config import GameConfig, PhysicsConfig, LayoutConfig, DynamicsConfig, ObjectiveConfig
from parametric_physics_platformer.engine import PlatformerEngine
from parametric_physics_platformer.level_gen import LevelSpec


class TestPlatformerEngine:
    def test_initialization(self):
        config = GameConfig()
        engine = PlatformerEngine(config)
        assert engine.physics is not None
        assert engine.player is None  # No player until level loaded

    def test_load_test_level(self):
        config = GameConfig()
        engine = PlatformerEngine(config)
        engine.load_test_level()

        assert engine.player is not None
        assert len(engine.platforms) > 0
        assert len(engine.goals) > 0

    def test_update_advances_physics(self):
        config = GameConfig()
        engine = PlatformerEngine(config)
        engine.load_test_level()

        # Player should be grounded after settle loop
        assert engine.player.is_grounded

        # Lift player off ground and verify they fall
        engine.player.body.position = (100, 300)
        engine.player.body.velocity = (0, 0)
        initial_y = engine.player.position[1]

        for _ in range(60):
            engine.update(1 / 60)

        # Player should have fallen
        assert engine.player.position[1] < initial_y

    def test_level_complete_on_goal(self):
        config = GameConfig()
        engine = PlatformerEngine(config)
        engine.load_test_level()

        assert not engine.level_complete

        # Trigger goal manually
        engine.goals[0].reached = True
        engine.update(1 / 60)

        assert engine.level_complete

    def test_player_dead_on_fall(self):
        config = GameConfig()
        engine = PlatformerEngine(config)
        engine.load_test_level()

        # Move player below screen
        engine.player.body.position = (100, -200)
        engine.update(1 / 60)

        assert engine.player_dead

    def test_reset_restores_state(self):
        config = GameConfig()
        engine = PlatformerEngine(config)
        engine.load_test_level()

        # Modify state
        engine.level_complete = True
        engine.player_dead = True

        # Reset
        engine.reset()

        assert not engine.level_complete
        assert not engine.player_dead
        assert engine.player is not None

    def test_get_state(self):
        config = GameConfig()
        engine = PlatformerEngine(config)
        engine.load_test_level()

        state = engine.get_state()

        assert "level_complete" in state
        assert "player_dead" in state
        assert "player_position" in state
        assert "player_velocity" in state
        assert "player_grounded" in state

    def test_different_jump_configs(self):
        """Different jump_height values produce different fall speeds."""
        results = {}

        # Test with different jump heights (behavioral param)
        # Same duration means higher jump = stronger gravity = faster fall
        for jump_height in [60.0, 120.0, 200.0]:
            config = GameConfig(physics=PhysicsConfig(jump_height=jump_height))
            engine = PlatformerEngine(config)
            engine.load_test_level()

            # Lift player off ground so they free-fall
            engine.player.body.position = (100, 300)
            engine.player.body.velocity = (0, 0)

            # Only run 10 frames so player is still falling (hasn't landed)
            for _ in range(10):
                engine.update(1 / 60)

            # Record velocity (not position, since they start at same place)
            results[jump_height] = engine.player.velocity[1]

        # Higher jump_height with same duration = stronger gravity = faster downward velocity
        # More negative = falling faster
        assert results[200.0] < results[120.0] < results[60.0]

    def test_load_generated_level(self):
        """Test loading a procedurally generated level."""
        config = GameConfig(
            layout=LayoutConfig(platform_density=0.5, gap_size_mean=50),
            dynamics=DynamicsConfig(hazard_density=0.2),
            objectives=ObjectiveConfig(goal_distance=0.8),
        )
        engine = PlatformerEngine(config)
        spec = engine.load_generated_level(seed=42)

        assert isinstance(spec, LevelSpec)
        assert engine.player is not None
        assert len(engine.platforms) > 0
        assert len(engine.goals) > 0

    def test_load_generated_level_is_reproducible(self):
        """Same seed should produce same level."""
        config = GameConfig()

        engine1 = PlatformerEngine(config)
        spec1 = engine1.load_generated_level(seed=123)

        engine2 = PlatformerEngine(config)
        spec2 = engine2.load_generated_level(seed=123)

        assert spec1.platforms == spec2.platforms
        assert spec1.goals == spec2.goals

    def test_load_generated_level_different_seeds(self):
        """Different seeds should produce different levels."""
        config = GameConfig()

        engine1 = PlatformerEngine(config)
        spec1 = engine1.load_generated_level(seed=1)

        engine2 = PlatformerEngine(config)
        spec2 = engine2.load_generated_level(seed=2)

        assert spec1.platforms != spec2.platforms

    def test_reset_after_generated_level_creates_new_level(self):
        """Reset with generated levels should create a new level."""
        config = GameConfig()
        engine = PlatformerEngine(config)
        spec1 = engine.load_generated_level(seed=42)
        platforms1 = [(p.x, p.y) for p in engine.platforms]

        # Reset should generate a new level
        engine.reset()
        platforms2 = [(p.x, p.y) for p in engine.platforms]

        # New level should be different (different random seed)
        assert platforms1 != platforms2
        assert engine.player is not None
        assert not engine.level_complete
        assert not engine.player_dead
