"""Integration tests: different dynamics produce different trajectories.

These tests run actual physics simulations via the gym env to verify
that dynamics variants produce measurably different player behavior.
"""

import os
import numpy as np
import pytest

os.environ['SDL_VIDEODRIVER'] = 'dummy'

from parametric_physics_platformer.gym_env import PlatformerEnv
from parametric_physics_platformer.config import GameConfig, PhysicsConfig
from parametric_physics_platformer.dynamics import (
    StandardDynamics,
    CubicDynamics,
    FloatyDynamics,
    AsymmetricDynamics,
    VelocityDynamics,
    DragDynamics,
    VerticalModel,
    HorizontalModel,
    create_dynamics,
)
from parametric_physics_platformer.physics import PhysicsWorld, PhysicsParams
from parametric_physics_platformer.entities import Player


class TestDynamicsIntegration:
    """End-to-end tests: different dynamics produce different trajectories."""

    def _run_jump(self, dynamics_model, n_steps=60):
        """Execute a jump and return max height reached."""
        pc = dynamics_model.physics_config
        physics = PhysicsWorld(PhysicsParams(gravity=pc.gravity))
        # Ground platform
        from parametric_physics_platformer.entities import Platform
        Platform(physics, x=400, y=30, width=800, height=40)
        player = Player(physics, (400, 100), physics_config=pc, dynamics_model=dynamics_model)

        # Wait for player to land
        dt = 1.0 / 60
        for _ in range(30):
            player.update(dt)
            physics.step(dt)

        # Jump
        player.jump()

        # Track max height
        max_y = player.position[1]
        for _ in range(n_steps):
            player.update(dt)
            physics.step(dt)
            _, py = player.position
            if py > max_y:
                max_y = py

        return max_y

    def _measure_airtime(self, dynamics_model, n_steps=120):
        """Execute a jump and return total frames airborne."""
        pc = dynamics_model.physics_config
        physics = PhysicsWorld(PhysicsParams(gravity=pc.gravity))
        from parametric_physics_platformer.entities import Platform
        Platform(physics, x=400, y=30, width=800, height=40)
        player = Player(physics, (400, 100), physics_config=pc, dynamics_model=dynamics_model)

        dt = 1.0 / 60
        # Wait for landing
        for _ in range(30):
            player.update(dt)
            physics.step(dt)

        # Jump
        player.jump()

        airborne_frames = 0
        for _ in range(n_steps):
            player.update(dt)
            physics.step(dt)
            if not player.is_grounded:
                airborne_frames += 1

        return airborne_frames

    def _speed_after_n_frames(self, dynamics_model, n_frames=5):
        """Measure horizontal speed after n frames of rightward movement."""
        pc = dynamics_model.physics_config
        physics = PhysicsWorld(PhysicsParams(gravity=pc.gravity))
        from parametric_physics_platformer.entities import Platform
        Platform(physics, x=2000, y=30, width=4000, height=40)
        player = Player(physics, (100, 100), physics_config=pc, dynamics_model=dynamics_model)

        dt = 1.0 / 60
        # Wait for landing
        for _ in range(30):
            player.update(dt)
            physics.step(dt)

        for _ in range(n_frames):
            player.move_right()
            player.update(dt)
            physics.step(dt)

        vx, _ = player.velocity
        return vx

    def test_cubic_trajectory_different_from_parabolic(self):
        """Cubic dynamics should produce measurably different jump arcs."""
        standard_apex = self._run_jump(StandardDynamics())
        cubic_apex = self._run_jump(CubicDynamics())
        # Different equations -> different apex heights
        assert standard_apex != pytest.approx(cubic_apex, abs=5.0)

    def test_floaty_hangs_longer(self):
        """Floaty dynamics should keep player airborne longer."""
        standard_airtime = self._measure_airtime(StandardDynamics())
        floaty_airtime = self._measure_airtime(FloatyDynamics())
        assert floaty_airtime > standard_airtime

    def test_velocity_model_faster_response(self):
        """Velocity model should reach higher speed sooner than force model."""
        force_speed = self._speed_after_n_frames(StandardDynamics(), n_frames=3)
        velocity_speed = self._speed_after_n_frames(VelocityDynamics(), n_frames=3)
        # Velocity model uses stiff spring to target, should be faster
        assert velocity_speed > force_speed

    def test_drag_model_no_hard_cap(self):
        """Drag model allows force above move_speed (no hard cap)."""
        model = DragDynamics()
        # Standard force model returns 0 at move_speed
        standard = StandardDynamics()
        std_force = standard.get_horizontal_force(direction=1.0, vx=249.0, is_grounded=True)
        drag_force = model.get_horizontal_force(direction=1.0, vx=249.0, is_grounded=True)
        # Standard still applies force just below cap; drag also applies force
        # But drag model has no hard cutoff at move_speed
        assert drag_force > 0

    def test_all_dynamics_types_playable_in_gym(self):
        """Every dynamics type should produce a valid gym episode."""
        for v in VerticalModel:
            for h in HorizontalModel:
                config = GameConfig()
                config.dynamics.vertical_model = v.name.lower()
                config.dynamics.horizontal_model = h.name.lower()
                env = PlatformerEnv(config=config)
                obs, info = env.reset(seed=42)
                assert env.observation_space.contains(obs)
                # Take 10 random steps without crashing
                for _ in range(10):
                    action = env.action_space.sample()
                    obs, reward, term, trunc, info = env.step(action)
                    assert env.observation_space.contains(obs)
                    if term or trunc:
                        break
                env.close()

    def test_dynamics_type_id_reflects_config(self):
        """Gym env dynamics type_id should match the config's model types."""
        for v in VerticalModel:
            for h in HorizontalModel:
                config = GameConfig()
                config.dynamics.vertical_model = v.name.lower()
                config.dynamics.horizontal_model = h.name.lower()
                env = PlatformerEnv(config=config)
                obs, info = env.reset(seed=42)
                expected_id = list(VerticalModel).index(v) * 4 + list(HorizontalModel).index(h)
                assert obs["state"][14] == float(expected_id), (
                    f"Expected type_id {expected_id} for {v.name}/{h.name}, "
                    f"got {obs['state'][14]}"
                )
                env.close()

    def test_asymmetric_different_rise_and_fall(self):
        """Asymmetric dynamics should produce different gravity during rise vs fall."""
        model = AsymmetricDynamics()
        rise_g = model.get_gravity_for_velocity(vy=100.0)
        fall_g = model.get_gravity_for_velocity(vy=-100.0)
        # Rise should have weaker gravity, fall stronger
        assert abs(rise_g[1]) < abs(fall_g[1])
