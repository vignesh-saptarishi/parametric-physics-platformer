"""Tests for physics system."""

import pytest
import pymunk

from parametric_physics_platformer.physics import PhysicsWorld, PhysicsParams, COLLISION_PLAYER, COLLISION_PLATFORM


class TestPhysicsParams:
    def test_default_gravity(self):
        params = PhysicsParams()
        assert params.gravity == -900.0

    def test_custom_gravity(self):
        params = PhysicsParams(gravity=-500.0)
        assert params.gravity == -500.0


class TestPhysicsWorld:
    def test_initialization(self):
        physics = PhysicsWorld()
        assert physics.space is not None
        assert physics.space.gravity == (0, -900.0)

    def test_custom_gravity(self):
        params = PhysicsParams(gravity=-400.0)
        physics = PhysicsWorld(params)
        assert physics.space.gravity == (0, -400.0)

    def test_set_gravity(self):
        physics = PhysicsWorld()
        physics.set_gravity(-1200.0)
        assert physics.space.gravity == (0, -1200.0)
        assert physics.params.gravity == -1200.0

    def test_step_advances_simulation(self):
        physics = PhysicsWorld()

        # Create a dynamic body
        body = pymunk.Body(1, 100)
        body.position = (100, 200)
        shape = pymunk.Circle(body, 10)
        physics.add_body(body, shape)

        initial_y = body.position.y

        # Step simulation
        for _ in range(60):
            physics.step(1 / 60)

        # Body should have fallen
        assert body.position.y < initial_y

    def test_create_static_box(self):
        physics = PhysicsWorld()
        shape = physics.create_static_box(100, 50, 200, 40)

        assert shape is not None
        assert shape.collision_type == COLLISION_PLATFORM
        assert shape in physics.space.shapes

    def test_grounded_detection(self):
        physics = PhysicsWorld()

        # Create platform
        physics.create_static_box(100, 30, 200, 40, collision_type=COLLISION_PLATFORM)

        # Create dynamic body (player-like)
        body = pymunk.Body(1, pymunk.moment_for_box(1, (32, 48)))
        body.position = (100, 200)
        shape = pymunk.Poly.create_box(body, (32, 48))
        shape.collision_type = COLLISION_PLAYER
        physics.add_body(body, shape)

        # Initially not grounded
        assert not physics.is_grounded(body)

        # Let it fall and land
        for _ in range(100):
            physics.step(1 / 60)

        # Should now be grounded
        assert physics.is_grounded(body)
