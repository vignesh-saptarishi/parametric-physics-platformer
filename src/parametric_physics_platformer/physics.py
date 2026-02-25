"""Physics system using pymunk for 2D platformer dynamics.

This module handles all physics simulation including gravity, collisions,
and movement. Physics parameters are configurable for the world synthesis project.
"""

import pymunk
from typing import Tuple, Optional
from dataclasses import dataclass


# Collision types for different entity categories
COLLISION_PLAYER = 1
COLLISION_PLATFORM = 2
COLLISION_HAZARD = 3
COLLISION_GOAL = 4
COLLISION_SPRING = 5
COLLISION_COLLECTIBLE = 6


@dataclass
class PhysicsParams:
    """Physics parameters that affect movement feel.

    These are the tunable parameters for Phase 1 iteration.
    Starting with gravity only per the iteration block.
    """
    gravity: float = -900.0  # Pixels/sec^2, negative = down

    # Future parameters (to be enabled in later iterations):
    # friction: float = 0.7
    # air_control: float = 0.3
    # max_speed: float = 300.0


class PhysicsWorld:
    """Manages the pymunk physics simulation.

    Wraps pymunk.Space with game-specific configuration and helpers.
    """

    def __init__(self, params: Optional[PhysicsParams] = None):
        """Initialize physics world with given parameters.

        Args:
            params: Physics parameters. Uses defaults if None.
        """
        self.params = params or PhysicsParams()

        # Create pymunk space
        self.space = pymunk.Space()
        self.space.gravity = (0, self.params.gravity)

        # Collision handlers will be set up by entities
        self._setup_collision_handlers()

        # Track grounded state for player(s)
        self._grounded_bodies: set[pymunk.Body] = set()

        # Track hazard collisions (shape -> triggered)
        self._triggered_hazard_shapes: set[pymunk.Shape] = set()

    def _setup_collision_handlers(self) -> None:
        """Configure collision callbacks between entity types."""
        # Player <-> Platform collision
        self.space.on_collision(
            collision_type_a=COLLISION_PLAYER,
            collision_type_b=COLLISION_PLATFORM,
            begin=self._player_platform_begin,
            separate=self._player_platform_separate,
        )

        # Player <-> Hazard collision
        self.space.on_collision(
            collision_type_a=COLLISION_PLAYER,
            collision_type_b=COLLISION_HAZARD,
            begin=self._player_hazard_begin,
        )

    def _player_hazard_begin(
        self, arbiter: pymunk.Arbiter, _space: pymunk.Space, _data
    ) -> None:
        """Called when player touches a hazard."""
        hazard_shape = arbiter.shapes[1]  # shapes[1] is the hazard (collision_type_b)
        self._triggered_hazard_shapes.add(hazard_shape)

    def is_hazard_triggered(self, shape: pymunk.Shape) -> bool:
        """Check if a hazard shape was triggered."""
        return shape in self._triggered_hazard_shapes

    def clear_hazard_triggers(self) -> None:
        """Clear all hazard triggers (call on reset)."""
        self._triggered_hazard_shapes.clear()

    def _player_platform_begin(
        self, arbiter: pymunk.Arbiter, _space: pymunk.Space, _data
    ) -> None:
        """Called when player starts touching a platform."""
        player_shape = arbiter.shapes[0]

        # Check if collision is from above (player landing on platform)
        # Normal points from shape_a (player) to shape_b (platform)
        # When player lands on platform from above, normal points downward
        normal = arbiter.contact_point_set.normal
        if normal.y < -0.5:  # Collision from above (normal pointing down)
            self._grounded_bodies.add(player_shape.body)

        # Process collision normally (via arbiter.process_collision)
        arbiter.process_collision = True

    def _player_platform_separate(
        self, arbiter: pymunk.Arbiter, _space: pymunk.Space, _data
    ) -> None:
        """Called when player stops touching a platform."""
        player_shape = arbiter.shapes[0]
        self._grounded_bodies.discard(player_shape.body)

    def is_grounded(self, body: pymunk.Body) -> bool:
        """Check if a body is currently on ground."""
        return body in self._grounded_bodies

    def step(self, dt: float) -> None:
        """Advance physics simulation by dt seconds.

        Args:
            dt: Time step in seconds. Typically 1/60 for 60fps.
        """
        # Use multiple substeps for stability
        substeps = 3
        for _ in range(substeps):
            self.space.step(dt / substeps)

    def add_body(self, body: pymunk.Body, *shapes: pymunk.Shape) -> None:
        """Add a body and its shapes to the physics world."""
        self.space.add(body)
        for shape in shapes:
            self.space.add(shape)

    def remove_body(self, body: pymunk.Body) -> None:
        """Remove a body and all its shapes from the physics world."""
        for shape in body.shapes:
            self.space.remove(shape)
        self.space.remove(body)

    def remove_shape(self, shape: pymunk.Shape) -> None:
        """Remove a shape from the physics world (for static shapes)."""
        self.space.remove(shape)

    def create_static_segment(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        thickness: float = 5.0,
        collision_type: int = COLLISION_PLATFORM,
        friction: float = 1.0,
    ) -> pymunk.Shape:
        """Create a static line segment (useful for ground/walls).

        Args:
            p1: Start point (x, y)
            p2: End point (x, y)
            thickness: Line thickness for collision
            collision_type: Collision category
            friction: Surface friction coefficient

        Returns:
            The created shape (already added to space)
        """
        body = self.space.static_body
        shape = pymunk.Segment(body, p1, p2, thickness)
        shape.collision_type = collision_type
        shape.friction = friction
        self.space.add(shape)
        return shape

    def create_static_box(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        collision_type: int = COLLISION_PLATFORM,
        friction: float = 1.0,
    ) -> pymunk.Shape:
        """Create a static rectangular platform.

        Args:
            x, y: Center position
            width, height: Dimensions
            collision_type: Collision category
            friction: Surface friction coefficient

        Returns:
            The created shape (already added to space)
        """
        body = self.space.static_body
        # Create box vertices centered at origin, then offset
        half_w, half_h = width / 2, height / 2
        vertices = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        ]
        shape = pymunk.Poly(body, vertices, transform=pymunk.Transform.translation(x, y))
        shape.collision_type = collision_type
        shape.friction = friction
        self.space.add(shape)
        return shape

    def set_gravity(self, gravity: float) -> None:
        """Update gravity mid-simulation.

        Args:
            gravity: New gravity value (negative = down)
        """
        self.params.gravity = gravity
        self.space.gravity = (0, gravity)
