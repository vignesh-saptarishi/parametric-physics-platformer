"""Game entities: Player, platforms, hazards, goals.

Each entity wraps a pymunk body/shape with game-specific behavior.
"""

import pymunk
from typing import Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass

from .physics import (
    PhysicsWorld,
    COLLISION_PLAYER,
    COLLISION_PLATFORM,
    COLLISION_HAZARD,
    COLLISION_GOAL,
    COLLISION_SPRING,
    COLLISION_COLLECTIBLE,
)
from .config import PhysicsConfig

if TYPE_CHECKING:
    from .dynamics import DynamicsModel


@dataclass
class PlayerVisuals:
    """Player visual/collision configuration (non-behavioral)."""
    width: float = 32.0
    height: float = 48.0
    mass: float = 1.0


class Player:
    """Player entity with movement controls.

    Movement parameters come from PhysicsConfig (behavioral).
    Visual parameters come from PlayerVisuals.
    """

    def __init__(
        self,
        physics: PhysicsWorld,
        position: Tuple[float, float],
        physics_config: Optional[PhysicsConfig] = None,
        visuals: Optional[PlayerVisuals] = None,
        dynamics_model: Optional["DynamicsModel"] = None,
    ):
        """Create player at given position.

        Args:
            physics: The physics world to add player to
            position: Initial (x, y) position
            physics_config: Behavioral parameters. Uses defaults if None.
            visuals: Visual/collision config. Uses defaults if None.
            dynamics_model: Dynamics model for equations of motion. Defaults to StandardDynamics.
        """
        self.physics = physics
        self.physics_config = physics_config or PhysicsConfig()
        self.config = visuals or PlayerVisuals()  # For backwards compat

        # Set up dynamics model (lazy import to avoid circular)
        if dynamics_model is not None:
            self.dynamics_model = dynamics_model
        else:
            from .dynamics import StandardDynamics
            self.dynamics_model = StandardDynamics(physics_config=self.physics_config)

        # Create pymunk body and shape
        self.body = pymunk.Body(self.config.mass, pymunk.moment_for_box(
            self.config.mass, (self.config.width, self.config.height)
        ))
        self.body.position = position

        # Prevent rotation - player stays upright
        self.body.moment = float('inf')

        # Set custom velocity function for per-body gravity control
        self.body.velocity_func = self._velocity_func

        # Create collision shape
        self.shape = pymunk.Poly.create_box(
            self.body, (self.config.width, self.config.height)
        )
        self.shape.collision_type = COLLISION_PLAYER
        # Shape friction based on ground_friction config (0=ice, 1=sticky)
        self.shape.friction = self.physics_config.ground_friction

        # Add to physics world
        physics.add_body(self.body, self.shape)

        # Track state
        self._jump_requested = False
        self._airtime = 0.0

    @property
    def position(self) -> Tuple[float, float]:
        """Current position (x, y)."""
        return self.body.position.x, self.body.position.y

    @property
    def velocity(self) -> Tuple[float, float]:
        """Current velocity (vx, vy)."""
        return self.body.velocity.x, self.body.velocity.y

    @property
    def is_grounded(self) -> bool:
        """Whether player is on ground/platform."""
        return self.physics.is_grounded(self.body)

    @property
    def airtime(self) -> float:
        """Seconds since player left ground (0 if grounded)."""
        return self._airtime

    def _velocity_func(self, body, gravity, damping, dt):
        """Custom pymunk velocity function using dynamics model gravity."""
        from .dynamics import VerticalModel
        # Only asymmetric needs vy to pick rise/fall multiplier.
        # All other models use airtime. Using hasattr would break
        # CompositeDynamics which always has get_gravity_for_velocity
        # but falls back to get_gravity(0.0) for non-asymmetric.
        if self.dynamics_model.vertical == VerticalModel.ASYMMETRIC:
            custom_gravity = self.dynamics_model.get_gravity_for_velocity(body.velocity.y)
        else:
            custom_gravity = self.dynamics_model.get_gravity(self._airtime)
        pymunk.Body.update_velocity(body, custom_gravity, damping, dt)

    def move_left(self) -> None:
        """Apply leftward movement force."""
        self._apply_horizontal_force(-1)

    def move_right(self) -> None:
        """Apply rightward movement force."""
        self._apply_horizontal_force(1)

    def _apply_horizontal_force(self, direction: float) -> None:
        """Apply horizontal movement using the dynamics model.

        For FORCE/DRAG models: applies force via pymunk.
        For VELOCITY model: applies stiff spring force toward target.
        For IMPULSE model: applies direct velocity delta.
        """
        vx = self.velocity[0]

        # Impulse model: apply velocity change directly
        if hasattr(self.dynamics_model, 'get_velocity_impulse'):
            from .dynamics import HorizontalModel
            if self.dynamics_model.horizontal == HorizontalModel.IMPULSE:
                dv = self.dynamics_model.get_velocity_impulse(
                    direction=direction, is_grounded=self.is_grounded,
                )
                max_v = self.physics_config.move_speed
                new_vx = vx + dv
                # Clamp to max speed
                new_vx = max(-max_v, min(max_v, new_vx))
                self.body.velocity = (new_vx, self.body.velocity.y)
                return

        # Force/velocity/drag models: apply force via pymunk
        force = self.dynamics_model.get_horizontal_force(
            direction=direction, vx=vx, is_grounded=self.is_grounded,
        )
        if force != 0.0:
            self.body.apply_force_at_local_point((force, 0), (0, 0))

    def jump(self) -> None:
        """Request a jump. Only executes if grounded."""
        self._jump_requested = True

    def update(self, dt: Optional[float] = None) -> None:
        """Process pending actions. Call once per frame before physics step.

        Args:
            dt: Time step in seconds. Used for airtime tracking.
                Defaults to 1/60 for backward compatibility.
        """
        if dt is None:
            dt = 1 / 60

        # Track airtime for dynamics model
        if self.is_grounded:
            self._airtime = 0.0
        else:
            self._airtime += dt

        # Process jump request using behavioral jump_impulse
        if self._jump_requested and self.is_grounded:
            self.body.apply_impulse_at_local_point(
                (0, self.physics_config.jump_impulse), (0, 0)
            )
        self._jump_requested = False

        # Apply friction/damping
        vx, vy = self.velocity
        damping = self.dynamics_model.get_damping(vx, self.is_grounded)
        if damping < 1.0:
            self.body.velocity = (vx * damping, vy)


class Platform:
    """Static platform entity.

    Thin wrapper around physics static box for consistency.
    """

    def __init__(
        self,
        physics: PhysicsWorld,
        x: float,
        y: float,
        width: float,
        height: float = 20.0,
    ):
        """Create platform at given position.

        Args:
            physics: The physics world
            x, y: Center position
            width: Platform width
            height: Platform height (thickness)
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.shape = physics.create_static_box(
            x, y, width, height,
            collision_type=COLLISION_PLATFORM,
            friction=1.0,
        )

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get (left, bottom, right, top) bounds."""
        half_w = self.width / 2
        half_h = self.height / 2
        return (
            self.x - half_w,
            self.y - half_h,
            self.x + half_w,
            self.y + half_h,
        )


class Goal:
    """Goal area that player must reach.

    Triggers level completion when player touches it.
    """

    def __init__(
        self,
        physics: PhysicsWorld,
        x: float,
        y: float,
        width: float = 40.0,
        height: float = 60.0,
    ):
        """Create goal area.

        Args:
            physics: The physics world
            x, y: Center position
            width, height: Goal area dimensions
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.reached = False

        # Create sensor (non-solid collision shape)
        self.shape = physics.create_static_box(
            x, y, width, height,
            collision_type=COLLISION_GOAL,
            friction=0.0,
        )
        self.shape.sensor = True  # Don't block movement

        # Set up goal collision handler using new pymunk API
        physics.space.on_collision(
            collision_type_a=COLLISION_PLAYER,
            collision_type_b=COLLISION_GOAL,
            begin=self._on_player_enter,
        )

    def _on_player_enter(
        self, arbiter: pymunk.Arbiter, _space: pymunk.Space, _data
    ) -> None:
        """Called when player enters goal area."""
        self.reached = True
        arbiter.process_collision = False  # Don't process as solid collision

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get (left, bottom, right, top) bounds."""
        half_w = self.width / 2
        half_h = self.height / 2
        return (
            self.x - half_w,
            self.y - half_h,
            self.x + half_w,
            self.y + half_h,
        )


class Hazard:
    """Hazard that kills player on contact.

    Can be static or moving (patrol between bounds).
    """

    def __init__(
        self,
        physics: PhysicsWorld,
        x: float,
        y: float,
        width: float = 30.0,
        height: float = 30.0,
        speed: float = 0.0,
        patrol_distance: float = 100.0,
    ):
        """Create hazard area.

        Args:
            physics: The physics world
            x, y: Center position
            width, height: Hazard dimensions
            speed: Movement speed in pixels/sec (0 = static)
            patrol_distance: Total distance to patrol (centered on start position)
        """
        self.physics = physics
        self._start_x = x
        self._x = x
        self._y = y
        self.width = width
        self.height = height
        self.speed = speed
        self.patrol_distance = patrol_distance
        self._direction = 1.0  # 1 = right, -1 = left

        # For moving hazards, use kinematic body; for static, use static shape
        if speed > 0:
            # Kinematic body - moves but not affected by forces
            self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            self.body.position = (x, y)
            self.shape = pymunk.Poly.create_box(self.body, (width, height))
            self.shape.collision_type = COLLISION_HAZARD
            self.shape.sensor = True
            physics.space.add(self.body, self.shape)
        else:
            # Static hazard
            self.body = None
            self.shape = physics.create_static_box(
                x, y, width, height,
                collision_type=COLLISION_HAZARD,
                friction=0.0,
            )
            self.shape.sensor = True

    @property
    def x(self) -> float:
        """Current x position."""
        if self.body:
            return self.body.position.x
        return self._x

    @property
    def y(self) -> float:
        """Current y position."""
        if self.body:
            return self.body.position.y
        return self._y

    def update(self, dt: float) -> None:
        """Update hazard position for moving hazards.

        Args:
            dt: Time step in seconds.
        """
        if self.speed <= 0 or not self.body:
            return

        # Move in current direction
        new_x = self.body.position.x + self._direction * self.speed * dt

        # Check patrol bounds
        half_patrol = self.patrol_distance / 2
        min_x = self._start_x - half_patrol
        max_x = self._start_x + half_patrol

        if new_x >= max_x:
            new_x = max_x
            self._direction = -1.0
        elif new_x <= min_x:
            new_x = min_x
            self._direction = 1.0

        self.body.position = (new_x, self.body.position.y)

    @property
    def triggered(self) -> bool:
        """Check if this hazard was touched by player."""
        return self.physics.is_hazard_triggered(self.shape)

    @triggered.setter
    def triggered(self, value: bool) -> None:
        """For reset - clear trigger state."""
        if not value:
            # Can't directly clear one hazard, but engine clears all on reset
            pass

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get (left, bottom, right, top) bounds."""
        half_w = self.width / 2
        half_h = self.height / 2
        curr_x = self.x
        curr_y = self.y
        return (
            curr_x - half_w,
            curr_y - half_h,
            curr_x + half_w,
            curr_y + half_h,
        )


class TimedHazard:
    """Hazard that toggles between active and inactive states on a timer.

    When active, kills player on contact. When inactive, harmless.
    """

    def __init__(
        self,
        physics: PhysicsWorld,
        x: float,
        y: float,
        width: float = 30.0,
        height: float = 30.0,
        active_duration: float = 2.0,
        inactive_duration: float = 1.0,
        start_active: bool = True,
    ):
        """Create timed hazard.

        Args:
            physics: The physics world
            x, y: Center position
            width, height: Hazard dimensions
            active_duration: Seconds hazard is active (deadly)
            inactive_duration: Seconds hazard is inactive (safe)
            start_active: Whether hazard starts in active state
        """
        self.physics = physics
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.active_duration = active_duration
        self.inactive_duration = inactive_duration

        self._active = start_active
        self._timer = 0.0

        # Create sensor shape (always exists, but collision only checked when active)
        self.shape = physics.create_static_box(
            x, y, width, height,
            collision_type=COLLISION_HAZARD,
            friction=0.0,
        )
        self.shape.sensor = True

    @property
    def active(self) -> bool:
        """Whether hazard is currently deadly."""
        return self._active

    def update(self, dt: float) -> None:
        """Update hazard timer and toggle state.

        Args:
            dt: Time step in seconds.
        """
        self._timer += dt

        if self._active:
            if self._timer >= self.active_duration:
                self._active = False
                self._timer = 0.0
        else:
            if self._timer >= self.inactive_duration:
                self._active = True
                self._timer = 0.0

    @property
    def triggered(self) -> bool:
        """Check if this hazard was touched by player while active."""
        if not self._active:
            return False
        return self.physics.is_hazard_triggered(self.shape)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get (left, bottom, right, top) bounds."""
        half_w = self.width / 2
        half_h = self.height / 2
        return (
            self.x - half_w,
            self.y - half_h,
            self.x + half_w,
            self.y + half_h,
        )


class FlashingZone:
    """Area that cycles between safe and deadly states.

    Similar to TimedHazard but intended for larger floor/ceiling zones
    that require timing to traverse.
    """

    def __init__(
        self,
        physics: PhysicsWorld,
        x: float,
        y: float,
        width: float = 100.0,
        height: float = 20.0,
        safe_duration: float = 2.0,
        deadly_duration: float = 1.0,
        start_safe: bool = True,
    ):
        """Create flashing zone.

        Args:
            physics: The physics world
            x, y: Center position
            width, height: Zone dimensions
            safe_duration: Seconds zone is safe
            deadly_duration: Seconds zone is deadly
            start_safe: Whether zone starts in safe state
        """
        self.physics = physics
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.safe_duration = safe_duration
        self.deadly_duration = deadly_duration

        self._safe = start_safe
        self._timer = 0.0

        # Create sensor shape
        self.shape = physics.create_static_box(
            x, y, width, height,
            collision_type=COLLISION_HAZARD,
            friction=0.0,
        )
        self.shape.sensor = True

    @property
    def safe(self) -> bool:
        """Whether zone is currently safe to touch."""
        return self._safe

    @property
    def deadly(self) -> bool:
        """Whether zone is currently deadly."""
        return not self._safe

    def update(self, dt: float) -> None:
        """Update zone timer and toggle state.

        Args:
            dt: Time step in seconds.
        """
        self._timer += dt

        if self._safe:
            if self._timer >= self.safe_duration:
                self._safe = False
                self._timer = 0.0
        else:
            if self._timer >= self.deadly_duration:
                self._safe = True
                self._timer = 0.0

    @property
    def triggered(self) -> bool:
        """Check if this zone was touched by player while deadly."""
        if self._safe:
            return False
        return self.physics.is_hazard_triggered(self.shape)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get (left, bottom, right, top) bounds."""
        half_w = self.width / 2
        half_h = self.height / 2
        return (
            self.x - half_w,
            self.y - half_h,
            self.x + half_w,
            self.y + half_h,
        )


class Collectible:
    """Pickup item that disappears on contact and increments score.

    Used for optional objectives that reward exploration.
    Uses a shape-to-instance registry so a single collision handler
    can identify which collectible was touched.
    """

    # Class-level registry: shape -> Collectible instance
    _registry: dict = {}

    def __init__(
        self,
        physics: PhysicsWorld,
        x: float,
        y: float,
        width: float = 20.0,
        height: float = 20.0,
        value: int = 1,
    ):
        """Create collectible.

        Args:
            physics: The physics world
            x, y: Center position
            width, height: Collectible dimensions
            value: Score value when collected
        """
        self.physics = physics
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.value = value
        self._collected = False

        # Create sensor shape
        self.shape = physics.create_static_box(
            x, y, width, height,
            collision_type=COLLISION_COLLECTIBLE,
            friction=0.0,
        )
        self.shape.sensor = True

        # Register this instance and install shared handler
        Collectible._registry[self.shape] = self
        physics.space.on_collision(
            collision_type_a=COLLISION_PLAYER,
            collision_type_b=COLLISION_COLLECTIBLE,
            begin=Collectible._on_player_contact,
        )

    @staticmethod
    def _on_player_contact(
        arbiter: pymunk.Arbiter, _space: pymunk.Space, _data
    ) -> None:
        """Called when player touches any collectible."""
        collectible_shape = arbiter.shapes[1]
        collectible = Collectible._registry.get(collectible_shape)
        if collectible and not collectible._collected:
            collectible._collected = True

    @property
    def collected(self) -> bool:
        """Whether this collectible has been picked up."""
        return self._collected

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get (left, bottom, right, top) bounds."""
        half_w = self.width / 2
        half_h = self.height / 2
        return (
            self.x - half_w,
            self.y - half_h,
            self.x + half_w,
            self.y + half_h,
        )


class Spring:
    """Bouncer that launches player upward on contact.

    Applies an upward impulse when player touches it.
    """

    def __init__(
        self,
        physics: PhysicsWorld,
        x: float,
        y: float,
        width: float = 40.0,
        height: float = 20.0,
        launch_impulse: float = 1200.0,
    ):
        """Create spring.

        Args:
            physics: The physics world
            x, y: Center position
            width, height: Spring dimensions
            launch_impulse: Upward impulse applied to player on contact (default ~2x regular jump)
        """
        self.physics = physics
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.launch_impulse = launch_impulse
        self._triggered_this_frame = False

        # Create sensor shape
        self.shape = physics.create_static_box(
            x, y, width, height,
            collision_type=COLLISION_SPRING,
            friction=0.0,
        )
        self.shape.sensor = True

        # Set up spring collision handler
        physics.space.on_collision(
            collision_type_a=COLLISION_PLAYER,
            collision_type_b=COLLISION_SPRING,
            begin=self._on_player_contact,
        )

    def _on_player_contact(
        self, arbiter: pymunk.Arbiter, _space: pymunk.Space, _data
    ) -> None:
        """Called when player touches spring."""
        player_shape = arbiter.shapes[0]
        player_body = player_shape.body

        # Apply upward impulse
        player_body.velocity = (player_body.velocity.x, 0)  # Reset vertical velocity
        player_body.apply_impulse_at_local_point((0, self.launch_impulse), (0, 0))
        self._triggered_this_frame = True

    @property
    def triggered(self) -> bool:
        """Check if spring was activated this frame."""
        return self._triggered_this_frame

    def reset_trigger(self) -> None:
        """Reset trigger state (call at start of each frame)."""
        self._triggered_this_frame = False

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get (left, bottom, right, top) bounds."""
        half_w = self.width / 2
        half_h = self.height / 2
        return (
            self.x - half_w,
            self.y - half_h,
            self.x + half_w,
            self.y + half_h,
        )
