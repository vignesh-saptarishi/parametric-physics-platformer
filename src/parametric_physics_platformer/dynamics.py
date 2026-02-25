"""Dynamics models that define the equations of motion.

The standard platformer uses constant gravity (parabolic trajectories)
and force-based horizontal movement. This module provides an abstraction
that allows swapping the actual equations, not just parameters.

Two independent axes:
- Vertical: trajectory shape (parabolic, cubic, floaty, asymmetric)
- Horizontal: movement model (force, velocity, impulse, drag)
"""

import math
from enum import Enum, auto
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
from .config import PhysicsConfig


class VerticalModel(Enum):
    """Vertical trajectory type - changes jump arc shape."""
    PARABOLIC = auto()   # F = -mg (constant) -> y ~ t^2
    CUBIC = auto()       # F = -m*alpha*t (linear increase) -> y ~ t^3
    FLOATY = auto()      # F = -mg0*tanh(kt) -> float then snap
    ASYMMETRIC = auto()  # F_rise != F_fall -> piecewise parabolic


class HorizontalModel(Enum):
    """Horizontal movement type - changes how input maps to motion."""
    FORCE = auto()         # input -> acceleration (current)
    VELOCITY = auto()      # input -> direct velocity
    IMPULSE = auto()       # input -> per-frame delta-v
    DRAG_LIMITED = auto()  # input -> force + v^2 drag


@dataclass
class VerticalParams:
    """Raw equation coefficients for vertical trajectory models.

    These are raw equation coefficients that modulate the base gravity derived
    from PhysicsConfig. The behavioral consequences of these parameters are
    measured by the calibration system.
    """
    # Shared
    base_gravity: float = 0.0  # Derived from PhysicsConfig

    # Cubic-specific
    cubic_alpha: float = 3.0  # Rate of gravity increase with airtime

    # Floaty-specific
    floaty_k: float = 4.0  # Tanh steepness (higher = faster snap)

    # Asymmetric-specific
    rise_multiplier: float = 0.5  # Gravity multiplier during rise (< 1 = floatier rise)
    fall_multiplier: float = 2.0  # Gravity multiplier during fall (> 1 = faster fall)


@dataclass
class HorizontalParams:
    """Raw equation coefficients for horizontal movement models.

    These are raw equation coefficients. The behavioral consequences of these
    parameters are measured by the calibration system.
    """
    # Velocity-specific
    velocity_scale: float = 1.0  # Multiplier on move_speed for direct velocity

    # Impulse-specific
    impulse_strength: float = 50.0  # Velocity change per frame per unit input

    # Drag-specific
    drag_coefficient: float = 0.005  # v^2 drag coefficient


class DynamicsModel:
    """Base dynamics model combining vertical and horizontal components.

    Subclasses override get_gravity() and get_horizontal_force() to
    implement different equations of motion.
    """

    def __init__(
        self,
        vertical: VerticalModel = VerticalModel.PARABOLIC,
        horizontal: HorizontalModel = HorizontalModel.FORCE,
        physics_config: Optional[PhysicsConfig] = None,
        vertical_params: Optional[VerticalParams] = None,
        horizontal_params: Optional[HorizontalParams] = None,
    ):
        self.vertical = vertical
        self.horizontal = horizontal
        self.physics_config = physics_config or PhysicsConfig()
        self.vertical_params = vertical_params or VerticalParams()
        self.horizontal_params = horizontal_params or HorizontalParams()

        # Set base gravity from physics config
        self.vertical_params.base_gravity = self.physics_config.gravity

    def get_gravity(self, airtime: float) -> Tuple[float, float]:
        """Return gravity vector for current airtime.

        Args:
            airtime: Seconds since player left ground (0 if grounded).

        Returns:
            (gx, gy) gravity vector in pymunk coordinates.
        """
        raise NotImplementedError

    def get_horizontal_force(
        self,
        direction: float,
        vx: float,
        is_grounded: bool,
    ) -> float:
        """Return horizontal force to apply.

        Args:
            direction: Input direction [-1, 1].
            vx: Current horizontal velocity.
            is_grounded: Whether player is on ground.

        Returns:
            Horizontal force value.
        """
        raise NotImplementedError

    def get_damping(self, vx: float, is_grounded: bool) -> float:
        """Return velocity damping factor for this frame.

        Args:
            vx: Current horizontal velocity.
            is_grounded: Whether player is on ground.

        Returns:
            Multiplier for vx (1.0 = no damping, 0.0 = full stop).
        """
        if is_grounded:
            return 1.0 - 0.05 * self.physics_config.ground_friction
        return 1.0

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata dict for data collection."""
        return {
            "vertical": self.vertical.name.lower(),
            "horizontal": self.horizontal.name.lower(),
            "vertical_params": {
                "base_gravity": self.vertical_params.base_gravity,
                "cubic_alpha": self.vertical_params.cubic_alpha,
                "floaty_k": self.vertical_params.floaty_k,
                "rise_multiplier": self.vertical_params.rise_multiplier,
                "fall_multiplier": self.vertical_params.fall_multiplier,
            },
            "horizontal_params": {
                "velocity_scale": self.horizontal_params.velocity_scale,
                "impulse_strength": self.horizontal_params.impulse_strength,
                "drag_coefficient": self.horizontal_params.drag_coefficient,
            },
        }

    def get_type_id(self) -> int:
        """Return integer encoding of (vertical, horizontal) combination.

        Encodes as vertical_index * 4 + horizontal_index (0-15).
        """
        v_idx = list(VerticalModel).index(self.vertical)
        h_idx = list(HorizontalModel).index(self.horizontal)
        return v_idx * len(HorizontalModel) + h_idx


class StandardDynamics(DynamicsModel):
    """Standard dynamics: constant gravity + force-based movement.

    Reproduces the current platformer behavior exactly.
    """

    def __init__(self, physics_config: Optional[PhysicsConfig] = None, **kwargs):
        super().__init__(
            vertical=VerticalModel.PARABOLIC,
            horizontal=HorizontalModel.FORCE,
            physics_config=physics_config,
            **kwargs,
        )

    def get_gravity(self, airtime: float) -> Tuple[float, float]:
        return (0, self.vertical_params.base_gravity)

    def get_horizontal_force(
        self,
        direction: float,
        vx: float,
        is_grounded: bool,
    ) -> float:
        max_v = self.physics_config.move_speed
        control = 1.0 if is_grounded else self.physics_config.air_control
        force = direction * self.physics_config.move_accel * control

        # Only apply if not at max speed in that direction
        if direction > 0 and vx >= max_v:
            return 0.0
        if direction < 0 and vx <= -max_v:
            return 0.0
        return force


def _standard_horizontal_force(model: DynamicsModel, direction, vx, is_grounded):
    """Shared force-based horizontal movement (reused by vertical-only variants)."""
    max_v = model.physics_config.move_speed
    control = 1.0 if is_grounded else model.physics_config.air_control
    force = direction * model.physics_config.move_accel * control
    if direction > 0 and vx >= max_v:
        return 0.0
    if direction < 0 and vx <= -max_v:
        return 0.0
    return force


class CubicDynamics(DynamicsModel):
    """Cubic trajectory: gravity increases linearly with airtime.

    F(t) = -m * alpha * t
    Creates a trajectory where player hangs at apex then drops hard.
    The cubic_alpha parameter is calibrated so that at t=jump_duration,
    gravity roughly matches standard gravity magnitude.
    """

    def __init__(self, physics_config=None, **kwargs):
        super().__init__(
            vertical=VerticalModel.CUBIC,
            horizontal=HorizontalModel.FORCE,
            physics_config=physics_config,
            **kwargs,
        )

    def get_gravity(self, airtime: float) -> Tuple[float, float]:
        alpha = self.vertical_params.cubic_alpha
        base = self.vertical_params.base_gravity
        # Scale alpha so that at t=jump_duration, force ~ base_gravity
        scaled_alpha = abs(base) * alpha / max(self.physics_config.jump_duration, 0.1)
        gy = -scaled_alpha * airtime
        return (0, gy)

    def get_horizontal_force(self, direction, vx, is_grounded):
        return _standard_horizontal_force(self, direction, vx, is_grounded)


class FloatyDynamics(DynamicsModel):
    """Floaty-snap trajectory: gravity follows tanh curve.

    F(t) = -mg0 * tanh(k * t)
    Player floats initially, then gravity snaps in.
    Higher k = faster snap to full gravity.
    """

    def __init__(self, physics_config=None, **kwargs):
        super().__init__(
            vertical=VerticalModel.FLOATY,
            horizontal=HorizontalModel.FORCE,
            physics_config=physics_config,
            **kwargs,
        )

    def get_gravity(self, airtime: float) -> Tuple[float, float]:
        base = self.vertical_params.base_gravity
        k = self.vertical_params.floaty_k
        gy = base * math.tanh(k * airtime)
        return (0, gy)

    def get_horizontal_force(self, direction, vx, is_grounded):
        return _standard_horizontal_force(self, direction, vx, is_grounded)


class AsymmetricDynamics(DynamicsModel):
    """Asymmetric trajectory: different gravity during rise vs fall.

    F = -mg * rise_multiplier   when vy > 0 (rising)
    F = -mg * fall_multiplier   when vy <= 0 (falling)
    Creates piecewise parabolic arcs with different curvature up vs down.
    """

    def __init__(self, physics_config=None, **kwargs):
        super().__init__(
            vertical=VerticalModel.ASYMMETRIC,
            horizontal=HorizontalModel.FORCE,
            physics_config=physics_config,
            **kwargs,
        )

    def get_gravity(self, airtime: float) -> Tuple[float, float]:
        # Default: use fall multiplier (safe default when vy unknown)
        base = self.vertical_params.base_gravity
        return (0, base * self.vertical_params.fall_multiplier)

    def get_gravity_for_velocity(self, vy: float) -> Tuple[float, float]:
        """Get gravity based on vertical velocity direction."""
        base = self.vertical_params.base_gravity
        if vy > 0:
            return (0, base * self.vertical_params.rise_multiplier)
        else:
            return (0, base * self.vertical_params.fall_multiplier)

    def get_horizontal_force(self, direction, vx, is_grounded):
        return _standard_horizontal_force(self, direction, vx, is_grounded)


class VelocityDynamics(DynamicsModel):
    """Direct velocity control: input maps to target velocity.

    No momentum, no acceleration ramp. Instant response.
    """

    def __init__(self, physics_config=None, **kwargs):
        super().__init__(
            vertical=VerticalModel.PARABOLIC,
            horizontal=HorizontalModel.VELOCITY,
            physics_config=physics_config,
            **kwargs,
        )

    def get_gravity(self, airtime):
        return (0, self.vertical_params.base_gravity)

    def get_target_velocity(self, direction: float, is_grounded: bool) -> float:
        """Return target horizontal velocity."""
        control = 1.0 if is_grounded else self.physics_config.air_control
        scale = self.horizontal_params.velocity_scale
        return direction * self.physics_config.move_speed * control * scale

    def get_horizontal_force(self, direction, vx, is_grounded):
        # Stiff spring toward target velocity
        target = self.get_target_velocity(direction, is_grounded)
        return (target - vx) * 30.0


class ImpulseDynamics(DynamicsModel):
    """Impulse control: input gives per-frame velocity delta.

    Twitchy, responsive. Like force but bypasses pymunk integration.
    """

    def __init__(self, physics_config=None, **kwargs):
        super().__init__(
            vertical=VerticalModel.PARABOLIC,
            horizontal=HorizontalModel.IMPULSE,
            physics_config=physics_config,
            **kwargs,
        )

    def get_gravity(self, airtime):
        return (0, self.vertical_params.base_gravity)

    def get_velocity_impulse(self, direction: float, is_grounded: bool) -> float:
        """Return velocity change to apply this frame."""
        control = 1.0 if is_grounded else self.physics_config.air_control
        return direction * self.horizontal_params.impulse_strength * control

    def get_horizontal_force(self, direction, vx, is_grounded):
        # Impulse model applies delta-v directly, not force
        return 0.0


class DragDynamics(DynamicsModel):
    """Drag-limited movement: force-based with v^2 drag, no speed cap.

    More organic feeling — terminal velocity instead of hard cap.
    """

    def __init__(self, physics_config=None, **kwargs):
        super().__init__(
            vertical=VerticalModel.PARABOLIC,
            horizontal=HorizontalModel.DRAG_LIMITED,
            physics_config=physics_config,
            **kwargs,
        )

    def get_gravity(self, airtime):
        return (0, self.vertical_params.base_gravity)

    def get_horizontal_force(self, direction, vx, is_grounded):
        control = 1.0 if is_grounded else self.physics_config.air_control
        drive = direction * self.physics_config.move_accel * control
        drag = -self.horizontal_params.drag_coefficient * vx * abs(vx)
        return drive + drag


# ---------------------------------------------------------------------------
# Gravity dispatch table (vertical axis)
# ---------------------------------------------------------------------------

def _gravity_parabolic(model, airtime):
    return (0, model.vertical_params.base_gravity)


def _gravity_cubic(model, airtime):
    alpha = model.vertical_params.cubic_alpha
    base = model.vertical_params.base_gravity
    scaled_alpha = abs(base) * alpha / max(model.physics_config.jump_duration, 0.1)
    return (0, -scaled_alpha * airtime)


def _gravity_floaty(model, airtime):
    base = model.vertical_params.base_gravity
    k = model.vertical_params.floaty_k
    return (0, base * math.tanh(k * airtime))


def _gravity_asymmetric_fall(model, airtime):
    """Default for asymmetric when vy unknown — uses fall multiplier."""
    base = model.vertical_params.base_gravity
    return (0, base * model.vertical_params.fall_multiplier)


_GRAVITY_DISPATCH = {
    VerticalModel.PARABOLIC: _gravity_parabolic,
    VerticalModel.CUBIC: _gravity_cubic,
    VerticalModel.FLOATY: _gravity_floaty,
    VerticalModel.ASYMMETRIC: _gravity_asymmetric_fall,
}


# ---------------------------------------------------------------------------
# Horizontal dispatch table
# ---------------------------------------------------------------------------

def _horizontal_force(model, direction, vx, is_grounded):
    return _standard_horizontal_force(model, direction, vx, is_grounded)


def _horizontal_velocity(model, direction, vx, is_grounded):
    control = 1.0 if is_grounded else model.physics_config.air_control
    scale = model.horizontal_params.velocity_scale
    target = direction * model.physics_config.move_speed * control * scale
    return (target - vx) * 30.0


def _horizontal_impulse(model, direction, vx, is_grounded):
    return 0.0  # Impulse applied directly, not as force


def _horizontal_drag(model, direction, vx, is_grounded):
    control = 1.0 if is_grounded else model.physics_config.air_control
    drive = direction * model.physics_config.move_accel * control
    drag = -model.horizontal_params.drag_coefficient * vx * abs(vx)
    return drive + drag


_HORIZONTAL_DISPATCH = {
    HorizontalModel.FORCE: _horizontal_force,
    HorizontalModel.VELOCITY: _horizontal_velocity,
    HorizontalModel.IMPULSE: _horizontal_impulse,
    HorizontalModel.DRAG_LIMITED: _horizontal_drag,
}


class CompositeDynamics(DynamicsModel):
    """Composite dynamics: any vertical x any horizontal via dispatch tables."""

    def __init__(self, vertical, horizontal, physics_config=None, **kwargs):
        super().__init__(
            vertical=vertical,
            horizontal=horizontal,
            physics_config=physics_config,
            **kwargs,
        )
        self._gravity_fn = _GRAVITY_DISPATCH[vertical]
        self._horizontal_fn = _HORIZONTAL_DISPATCH[horizontal]

    def get_gravity(self, airtime):
        return self._gravity_fn(self, airtime)

    def get_gravity_for_velocity(self, vy: float) -> Tuple[float, float]:
        """Get gravity based on vertical velocity (for asymmetric)."""
        if self.vertical == VerticalModel.ASYMMETRIC:
            base = self.vertical_params.base_gravity
            if vy > 0:
                return (0, base * self.vertical_params.rise_multiplier)
            else:
                return (0, base * self.vertical_params.fall_multiplier)
        return self.get_gravity(0.0)

    def get_horizontal_force(self, direction, vx, is_grounded):
        return self._horizontal_fn(self, direction, vx, is_grounded)

    def get_target_velocity(self, direction: float, is_grounded: bool) -> float:
        """For velocity model compatibility."""
        control = 1.0 if is_grounded else self.physics_config.air_control
        scale = self.horizontal_params.velocity_scale
        return direction * self.physics_config.move_speed * control * scale

    def get_velocity_impulse(self, direction: float, is_grounded: bool) -> float:
        """For impulse model compatibility."""
        control = 1.0 if is_grounded else self.physics_config.air_control
        return direction * self.horizontal_params.impulse_strength * control


def create_dynamics(
    vertical: VerticalModel = VerticalModel.PARABOLIC,
    horizontal: HorizontalModel = HorizontalModel.FORCE,
    physics_config: Optional[PhysicsConfig] = None,
    vertical_params: Optional[VerticalParams] = None,
    horizontal_params: Optional[HorizontalParams] = None,
) -> DynamicsModel:
    """Factory: create a DynamicsModel with any vertical x horizontal combo."""
    return CompositeDynamics(
        vertical=vertical,
        horizontal=horizontal,
        physics_config=physics_config,
        vertical_params=vertical_params,
        horizontal_params=horizontal_params,
    )
