"""Configuration system for parameterizable platformer.

PhysicsConfig defines player attributes that parameterize the equations of motion.
Under parabolic dynamics, these attributes map directly to behavioral outcomes.
Under non-standard dynamics, actual behavior is measured by the calibration system.

This design separates:
- Player attributes (what we configure) - defined in PhysicsConfig
- Dynamics models (how motion equations work) - defined in dynamics.py
- Actual behavior (what the player experiences) - measured by calibration
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, ClassVar
import random


@dataclass
class PhysicsConfig:
    """Player attributes that parameterize the equations of motion.

    These attributes are used as base values in the dynamics equations. Under
    parabolic dynamics, they map directly to behavioral outcomes. Under non-standard
    dynamics (cubic, floaty, asymmetric, etc.), the dynamics model modulates these
    base values, and actual behavioral outcomes are measured by the calibration system.

    This design enables:
    - Systematic variation across dynamics models
    - Calibration to measure actual behavior
    - Separation of configuration from behavioral consequences
    """

    # === PLAYER ATTRIBUTES (what we configure) ===

    # Jump power attribute
    jump_height: float = 120.0  # Player jump power attribute (px). Equals actual apex height under parabolic dynamics; actual apex under other dynamics types is measured by calibration.
    jump_duration: float = 0.4  # Player jump timing attribute (s). Equals actual time-to-apex under parabolic dynamics; actual timing depends on dynamics model.

    # Movement attributes
    move_speed: float = 250.0  # Player movement power attribute (px/s). Maximum speed cap for force model; actual max speed depends on horizontal dynamics model and friction.
    accel_time: float = 0.15  # Player acceleration attribute (s). Time to reach max speed under force model.

    # Control attributes
    air_control: float = 0.5  # Air steering fraction (0-1). Fraction of ground control authority while airborne.

    # Friction attribute
    ground_friction: float = 0.3  # Ground friction attribute (0-1). Affects stopping behavior and maximum achievable speed.

    # === DERIVED PHYSICS VALUES (computed for simulation) ===

    @property
    def gravity(self) -> float:
        """Base gravity value derived from jump attributes. Using h = ½gt² at apex.

        Non-standard dynamics models use this as a base value and modulate it
        according to their equations (e.g., cubic scales by airtime, floaty uses tanh).
        """
        # At apex: v=0, started with v0, took jump_duration to get there
        # h = v0*t - ½gt²  and  v0 = g*t  →  h = ½gt²  →  g = 2h/t²
        return -2 * self.jump_height / (self.jump_duration ** 2)

    @property
    def jump_impulse(self) -> float:
        """Base jump impulse derived from jump attributes. v0 = g*t at apex.

        Used as the initial velocity boost when jumping. Actual jump behavior
        depends on the vertical dynamics model.
        """
        return -self.gravity * self.jump_duration

    @property
    def move_accel(self) -> float:
        """Base acceleration derived from movement attributes.

        Used by force-based horizontal models. Other horizontal models
        (velocity, impulse, drag) use move_speed differently.
        """
        return self.move_speed / self.accel_time

    # === SAMPLING RANGES (behavioral, interpretable) ===

    # Jump height: 60px (short hop) to 200px (huge leap)
    JUMP_HEIGHT_RANGE: ClassVar[Tuple[float, float]] = (60.0, 200.0)

    # Jump duration: 0.25s (snappy) to 0.6s (floaty)
    JUMP_DURATION_RANGE: ClassVar[Tuple[float, float]] = (0.25, 0.6)

    # Move speed: 150px/s (slow) to 400px/s (fast)
    MOVE_SPEED_RANGE: ClassVar[Tuple[float, float]] = (150.0, 400.0)

    # Accel time: 0.05s (instant) to 0.3s (sluggish)
    ACCEL_TIME_RANGE: ClassVar[Tuple[float, float]] = (0.05, 0.3)

    # Air control: 0.1 (committed jumps) to 0.9 (full air steering)
    AIR_CONTROL_RANGE: ClassVar[Tuple[float, float]] = (0.1, 0.9)

    # Ground friction: 0.0 (ice) to 0.8 (very sticky)
    GROUND_FRICTION_RANGE: ClassVar[Tuple[float, float]] = (0.0, 0.8)

    @classmethod
    def sample_jump_only(cls) -> "PhysicsConfig":
        """Sample with only jump parameters varied (Phase 1 probe)."""
        return cls(
            jump_height=random.uniform(*cls.JUMP_HEIGHT_RANGE),
            jump_duration=random.uniform(*cls.JUMP_DURATION_RANGE),
        )

    @classmethod
    def sample_full(cls) -> "PhysicsConfig":
        """Sample all physics parameters."""
        return cls(
            jump_height=random.uniform(*cls.JUMP_HEIGHT_RANGE),
            jump_duration=random.uniform(*cls.JUMP_DURATION_RANGE),
            move_speed=random.uniform(*cls.MOVE_SPEED_RANGE),
            accel_time=random.uniform(*cls.ACCEL_TIME_RANGE),
            air_control=random.uniform(*cls.AIR_CONTROL_RANGE),
            ground_friction=random.uniform(*cls.GROUND_FRICTION_RANGE),
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary (behavioral params only)."""
        return {
            "jump_height": self.jump_height,
            "jump_duration": self.jump_duration,
            "move_speed": self.move_speed,
            "accel_time": self.accel_time,
            "air_control": self.air_control,
            "ground_friction": self.ground_friction,
        }

    def to_dict_with_derived(self) -> Dict[str, float]:
        """Convert to dictionary including derived physics values."""
        return {
            **self.to_dict(),
            "gravity": self.gravity,
            "jump_impulse": self.jump_impulse,
            "move_accel": self.move_accel,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "PhysicsConfig":
        """Create from dictionary (ignores derived values)."""
        return cls(
            jump_height=d.get("jump_height", 120.0),
            jump_duration=d.get("jump_duration", 0.4),
            move_speed=d.get("move_speed", 250.0),
            accel_time=d.get("accel_time", 0.15),
            air_control=d.get("air_control", 0.5),
            ground_friction=d.get("ground_friction", 0.3),
        )


@dataclass
class LayoutConfig:
    """Level layout parameters.

    To be implemented in Phase 1 expansion.
    """
    platform_density: float = 0.5  # How many platforms (0-1)
    gap_size_mean: float = 60.0  # Average gap between platforms (must be achievable by physics)
    height_variance: float = 0.5  # Vertical spread (0-1)
    level_length: float = 2000.0  # Horizontal extent (wider than screen for scrolling)
    difficulty_sigma: float = 0.3  # Spread of platform placement around behavioral reference (0-1)

    # Class-level constants for sampling ranges
    PLATFORM_DENSITY_RANGE: ClassVar[Tuple[float, float]] = (0.2, 0.8)
    GAP_SIZE_RANGE: ClassVar[Tuple[float, float]] = (50.0, 200.0)
    HEIGHT_VARIANCE_RANGE: ClassVar[Tuple[float, float]] = (0.1, 0.9)
    LEVEL_LENGTH_RANGE: ClassVar[Tuple[float, float]] = (600.0, 1500.0)
    DIFFICULTY_SIGMA_RANGE: ClassVar[Tuple[float, float]] = (0.1, 0.5)

    @classmethod
    def sample(cls) -> "LayoutConfig":
        """Sample random layout config."""
        return cls(
            platform_density=random.uniform(*cls.PLATFORM_DENSITY_RANGE),
            gap_size_mean=random.uniform(*cls.GAP_SIZE_RANGE),
            height_variance=random.uniform(*cls.HEIGHT_VARIANCE_RANGE),
            level_length=random.uniform(*cls.LEVEL_LENGTH_RANGE),
            difficulty_sigma=random.uniform(*cls.DIFFICULTY_SIGMA_RANGE),
        )


@dataclass
class DynamicsConfig:
    """Dynamic elements parameters (hazards, enemies, springs, dynamics model).

    To be implemented in later Phase 1 iterations.
    """
    hazard_density: float = 0.0  # How many hazards (0-1)
    hazard_speed: float = 0.0  # Moving hazard speed
    enemy_count: int = 0
    enemy_aggression: float = 0.0
    spring_multiplier: float = 2.0  # Spring force as multiplier of normal jump (1.5 = 50% higher)
    timed_hazard_ratio: float = 0.0  # Fraction of hazards that are timed (0-1)
    flashing_zone_count: int = 0  # Number of flashing floor zones
    spring_density: float = 0.0  # How many springs (0-1)

    # Dynamics model type (equations of motion)
    vertical_model: str = "parabolic"  # parabolic, cubic, floaty, asymmetric
    horizontal_model: str = "force"  # force, velocity, impulse, drag_limited

    # Class-level constants for sampling ranges
    HAZARD_DENSITY_RANGE: ClassVar[Tuple[float, float]] = (0.0, 0.5)
    HAZARD_SPEED_RANGE: ClassVar[Tuple[float, float]] = (0.0, 200.0)
    ENEMY_COUNT_RANGE: ClassVar[Tuple[int, int]] = (0, 10)
    ENEMY_AGGRESSION_RANGE: ClassVar[Tuple[float, float]] = (0.0, 1.0)
    SPRING_MULTIPLIER_RANGE: ClassVar[Tuple[float, float]] = (1.5, 3.0)  # 1.5x to 3x normal jump
    TIMED_HAZARD_RATIO_RANGE: ClassVar[Tuple[float, float]] = (0.0, 0.6)
    FLASHING_ZONE_COUNT_RANGE: ClassVar[Tuple[int, int]] = (0, 3)
    SPRING_DENSITY_RANGE: ClassVar[Tuple[float, float]] = (0.0, 0.4)

    VERTICAL_MODELS: ClassVar[list] = ["parabolic", "cubic", "floaty", "asymmetric"]
    HORIZONTAL_MODELS: ClassVar[list] = ["force", "velocity", "impulse", "drag_limited"]

    # Non-zero minimums for --ensure-features mode
    HAZARD_DENSITY_MIN: ClassVar[float] = 0.15
    SPRING_DENSITY_MIN: ClassVar[float] = 0.1
    TIMED_HAZARD_RATIO_MIN: ClassVar[float] = 0.2
    FLASHING_ZONE_COUNT_MIN: ClassVar[int] = 1

    @classmethod
    def sample(cls, ensure_features: bool = False) -> "DynamicsConfig":
        """Sample random dynamics config.

        Args:
            ensure_features: If True, use non-zero minimums for hazards, springs, etc.
        """
        if ensure_features:
            hazard_density = random.uniform(cls.HAZARD_DENSITY_MIN, cls.HAZARD_DENSITY_RANGE[1])
            spring_density = random.uniform(cls.SPRING_DENSITY_MIN, cls.SPRING_DENSITY_RANGE[1])
            timed_hazard_ratio = random.uniform(cls.TIMED_HAZARD_RATIO_MIN, cls.TIMED_HAZARD_RATIO_RANGE[1])
            flashing_zone_count = random.randint(cls.FLASHING_ZONE_COUNT_MIN, cls.FLASHING_ZONE_COUNT_RANGE[1])
        else:
            hazard_density = random.uniform(*cls.HAZARD_DENSITY_RANGE)
            spring_density = random.uniform(*cls.SPRING_DENSITY_RANGE)
            timed_hazard_ratio = random.uniform(*cls.TIMED_HAZARD_RATIO_RANGE)
            flashing_zone_count = random.randint(*cls.FLASHING_ZONE_COUNT_RANGE)

        return cls(
            hazard_density=hazard_density,
            hazard_speed=random.uniform(*cls.HAZARD_SPEED_RANGE),
            enemy_count=random.randint(*cls.ENEMY_COUNT_RANGE),
            enemy_aggression=random.uniform(*cls.ENEMY_AGGRESSION_RANGE),
            spring_multiplier=random.uniform(*cls.SPRING_MULTIPLIER_RANGE),
            timed_hazard_ratio=timed_hazard_ratio,
            flashing_zone_count=flashing_zone_count,
            spring_density=spring_density,
            vertical_model=random.choice(cls.VERTICAL_MODELS),
            horizontal_model=random.choice(cls.HORIZONTAL_MODELS),
        )


@dataclass
class ObjectiveConfig:
    """Objective/goal parameters.

    To be implemented in later Phase 1 iterations.
    """
    goal_distance: float = 1.0  # Fraction of level to goal
    collectibles: int = 0  # Platform-based collectibles
    airspace_collectibles: int = 3  # Collectibles in open airspace (trajectory-shape-relevant)
    time_pressure: float = 0.0  # 0 = no limit, 1 = strict limit

    # Class-level constants for sampling ranges
    GOAL_DISTANCE_RANGE: ClassVar[Tuple[float, float]] = (0.3, 1.0)
    COLLECTIBLES_RANGE: ClassVar[Tuple[int, int]] = (0, 10)
    AIRSPACE_COLLECTIBLES_RANGE: ClassVar[Tuple[int, int]] = (0, 10)
    TIME_PRESSURE_RANGE: ClassVar[Tuple[float, float]] = (0.0, 1.0)

    # Non-zero minimum for --ensure-features mode
    COLLECTIBLES_MIN: ClassVar[int] = 3
    AIRSPACE_COLLECTIBLES_MIN: ClassVar[int] = 2

    @classmethod
    def sample(cls, ensure_features: bool = False) -> "ObjectiveConfig":
        """Sample random objective config.

        Args:
            ensure_features: If True, use non-zero minimums for collectibles.
        """
        if ensure_features:
            collectibles = random.randint(cls.COLLECTIBLES_MIN, cls.COLLECTIBLES_RANGE[1])
            airspace = random.randint(cls.AIRSPACE_COLLECTIBLES_MIN, cls.AIRSPACE_COLLECTIBLES_RANGE[1])
        else:
            collectibles = random.randint(*cls.COLLECTIBLES_RANGE)
            airspace = random.randint(*cls.AIRSPACE_COLLECTIBLES_RANGE)

        return cls(
            goal_distance=random.uniform(*cls.GOAL_DISTANCE_RANGE),
            collectibles=collectibles,
            airspace_collectibles=airspace,
            time_pressure=random.uniform(*cls.TIME_PRESSURE_RANGE),
        )


@dataclass
class GameConfig:
    """Complete game configuration combining all parameter groups."""
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    objectives: ObjectiveConfig = field(default_factory=ObjectiveConfig)

    # Display settings (not parameterized for world synthesis)
    screen_width: int = 800
    screen_height: int = 600
    fps: int = 60

    @classmethod
    def sample_jump_only(cls) -> "GameConfig":
        """Sample config with only jump parameters varied (Phase 1 probe)."""
        return cls(physics=PhysicsConfig.sample_jump_only())

    @classmethod
    def sample_physics_only(cls) -> "GameConfig":
        """Sample config with full physics variation."""
        return cls(physics=PhysicsConfig.sample_full())

    @classmethod
    def sample_full(cls, ensure_features: bool = False) -> "GameConfig":
        """Sample complete random configuration.

        Args:
            ensure_features: If True, use non-zero minimums for hazards, springs, collectibles, etc.
        """
        return cls(
            physics=PhysicsConfig.sample_full(),
            layout=LayoutConfig.sample(),
            dynamics=DynamicsConfig.sample(ensure_features=ensure_features),
            objectives=ObjectiveConfig.sample(ensure_features=ensure_features),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested dictionary."""
        return {
            "physics": self.physics.to_dict(),
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "fps": self.fps,
        }


# Predefined configurations for testing/demo
# All defined in behavioral terms - note how interpretable these are!
CONFIGS = {
    # Default balanced feel
    "default": GameConfig(),

    # High jump, floaty - easy platforming
    "floaty": GameConfig(physics=PhysicsConfig(
        jump_height=180.0,
        jump_duration=0.55,
        air_control=0.7,
    )),

    # Low jump, snappy - precision platforming
    "tight": GameConfig(physics=PhysicsConfig(
        jump_height=80.0,
        jump_duration=0.3,
        air_control=0.3,
    )),

    # Moon-like - very high, very slow
    "moon": GameConfig(physics=PhysicsConfig(
        jump_height=200.0,
        jump_duration=0.8,
        air_control=0.8,
    )),

    # Heavy - low jump, fast fall
    "heavy": GameConfig(physics=PhysicsConfig(
        jump_height=70.0,
        jump_duration=0.25,
        air_control=0.2,
    )),

    # --- Dynamics-aware presets (combine vertical/horizontal models with physics) ---

    # Bouncy: cubic trajectory (hang at apex, drop hard) + high jump
    "bouncy": GameConfig(
        physics=PhysicsConfig(
            jump_height=170.0, jump_duration=0.45,
            move_speed=280.0, air_control=0.6, ground_friction=0.3,
        ),
        dynamics=DynamicsConfig(vertical_model="cubic", horizontal_model="force"),
    ),

    # Ice: drag-limited movement (organic deceleration) + zero friction
    "ice": GameConfig(
        physics=PhysicsConfig(
            jump_height=110.0, jump_duration=0.38,
            move_speed=350.0, air_control=0.7, ground_friction=0.0,
        ),
        dynamics=DynamicsConfig(vertical_model="parabolic", horizontal_model="drag_limited"),
    ),

    # Hover: floaty gravity + direct velocity (instant direction changes)
    "hover": GameConfig(
        physics=PhysicsConfig(
            jump_height=160.0, jump_duration=0.6,
            move_speed=300.0, air_control=0.9, ground_friction=0.2,
        ),
        dynamics=DynamicsConfig(vertical_model="floaty", horizontal_model="velocity"),
    ),

    # Twitch: asymmetric (slow rise, fast fall) + impulse (twitchy)
    "twitch": GameConfig(
        physics=PhysicsConfig(
            jump_height=100.0, jump_duration=0.3,
            move_speed=320.0, accel_time=0.05, air_control=0.8, ground_friction=0.4,
        ),
        dynamics=DynamicsConfig(vertical_model="asymmetric", horizontal_model="impulse"),
    ),

    # Tank: committed movement, heavy feel, no air steering
    "tank": GameConfig(
        physics=PhysicsConfig(
            jump_height=90.0, jump_duration=0.28,
            move_speed=180.0, accel_time=0.25, air_control=0.1, ground_friction=0.6,
        ),
        dynamics=DynamicsConfig(vertical_model="parabolic", horizontal_model="force"),
    ),

    # Astronaut: floaty everything, impulse movement, space-like
    "astronaut": GameConfig(
        physics=PhysicsConfig(
            jump_height=200.0, jump_duration=0.7,
            move_speed=200.0, air_control=0.9, ground_friction=0.05,
        ),
        dynamics=DynamicsConfig(vertical_model="floaty", horizontal_model="impulse"),
    ),
}
