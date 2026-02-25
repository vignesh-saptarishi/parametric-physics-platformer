"""Parameter constraints and validation for playable worlds.

Ensures generated configurations produce playable levels by:
1. Validating individual parameter ranges
2. Checking cross-parameter consistency (e.g., gaps must be jumpable)
3. Providing constraint-aware sampling

Key insight: A "valid" config means the level is POSSIBLE to complete,
not necessarily easy or hard. Difficulty is a separate dimension.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import random

from .config import PhysicsConfig, LayoutConfig, DynamicsConfig, ObjectiveConfig, GameConfig


@dataclass
class ConstraintViolation:
    """Describes a constraint violation."""
    param: str
    message: str
    severity: str  # "error" = unplayable, "warning" = difficult but possible


@dataclass
class ConstraintResult:
    """Result of constraint validation."""
    valid: bool
    violations: List[ConstraintViolation]

    def __bool__(self) -> bool:
        return self.valid


class ParameterConstraints:
    """Defines and checks constraints on game parameters.

    Constraints are rules that ensure generated worlds are playable.
    They can be:
    - Hard constraints: Must be satisfied (gaps jumpable, goal reachable)
    - Soft constraints: Warnings for extreme difficulty
    """

    # Physics bounds (behavioral)
    JUMP_HEIGHT_MIN = 40.0   # Minimum useful jump
    JUMP_HEIGHT_MAX = 300.0  # Screen height limit
    JUMP_DURATION_MIN = 0.15  # Too fast = uncontrollable
    JUMP_DURATION_MAX = 1.0   # Too slow = boring
    MOVE_SPEED_MIN = 100.0    # Minimum useful speed
    MOVE_SPEED_MAX = 600.0    # Too fast = uncontrollable
    AIR_CONTROL_MIN = 0.0     # No air control (committed jumps)
    AIR_CONTROL_MAX = 1.0     # Full air control

    # Layout bounds
    PLATFORM_DENSITY_MIN = 0.1  # Almost no platforms
    PLATFORM_DENSITY_MAX = 1.0  # Very dense
    GAP_SIZE_MIN = 20.0   # Minimum gap
    GAP_SIZE_MAX = 400.0  # Maximum gap
    HEIGHT_VARIANCE_MIN = 0.0  # Flat
    HEIGHT_VARIANCE_MAX = 1.0  # Very varied

    # Cross-parameter constraints
    MIN_REACHABLE_RATIO = 0.5  # Gaps must be <= 50% of max jump distance
    MIN_HEIGHT_RATIO = 0.6     # Height changes must be <= 60% of max jump height

    @classmethod
    def validate_physics(cls, physics: PhysicsConfig) -> ConstraintResult:
        """Validate physics config."""
        violations = []

        # Jump height
        if physics.jump_height < cls.JUMP_HEIGHT_MIN:
            violations.append(ConstraintViolation(
                "jump_height",
                f"Jump height {physics.jump_height} < min {cls.JUMP_HEIGHT_MIN}",
                "error"
            ))
        if physics.jump_height > cls.JUMP_HEIGHT_MAX:
            violations.append(ConstraintViolation(
                "jump_height",
                f"Jump height {physics.jump_height} > max {cls.JUMP_HEIGHT_MAX}",
                "warning"
            ))

        # Jump duration
        if physics.jump_duration < cls.JUMP_DURATION_MIN:
            violations.append(ConstraintViolation(
                "jump_duration",
                f"Jump duration {physics.jump_duration} < min {cls.JUMP_DURATION_MIN}",
                "error"
            ))
        if physics.jump_duration > cls.JUMP_DURATION_MAX:
            violations.append(ConstraintViolation(
                "jump_duration",
                f"Jump duration {physics.jump_duration} > max {cls.JUMP_DURATION_MAX}",
                "warning"
            ))

        # Move speed
        if physics.move_speed < cls.MOVE_SPEED_MIN:
            violations.append(ConstraintViolation(
                "move_speed",
                f"Move speed {physics.move_speed} < min {cls.MOVE_SPEED_MIN}",
                "error"
            ))

        # Air control
        if not (cls.AIR_CONTROL_MIN <= physics.air_control <= cls.AIR_CONTROL_MAX):
            violations.append(ConstraintViolation(
                "air_control",
                f"Air control {physics.air_control} outside [0, 1]",
                "error"
            ))

        errors = [v for v in violations if v.severity == "error"]
        return ConstraintResult(valid=len(errors) == 0, violations=violations)

    @classmethod
    def validate_layout(cls, layout: LayoutConfig, physics: PhysicsConfig) -> ConstraintResult:
        """Validate layout config against physics capabilities."""
        violations = []

        # Compute physics capabilities
        max_jump_dist = physics.move_speed * 2 * physics.jump_duration * (1 + physics.air_control) / 2
        max_jump_height = physics.jump_height

        # Gap size must be achievable
        max_achievable_gap = max_jump_dist * cls.MIN_REACHABLE_RATIO
        if layout.gap_size_mean > max_achievable_gap:
            violations.append(ConstraintViolation(
                "gap_size_mean",
                f"Gap size {layout.gap_size_mean} > achievable {max_achievable_gap:.0f} (given physics)",
                "error"
            ))

        # Height variance must be achievable
        max_height_change = max_jump_height * (0.5 + layout.height_variance * 0.5)
        if max_height_change > max_jump_height:
            violations.append(ConstraintViolation(
                "height_variance",
                f"Height variance may produce unjumpable heights",
                "warning"
            ))

        # Density bounds
        if not (cls.PLATFORM_DENSITY_MIN <= layout.platform_density <= cls.PLATFORM_DENSITY_MAX):
            violations.append(ConstraintViolation(
                "platform_density",
                f"Platform density {layout.platform_density} outside valid range",
                "error"
            ))

        errors = [v for v in violations if v.severity == "error"]
        return ConstraintResult(valid=len(errors) == 0, violations=violations)

    @classmethod
    def validate_dynamics(cls, dynamics: DynamicsConfig) -> ConstraintResult:
        """Validate dynamics config."""
        violations = []

        if not (0 <= dynamics.hazard_density <= 1):
            violations.append(ConstraintViolation(
                "hazard_density",
                f"Hazard density {dynamics.hazard_density} outside [0, 1]",
                "error"
            ))

        if dynamics.hazard_density > 0.7:
            violations.append(ConstraintViolation(
                "hazard_density",
                f"Hazard density {dynamics.hazard_density} is very high",
                "warning"
            ))

        errors = [v for v in violations if v.severity == "error"]
        return ConstraintResult(valid=len(errors) == 0, violations=violations)

    @classmethod
    def validate_config(cls, config: GameConfig) -> ConstraintResult:
        """Validate full game config."""
        all_violations = []

        # Validate each component
        physics_result = cls.validate_physics(config.physics)
        all_violations.extend(physics_result.violations)

        layout_result = cls.validate_layout(config.layout, config.physics)
        all_violations.extend(layout_result.violations)

        dynamics_result = cls.validate_dynamics(config.dynamics)
        all_violations.extend(dynamics_result.violations)

        errors = [v for v in all_violations if v.severity == "error"]
        return ConstraintResult(valid=len(errors) == 0, violations=all_violations)


class ConstrainedSampler:
    """Samples parameters while respecting constraints.

    Strategies:
    1. Sample physics first (independent)
    2. Sample layout constrained by physics
    3. Sample dynamics constrained by layout
    4. Rejection sampling for edge cases
    """

    def __init__(self, max_attempts: int = 100):
        self.max_attempts = max_attempts

    def sample_physics(
        self,
        jump_height_range: Optional[Tuple[float, float]] = None,
        jump_duration_range: Optional[Tuple[float, float]] = None,
        move_speed_range: Optional[Tuple[float, float]] = None,
        air_control_range: Optional[Tuple[float, float]] = None,
    ) -> PhysicsConfig:
        """Sample valid physics config with optional range overrides."""
        jh_range = jump_height_range or PhysicsConfig.JUMP_HEIGHT_RANGE
        jd_range = jump_duration_range or PhysicsConfig.JUMP_DURATION_RANGE
        ms_range = move_speed_range or PhysicsConfig.MOVE_SPEED_RANGE
        ac_range = air_control_range or PhysicsConfig.AIR_CONTROL_RANGE

        # Clamp to valid bounds
        jh_range = (
            max(jh_range[0], ParameterConstraints.JUMP_HEIGHT_MIN),
            min(jh_range[1], ParameterConstraints.JUMP_HEIGHT_MAX),
        )
        jd_range = (
            max(jd_range[0], ParameterConstraints.JUMP_DURATION_MIN),
            min(jd_range[1], ParameterConstraints.JUMP_DURATION_MAX),
        )
        ms_range = (
            max(ms_range[0], ParameterConstraints.MOVE_SPEED_MIN),
            min(ms_range[1], ParameterConstraints.MOVE_SPEED_MAX),
        )
        ac_range = (
            max(ac_range[0], ParameterConstraints.AIR_CONTROL_MIN),
            min(ac_range[1], ParameterConstraints.AIR_CONTROL_MAX),
        )

        return PhysicsConfig(
            jump_height=random.uniform(*jh_range),
            jump_duration=random.uniform(*jd_range),
            move_speed=random.uniform(*ms_range),
            air_control=random.uniform(*ac_range),
        )

    def sample_layout(
        self,
        physics: PhysicsConfig,
        difficulty: float = 0.5,  # 0 = easy, 1 = hard
    ) -> LayoutConfig:
        """Sample layout config constrained by physics.

        Args:
            physics: Physics config to constrain against
            difficulty: Target difficulty level [0, 1]
        """
        # Compute physics capabilities
        max_jump_dist = physics.move_speed * 2 * physics.jump_duration * (1 + physics.air_control) / 2

        # Scale gap size by difficulty (harder = larger gaps, up to limit)
        max_gap = max_jump_dist * ParameterConstraints.MIN_REACHABLE_RATIO
        # Min gap is 20% of achievable, max is 80% (leave margin for validation)
        min_gap = max_gap * 0.2
        gap_size = min_gap + difficulty * (max_gap * 0.8 - min_gap)

        # Scale height variance by difficulty
        height_variance = 0.2 + difficulty * 0.6

        # Platform density inversely related to difficulty
        platform_density = 0.8 - difficulty * 0.5

        return LayoutConfig(
            platform_density=platform_density,
            gap_size_mean=gap_size,
            height_variance=height_variance,
            level_length=600 + difficulty * 400,
        )

    def sample_dynamics(
        self,
        difficulty: float = 0.5,
    ) -> DynamicsConfig:
        """Sample dynamics config."""
        # Hazard density scales with difficulty
        hazard_density = difficulty * 0.4  # Max 40% hazards

        return DynamicsConfig(
            hazard_density=hazard_density,
            hazard_speed=difficulty * 100,
            enemy_count=int(difficulty * 5),
            enemy_aggression=difficulty,
        )

    def sample_objectives(
        self,
        difficulty: float = 0.5,
    ) -> ObjectiveConfig:
        """Sample objectives config."""
        return ObjectiveConfig(
            goal_distance=0.5 + difficulty * 0.5,  # Harder = goal further
            collectibles=int((1 - difficulty) * 5),  # Easier = more collectibles
            time_pressure=difficulty * 0.5,
        )

    def sample_config(
        self,
        difficulty: float = 0.5,
        physics_overrides: Optional[Dict[str, Any]] = None,
    ) -> GameConfig:
        """Sample a complete valid config.

        Args:
            difficulty: Target difficulty [0, 1]
            physics_overrides: Optional overrides for physics params

        Returns:
            Valid GameConfig
        """
        # Sample physics (possibly with overrides)
        if physics_overrides:
            physics = PhysicsConfig(**{
                **PhysicsConfig().__dict__,
                **physics_overrides,
            })
        else:
            physics = self.sample_physics()

        # Sample other components constrained by physics and difficulty
        layout = self.sample_layout(physics, difficulty)
        dynamics = self.sample_dynamics(difficulty)
        objectives = self.sample_objectives(difficulty)

        config = GameConfig(
            physics=physics,
            layout=layout,
            dynamics=dynamics,
            objectives=objectives,
        )

        # Validate
        result = ParameterConstraints.validate_config(config)
        if not result.valid:
            errors = [v for v in result.violations if v.severity == "error"]
            raise ValueError(f"Generated invalid config: {errors}")

        return config

    def sample_difficulty_range(
        self,
        n_samples: int,
        min_difficulty: float = 0.0,
        max_difficulty: float = 1.0,
    ) -> List[GameConfig]:
        """Sample configs across a difficulty range.

        Useful for curriculum learning or difficulty studies.
        """
        configs = []
        for i in range(n_samples):
            difficulty = min_difficulty + (max_difficulty - min_difficulty) * i / (n_samples - 1)
            configs.append(self.sample_config(difficulty=difficulty))
        return configs
