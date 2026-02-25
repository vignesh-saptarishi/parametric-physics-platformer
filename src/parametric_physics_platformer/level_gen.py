"""Procedural level generation using measured behavioral references.

Generates levels with a controllable difficulty distribution by combining:
- BehavioralProfile: Measured reachability (actual apex, max speed, jump reach)
- LayoutConfig: Platform density, gap scaling, difficulty_sigma
- DynamicsConfig: Hazards, springs, dynamics model
- ObjectiveConfig: Goals, collectibles (platform + airspace)

Key design: Platform placement is distribution-based, centered on measured
behavioral references with configurable noise (difficulty_sigma). Each level
naturally contains a gradient from easy to impossible transitions. No
playability guarantee — the world model must discover what's reachable.
"""

import random
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, TYPE_CHECKING

from .config import PhysicsConfig, LayoutConfig, DynamicsConfig, ObjectiveConfig, GameConfig
from .physics import PhysicsWorld
from .entities import Platform, Goal, Hazard, TimedHazard, FlashingZone, Spring, Collectible

if TYPE_CHECKING:
    from .calibration import BehavioralProfile


@dataclass
class LevelSpec:
    """Specification for a generated level."""
    platforms: List[Tuple[float, float, float]]  # (x, y, width)
    goals: List[Tuple[float, float]]  # (x, y)
    hazards: List[Tuple[float, float]]  # (x, y)
    player_start: Tuple[float, float]

    # New entity specs
    timed_hazards: Optional[List[Tuple[float, float, float, float]]] = None  # (x, y, active_dur, inactive_dur)
    flashing_zones: Optional[List[Tuple[float, float, float, float, float]]] = None  # (x, y, width, safe_dur, deadly_dur)
    springs: Optional[List[Tuple[float, float, float]]] = None  # (x, y, impulse)
    collectibles: Optional[List[Tuple[float, float, int]]] = None  # (x, y, value)

    # Metadata for analysis
    total_width: float = 0.0
    max_height: float = 0.0
    num_gaps: int = 0
    avg_gap_size: float = 0.0

    def __post_init__(self):
        if self.timed_hazards is None:
            self.timed_hazards = []
        if self.flashing_zones is None:
            self.flashing_zones = []
        if self.springs is None:
            self.springs = []
        if self.collectibles is None:
            self.collectibles = []


class LevelGenerator:
    """Generates levels using measured behavioral references.

    Uses a BehavioralProfile (measured actual reachability) to center
    platform placement, with LayoutConfig.difficulty_sigma controlling
    the spread. Each level contains a natural gradient from easy to
    impossible transitions.
    """

    def __init__(
        self,
        physics: PhysicsConfig,
        layout: LayoutConfig,
        dynamics: DynamicsConfig,
        objectives: ObjectiveConfig,
        screen_width: int = 800,
        screen_height: int = 600,
        behavioral_profile: Optional["BehavioralProfile"] = None,
    ):
        self.physics = physics
        self.layout = layout
        self.dynamics = dynamics
        self.objectives = objectives
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.behavioral_profile = behavioral_profile

        # Compute reachability from measured profile or fall back to declared params
        self._compute_reachability()

    def _compute_reachability(self) -> None:
        """Compute reachability from behavioral profile (measured) or physics config (fallback)."""
        if self.behavioral_profile is not None:
            # Use measured behavioral references — accurate for all dynamics types
            self.max_jump_height = self.behavioral_profile.actual_apex_height
            self.max_jump_distance = self.behavioral_profile.horizontal_jump_reach
        else:
            # Fallback: declared params (only accurate for parabolic dynamics)
            self.max_jump_height = self.physics.jump_height
            air_time = 2 * self.physics.jump_duration
            avg_control = (1.0 + self.physics.air_control) / 2
            self.max_jump_distance = self.physics.move_speed * air_time * avg_control

        # Reference points for platform placement distribution
        self.comfortable_height = self.max_jump_height * 0.7
        self.comfortable_distance = self.max_jump_distance * 0.6

    def generate(self, seed: Optional[int] = None) -> LevelSpec:
        """Generate a level from current config.

        Args:
            seed: Random seed for reproducibility

        Returns:
            LevelSpec with all level elements
        """
        if seed is not None:
            random.seed(seed)

        platforms = []
        hazards = []
        timed_hazards = []
        flashing_zones = []
        springs = []
        collectibles = []
        goals = []

        # Level width from config (can be wider than screen for scrolling)
        level_width = max(self.layout.level_length, self.screen_width)

        # Ground platform spans entire level
        ground_y = 50
        platforms.append((level_width / 2, ground_y, level_width))

        # Generate platforms based on layout config
        platforms.extend(self._generate_platforms(ground_y))

        # Place hazards based on dynamics config (split between static and timed)
        if self.dynamics.hazard_density > 0:
            all_hazard_positions = self._place_hazards(platforms)
            num_timed = int(len(all_hazard_positions) * self.dynamics.timed_hazard_ratio)
            for i, (hx, hy) in enumerate(all_hazard_positions):
                if i < num_timed:
                    active_dur = random.uniform(1.5, 3.0)
                    inactive_dur = random.uniform(1.0, 2.5)
                    timed_hazards.append((hx, hy, active_dur, inactive_dur))
                else:
                    hazards.append((hx, hy))

        # Place flashing zones on ground
        if self.dynamics.flashing_zone_count > 0:
            flashing_zones.extend(self._place_flashing_zones(platforms, ground_y))

        # Place springs on platforms
        if self.dynamics.spring_density > 0:
            springs.extend(self._place_springs(platforms))

        # Place platform-based collectibles
        if self.objectives.collectibles > 0:
            collectibles.extend(self._place_collectibles(platforms))

        # Place airspace collectibles (random positions in jumpable region)
        if self.objectives.airspace_collectibles > 0:
            collectibles.extend(self._place_airspace_collectibles(ground_y))

        # Place goal based on objectives config
        goal_pos = self._place_goal(platforms)
        goals.append(goal_pos)

        # Player starts on ground
        player_start = (100.0, ground_y + 50)

        # Compute metadata
        xs = [p[0] for p in platforms]
        ys = [p[1] for p in platforms]
        widths = [p[2] for p in platforms]

        return LevelSpec(
            platforms=platforms,
            goals=goals,
            hazards=hazards,
            timed_hazards=timed_hazards,
            flashing_zones=flashing_zones,
            springs=springs,
            collectibles=collectibles,
            player_start=player_start,
            total_width=max(x + w/2 for x, w in zip(xs, widths)),
            max_height=max(ys),
            num_gaps=len(platforms) - 1,
            avg_gap_size=self._compute_avg_gap(platforms),
        )

    def _generate_platforms(self, ground_y: float) -> List[Tuple[float, float, float]]:
        """Generate floating platforms using distribution-based placement.

        Platform heights are drawn from a Normal distribution centered on
        the measured behavioral reference (comfortable_height) with spread
        controlled by difficulty_sigma. This naturally produces a gradient
        from easy (near center) to impossible (in tails).
        """
        platforms = []
        sigma = self.layout.difficulty_sigma

        # Determine number of platforms from density WITH RANDOMNESS
        base_platforms = int(3 + self.layout.platform_density * 7)  # 3-10 base
        num_platforms = base_platforms + random.randint(-2, 2)  # Add variance
        num_platforms = max(2, num_platforms)  # At least 2

        # Platform width range based on difficulty
        min_width = 60 + (1 - self.layout.platform_density) * 60  # 60-120
        max_width = 120 + (1 - self.layout.platform_density) * 80  # 120-200

        # Height distribution parameters derived from measured reachability
        height_center = ground_y + self.comfortable_height
        height_spread = self.max_jump_height * sigma

        # Gap distribution parameters derived from measured reachability
        gap_center = self.comfortable_distance * 0.5
        gap_spread = self.max_jump_distance * sigma * 0.5

        # Generate platform chain
        x = 100 + random.uniform(0, 100)  # Randomize start position

        for i in range(num_platforms):
            # Platform dimensions
            width = random.uniform(min_width, max_width)

            # Height from Normal distribution centered on comfortable height
            y = random.gauss(height_center, max(height_spread, 10.0))

            # Occasionally make platforms at similar heights
            if random.random() < 0.3 and i > 0 and platforms:
                prev_y = platforms[-1][1]
                y = prev_y + random.gauss(0, 30)

            # Clamp to screen bounds only (not to reachability)
            y = max(ground_y + 40, min(self.screen_height - 50, y))

            # Gap from Normal distribution centered on comfortable distance
            gap = random.gauss(gap_center, max(gap_spread, 15.0))
            gap = max(20, gap)  # Minimum gap so platforms don't overlap

            # Don't go past level length
            level_width = max(self.layout.level_length, self.screen_width)
            if x + width/2 > level_width - 50:
                break

            platforms.append((x + width/2, y, width))

            x += width + gap

        return platforms

    def _place_hazards(self, platforms: List[Tuple[float, float, float]]) -> List[Tuple[float, float]]:
        """Place hazards based on dynamics config."""
        hazards = []

        # Number of hazards based on density
        num_hazards = int(self.dynamics.hazard_density * len(platforms))

        # Place hazards near platforms (but not blocking them)
        available_platforms = platforms[1:]  # Skip ground
        random.shuffle(available_platforms)

        for i in range(min(num_hazards, len(available_platforms))):
            px, py, pw = available_platforms[i]

            # Place hazard below or beside platform
            if random.random() < 0.5:
                # Below platform (pit hazard)
                hx = px + random.uniform(-pw/3, pw/3)
                hy = py - 40
            else:
                # Beside platform
                side = random.choice([-1, 1])
                hx = px + side * (pw/2 + 20)
                hy = py

            hazards.append((hx, hy))

        return hazards

    def _place_flashing_zones(
        self, platforms: List[Tuple[float, float, float]], ground_y: float
    ) -> List[Tuple[float, float, float, float, float]]:
        """Place flashing zones on the ground between platforms."""
        zones = []
        floating = sorted(platforms[1:], key=lambda p: p[0])

        # Place zones in gaps between early floating platforms
        for i in range(min(self.dynamics.flashing_zone_count, len(floating))):
            px, py, pw = floating[i]
            zone_x = px
            zone_width = random.uniform(80, 150)
            safe_dur = random.uniform(1.5, 3.0)
            deadly_dur = random.uniform(1.0, 2.5)
            zones.append((zone_x, ground_y + 10, zone_width, safe_dur, deadly_dur))

        return zones

    def _place_springs(
        self, platforms: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        """Place springs on platforms based on spring_density."""
        springs = []
        floating = platforms[1:]  # Skip ground
        num_springs = max(1, int(self.dynamics.spring_density * len(floating)))

        candidates = list(floating)
        random.shuffle(candidates)

        spring_impulse = self.physics.jump_impulse * self.dynamics.spring_multiplier

        for i in range(min(num_springs, len(candidates))):
            px, py, pw = candidates[i]
            # Place spring on top of platform
            springs.append((px, py + 20, spring_impulse))

        return springs

    def _place_collectibles(
        self, platforms: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, int]]:
        """Place collectibles around the level."""
        collectibles = []
        floating = platforms[1:]
        num = self.objectives.collectibles

        # Scatter collectibles above platforms
        candidates = list(floating)
        random.shuffle(candidates)

        for i in range(min(num, len(candidates))):
            px, py, pw = candidates[i]
            cx = px + random.uniform(-pw/3, pw/3)
            cy = py + random.uniform(30, 60)
            value = 1
            collectibles.append((cx, cy, value))

        # If we need more collectibles than platforms, add some on ground
        remaining = num - len(collectibles)
        if remaining > 0:
            level_width = max(self.layout.level_length, self.screen_width)
            for _ in range(remaining):
                cx = random.uniform(150, level_width - 100)
                cy = 100 + random.uniform(0, 30)
                collectibles.append((cx, cy, 1))

        return collectibles

    def _place_airspace_collectibles(
        self,
        ground_y: float,
    ) -> List[Tuple[float, float, int]]:
        """Place collectibles at random positions in the jumpable airspace.

        These are NOT above platforms — they float in open air, making
        trajectory shape decision-relevant for RL agents. Some will be
        reachable, some won't, depending on dynamics and trajectory shape.

        Height ceiling accounts for springs (which can launch well above
        normal jump apex) and uses the full screen height. With high jumps,
        low gravity, or spring bounces, players can reach surprisingly high.
        """
        collectibles = []
        num = self.objectives.airspace_collectibles
        level_width = max(self.layout.level_length, self.screen_width)

        # Height range: use full vertical space
        # - min: just above ground (easy pickups)
        # - max: screen height (springs can launch 2-3x jump height)
        min_y = ground_y + 40
        spring_reach = self.max_jump_height * self.dynamics.spring_multiplier
        max_y = ground_y + max(self.max_jump_height * 1.5, spring_reach)
        # Clamp to screen bounds
        max_y = min(max_y, self.screen_height - 20)

        for _ in range(num):
            cx = random.uniform(150, level_width - 100)
            cy = random.uniform(min_y, max(min_y + 10, max_y))
            collectibles.append((cx, cy, 1))

        return collectibles

    def _place_goal(self, platforms: List[Tuple[float, float, float]]) -> Tuple[float, float]:
        """Place goal on the rightmost platform."""
        if len(platforms) <= 1:
            # Only ground, place goal on right side
            return (self.screen_width - 100, 100)

        # Find rightmost floating platform (excluding ground)
        floating_platforms = platforms[1:]
        if not floating_platforms:
            return (self.screen_width - 100, 100)

        rightmost = max(floating_platforms, key=lambda p: p[0])
        px, py, pw = rightmost
        return (px, py + 50)

    def _compute_avg_gap(self, platforms: List[Tuple[float, float, float]]) -> float:
        """Compute average gap between platforms."""
        if len(platforms) < 2:
            return 0.0

        gaps = []
        sorted_platforms = sorted(platforms, key=lambda p: p[0])

        for i in range(len(sorted_platforms) - 1):
            x1, _, w1 = sorted_platforms[i]
            x2, _, w2 = sorted_platforms[i + 1]
            gap = (x2 - w2/2) - (x1 + w1/2)
            if gap > 0:
                gaps.append(gap)

        return sum(gaps) / len(gaps) if gaps else 0.0

    @classmethod
    def from_config(
        cls,
        config: GameConfig,
        behavioral_profile: Optional["BehavioralProfile"] = None,
    ) -> "LevelGenerator":
        """Create generator from full game config.

        Args:
            config: Full game configuration.
            behavioral_profile: Measured behavioral outcomes from calibration.
                If provided, level gen uses measured reachability. If None,
                falls back to declared PhysicsConfig params (only accurate
                for parabolic dynamics).
        """
        return cls(
            physics=config.physics,
            layout=config.layout,
            dynamics=config.dynamics,
            objectives=config.objectives,
            screen_width=config.screen_width,
            screen_height=config.screen_height,
            behavioral_profile=behavioral_profile,
        )


def build_level(
    physics: PhysicsWorld,
    spec: LevelSpec,
) -> Tuple[
    List[Platform], List[Goal], List[Hazard],
    List[TimedHazard], List[FlashingZone], List[Spring], List[Collectible],
]:
    """Build actual game entities from a level spec.

    Args:
        physics: Physics world to add entities to
        spec: Level specification

    Returns:
        Tuple of (platforms, goals, hazards, timed_hazards, flashing_zones, springs, collectibles)
    """
    platforms = []
    for x, y, width in spec.platforms:
        platforms.append(Platform(physics, x, y, width))

    goals = []
    for x, y in spec.goals:
        goals.append(Goal(physics, x, y))

    hazards = []
    for x, y in spec.hazards:
        hazards.append(Hazard(physics, x, y))

    timed_hazards = []
    for x, y, active_dur, inactive_dur in spec.timed_hazards:
        timed_hazards.append(TimedHazard(
            physics, x, y, active_duration=active_dur, inactive_duration=inactive_dur
        ))

    flashing_zones = []
    for x, y, width, safe_dur, deadly_dur in spec.flashing_zones:
        flashing_zones.append(FlashingZone(
            physics, x, y, width=width, safe_duration=safe_dur, deadly_duration=deadly_dur
        ))

    springs = []
    for x, y, impulse in spec.springs:
        springs.append(Spring(physics, x, y, launch_impulse=impulse))

    collectibles = []
    for x, y, value in spec.collectibles:
        collectibles.append(Collectible(physics, x, y, value=value))

    return platforms, goals, hazards, timed_hazards, flashing_zones, springs, collectibles
