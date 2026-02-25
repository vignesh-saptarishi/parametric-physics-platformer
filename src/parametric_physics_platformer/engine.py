"""Core game engine with rendering and game loop.

Coordinates physics, entities, and pygame rendering into a playable game.
"""

import pygame
from typing import Optional, List, Tuple, Dict, Any

from .physics import PhysicsWorld, PhysicsParams
from .entities import Player, Platform, Goal, Hazard, TimedHazard, FlashingZone, Spring, Collectible
from .config import GameConfig, PhysicsConfig
from .level_gen import LevelGenerator, LevelSpec, build_level
from .annotations import AnnotationCollector


# Colors (RGB)
COLOR_BG = (40, 44, 52)
COLOR_PLAYER = (97, 175, 239)
COLOR_PLATFORM = (152, 195, 121)
COLOR_GOAL = (229, 192, 123)
COLOR_HAZARD = (224, 108, 117)
COLOR_HAZARD_INACTIVE = (100, 180, 100)  # Green for inactive timed hazards (safe)
COLOR_FLASHING_SAFE = (100, 180, 100)  # Green when safe
COLOR_FLASHING_DEADLY = (224, 108, 117)  # Red when deadly
COLOR_SPRING = (255, 200, 100)  # Orange/yellow for springs
COLOR_COLLECTIBLE = (255, 215, 0)  # Gold for collectibles


# Player start position for test level
DEFAULT_PLAYER_START: Tuple[float, float] = (100.0, 200.0)


class PlatformerEngine:
    """Main game engine coordinating all systems.

    Handles:
    - Game loop with fixed timestep
    - Pygame rendering
    - Keyboard input
    - Level management
    """

    def __init__(self, config: Optional[GameConfig] = None):
        """Initialize game engine.

        Args:
            config: Game configuration. Uses defaults if None.
        """
        self.config = config or GameConfig()

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.config.screen_width, self.config.screen_height)
        )
        pygame.display.set_caption("Platformer - Phase 1 Probe")
        self.clock = pygame.time.Clock()

        # Initialize physics - convert PhysicsConfig to PhysicsParams
        physics_params = PhysicsParams(gravity=self.config.physics.gravity)
        self.physics = PhysicsWorld(physics_params)

        # Game state
        self.player: Optional[Player] = None
        self.platforms: List[Platform] = []
        self.goals: List[Goal] = []
        self.hazards: List[Hazard] = []
        self.timed_hazards: List[TimedHazard] = []
        self.flashing_zones: List[FlashingZone] = []
        self.springs: List[Spring] = []
        self.collectibles: List[Collectible] = []

        self.running = False
        self.level_complete = False
        self.player_dead = False
        self.death_cause: Optional[str] = None  # "hazard", "timed_hazard", "flashing_zone", "fall"

        # Score and timer tracking
        self.score = 0
        self.episode_steps = 0

        # Input state
        self._keys_pressed: Dict[int, bool] = {}

        # Level generation state
        self._use_generated_levels = False
        self._player_start = DEFAULT_PLAYER_START
        self._current_spec: Optional[LevelSpec] = None

        # Dynamics model (None = default StandardDynamics)
        self.dynamics_model = None
        self.randomize_dynamics_on_reset = False

        # Randomization on reset
        self.randomize_on_reset = False
        self.ensure_features = False  # Use non-zero minimums when randomizing

        # Annotation mode
        self._annotator: Optional[AnnotationCollector] = None
        self._annotation_state: Optional[str] = None  # None, "physics_feel", "layout_playable", "level_rating", "done"
        self._annotation_feel: Optional[str] = None
        self._annotation_layout: Optional[str] = None
        self._annotation_rating: Optional[int] = None
        self._death_logged = False  # prevent duplicate death logs per episode

        # Camera state (for scrolling levels)
        self.camera_x = 0.0  # Camera position in world coordinates
        self.camera_offset = 0.3  # Player position on screen (0.3 = 30% from left)

    def enable_annotations(self, annotator: AnnotationCollector) -> None:
        """Enable annotation mode for human playtesting.

        When enabled, shows end-of-level prompts for physics feel and level rating,
        and auto-logs deaths with position/timestep/cause.
        """
        self._annotator = annotator

    def _begin_annotation_episode(self) -> None:
        """Start annotation tracking for a new episode."""
        if not self._annotator:
            return
        self._annotator.begin_episode(
            physics_config=self.config.physics.to_dict(),
        )
        self._death_logged = False
        self._annotation_state = None
        self._annotation_feel = None
        self._annotation_layout = None
        self._annotation_rating = None

    def _on_level_end(self) -> None:
        """Called when level ends — auto-log death and start annotation prompt."""
        if not self._annotator:
            return

        # Auto-log death
        if self.player_dead and not self._death_logged:
            pos = self.player.position if self.player else (0, 0)
            self._annotator.log_death(
                position=pos,
                timestep=self.episode_steps,
                cause=self.death_cause,
            )
            self._death_logged = True

        # Start annotation prompt
        if self._annotation_state is None:
            self._annotation_state = "physics_feel"

    def _handle_annotation_key(self, key: int) -> None:
        """Handle key press during annotation overlay."""
        if self._annotation_state == "physics_feel":
            if key == pygame.K_g:
                self._annotation_feel = "good"
                self._annotation_state = "layout_playable"
            elif key == pygame.K_b:
                self._annotation_feel = "bad"
                self._annotation_state = "layout_playable"
            elif key in (pygame.K_RETURN, pygame.K_SPACE):
                self._annotation_state = "layout_playable"

        elif self._annotation_state == "layout_playable":
            if key == pygame.K_y:
                self._annotation_layout = "yes"
                self._annotation_state = "level_rating"
            elif key == pygame.K_n:
                self._annotation_layout = "no"
                self._annotation_state = "level_rating"
            elif key in (pygame.K_RETURN, pygame.K_SPACE):
                self._annotation_state = "level_rating"

        elif self._annotation_state == "level_rating":
            if key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                self._annotation_rating = key - pygame.K_0
                self._finish_annotation()
            elif key in (pygame.K_RETURN, pygame.K_SPACE):
                self._finish_annotation()

    def _finish_annotation(self) -> None:
        """Save annotation and move to 'done' state."""
        if not self._annotator:
            return

        self._annotator.annotate(
            physics_feel=self._annotation_feel,
            layout_playable=self._annotation_layout,
            level_rating=self._annotation_rating,
        )

        outcome = "goal" if self.level_complete else "death" if self.player_dead else "quit"
        self._annotator.end_episode(
            outcome=outcome,
            score=self.score,
            steps=self.episode_steps,
        )
        self._annotator.save()
        self._annotation_state = "done"

    def _render_annotation_overlay(self) -> None:
        """Draw annotation prompt overlay on screen."""
        if not self._annotation_state or self._annotation_state == "done":
            return

        # Semi-transparent backdrop
        overlay = pygame.Surface((self.config.screen_width, self.config.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 28)
        cx = self.config.screen_width // 2
        cy = self.config.screen_height // 2

        if self._annotation_state == "physics_feel":
            title = font.render("How did the physics feel?", True, (255, 255, 255))
            self.screen.blit(title, title.get_rect(center=(cx, cy - 40)))

            options = small_font.render("G = Good    B = Bad    Enter = Skip", True, (200, 200, 200))
            self.screen.blit(options, options.get_rect(center=(cx, cy + 10)))

        elif self._annotation_state == "layout_playable":
            # Show previous answer if given
            if self._annotation_feel:
                feel_text = small_font.render(
                    f"Physics: {self._annotation_feel.upper()}", True, (150, 255, 150)
                )
                self.screen.blit(feel_text, feel_text.get_rect(center=(cx, cy - 70)))

            title = font.render("Were platforms reachable/playable?", True, (255, 255, 255))
            self.screen.blit(title, title.get_rect(center=(cx, cy - 20)))

            options = small_font.render("Y = Yes    N = No    Enter = Skip", True, (200, 200, 200))
            self.screen.blit(options, options.get_rect(center=(cx, cy + 20)))

        elif self._annotation_state == "level_rating":
            # Show previous answers
            prev_y = cy - 90
            if self._annotation_feel:
                feel_text = small_font.render(
                    f"Physics: {self._annotation_feel.upper()}", True, (150, 255, 150)
                )
                self.screen.blit(feel_text, feel_text.get_rect(center=(cx, prev_y)))
                prev_y += 25
            if self._annotation_layout:
                layout_text = small_font.render(
                    f"Layout: {self._annotation_layout.upper()}", True, (150, 255, 150)
                )
                self.screen.blit(layout_text, layout_text.get_rect(center=(cx, prev_y)))

            title = font.render("Rate this level (1-5)?", True, (255, 255, 255))
            self.screen.blit(title, title.get_rect(center=(cx, cy - 20)))

            options = small_font.render("1-5 = Rating    Enter = Skip", True, (200, 200, 200))
            self.screen.blit(options, options.get_rect(center=(cx, cy + 20)))

    def _settle_player(self) -> None:
        """Run physics settle loop so player lands on ground.

        Needed because pymunk requires at least one step to detect collisions,
        and floaty/cubic dynamics have zero gravity at airtime=0 so a single
        step isn't enough — player.update() must increment airtime so gravity
        builds up.
        """
        dt = 1 / self.config.fps
        for _ in range(30):
            self.player.update(dt)
            self.physics.step(dt)
            if self.player.is_grounded:
                break
        # Reset airtime after settling (player is now on ground)
        self.player._airtime = 0.0

    def load_test_level(self) -> None:
        """Load a simple test level for probe iteration.

        This is the minimal level for testing physics feel.
        """
        # Ground platform
        ground = Platform(
            self.physics,
            x=self.config.screen_width / 2,
            y=30,
            width=self.config.screen_width,
            height=40,
        )
        self.platforms.append(ground)

        # Some stepping platforms
        platforms_data = [
            (200, 150, 150),   # (x, y, width)
            (400, 250, 120),
            (600, 350, 100),
            (350, 450, 200),
        ]
        for x, y, w in platforms_data:
            plat = Platform(self.physics, x, y, w)
            self.platforms.append(plat)

        # Goal at the top
        self.goals.append(Goal(self.physics, 350, 520))

        # Create player with physics config for behavioral params
        self.player = Player(
            self.physics,
            DEFAULT_PLAYER_START,
            physics_config=self.config.physics,
            dynamics_model=self.dynamics_model,
        )

        self._settle_player()

        # Store player start for reset
        self._player_start = DEFAULT_PLAYER_START
        self._current_spec: Optional[LevelSpec] = None
        self._use_generated_levels = False

    def load_feature_level(self, feature: str) -> None:
        """Load a test level showcasing a specific feature.

        Args:
            feature: One of 'timed-hazards', 'flashing-zones', 'springs', 'collectibles', 'all'
        """
        self._clear_level()

        # Ground platform
        ground = Platform(
            self.physics,
            x=600,
            y=30,
            width=1200,
            height=40,
        )
        self.platforms.append(ground)

        if feature == "timed-hazards":
            self._setup_timed_hazards_level()
        elif feature == "flashing-zones":
            self._setup_flashing_zones_level()
        elif feature == "springs":
            self._setup_springs_level()
        elif feature == "collectibles":
            self._setup_collectibles_level()
        elif feature == "all":
            self._setup_all_features_level()
        elif feature.startswith("dynamics-"):
            self._setup_dynamics_feature_level(feature)
        else:
            raise ValueError(f"Unknown feature: {feature}")

        # Create player
        self.player = Player(
            self.physics,
            DEFAULT_PLAYER_START,
            physics_config=self.config.physics,
            dynamics_model=self.dynamics_model,
        )

        self._settle_player()

        self._player_start = DEFAULT_PLAYER_START
        self._current_spec = None
        self._use_generated_levels = False

    def _setup_timed_hazards_level(self) -> None:
        """Level showcasing timed hazards with different timings."""
        # Platforms to jump across
        platforms_data = [
            (200, 150, 120),
            (400, 150, 120),
            (600, 150, 120),
            (800, 150, 120),
        ]
        for x, y, w in platforms_data:
            self.platforms.append(Platform(self.physics, x, y, w))

        # Timed hazards between platforms - must time your jumps
        # Equal cycle (2s on, 2s off)
        self.timed_hazards.append(TimedHazard(
            self.physics, 300, 100, width=60, height=80,
            active_duration=2.0, inactive_duration=2.0
        ))
        # Mostly active (3s on, 1.5s off) - tighter window
        self.timed_hazards.append(TimedHazard(
            self.physics, 500, 100, width=60, height=80,
            active_duration=3.0, inactive_duration=1.5
        ))
        # Slow cycle (3s on, 3s off) - more forgiving
        self.timed_hazards.append(TimedHazard(
            self.physics, 700, 100, width=60, height=80,
            active_duration=3.0, inactive_duration=3.0
        ))

        # Goal at end
        self.goals.append(Goal(self.physics, 800, 220))

    def _setup_flashing_zones_level(self) -> None:
        """Level showcasing flashing zones as floor hazards."""
        # Higher platforms to avoid the danger zones
        platforms_data = [
            (200, 200, 100),
            (450, 200, 100),
            (700, 200, 100),
        ]
        for x, y, w in platforms_data:
            self.platforms.append(Platform(self.physics, x, y, w))

        # Flashing floor zones - must time your movement or use platforms
        # Zone 1: Equal timing
        self.flashing_zones.append(FlashingZone(
            self.physics, 325, 70, width=150, height=40,
            safe_duration=2.5, deadly_duration=2.5
        ))
        # Zone 2: Mostly safe
        self.flashing_zones.append(FlashingZone(
            self.physics, 575, 70, width=150, height=40,
            safe_duration=3.0, deadly_duration=1.5
        ))
        # Zone 3: Mostly deadly - must use platform
        self.flashing_zones.append(FlashingZone(
            self.physics, 825, 70, width=150, height=40,
            safe_duration=1.5, deadly_duration=3.0, start_safe=False
        ))

        # Goal at end
        self.goals.append(Goal(self.physics, 900, 100))

    def _setup_springs_level(self) -> None:
        """Level showcasing springs for vertical traversal."""
        # Low platform
        self.platforms.append(Platform(self.physics, 250, 100, 150))

        # High platforms - need springs to reach
        self.platforms.append(Platform(self.physics, 450, 350, 150))
        self.platforms.append(Platform(self.physics, 650, 500, 150))

        # Calculate spring impulse from config (multiplier * normal jump)
        base_impulse = self.config.physics.jump_impulse
        multiplier = self.config.dynamics.spring_multiplier
        spring_impulse = base_impulse * multiplier

        # Springs to launch player
        # Spring 1: On ground, launches to first high platform
        self.springs.append(Spring(
            self.physics, 350, 60, width=50, height=20,
            launch_impulse=spring_impulse
        ))
        # Spring 2: On low platform, launches higher
        self.springs.append(Spring(
            self.physics, 250, 120, width=50, height=20,
            launch_impulse=spring_impulse * 0.9  # Slightly weaker
        ))
        # Spring 3: Chain to reach top
        self.springs.append(Spring(
            self.physics, 450, 370, width=50, height=20,
            launch_impulse=spring_impulse
        ))

        # Goal at top
        self.goals.append(Goal(self.physics, 650, 570))

    def _setup_all_features_level(self) -> None:
        """Level combining all new features."""
        # Platforms
        platforms_data = [
            (200, 150, 120),
            (500, 150, 120),
            (350, 350, 100),
            (650, 450, 150),
        ]
        for x, y, w in platforms_data:
            self.platforms.append(Platform(self.physics, x, y, w))

        # Timed hazard blocking path
        self.timed_hazards.append(TimedHazard(
            self.physics, 350, 100, width=60, height=60,
            active_duration=2.5, inactive_duration=2.0
        ))

        # Flashing zone on ground
        self.flashing_zones.append(FlashingZone(
            self.physics, 450, 70, width=100, height=40,
            safe_duration=2.5, deadly_duration=2.0
        ))

        # Springs for vertical movement (uses config multiplier)
        spring_impulse = self.config.physics.jump_impulse * self.config.dynamics.spring_multiplier
        self.springs.append(Spring(
            self.physics, 200, 170, width=40, height=15,
            launch_impulse=spring_impulse
        ))
        self.springs.append(Spring(
            self.physics, 500, 170, width=40, height=15,
            launch_impulse=spring_impulse
        ))

        # Collectibles scattered around
        self.collectibles.append(Collectible(self.physics, 350, 400, value=1))
        self.collectibles.append(Collectible(self.physics, 550, 100, value=1))
        self.collectibles.append(Collectible(self.physics, 650, 500, value=3))

        # Goal at top
        self.goals.append(Goal(self.physics, 650, 520))

    def _setup_collectibles_level(self) -> None:
        """Level showcasing collectibles with exploration vs speed trade-off."""
        # Main path platforms
        platforms_data = [
            (200, 150, 120),
            (400, 150, 120),
            (600, 150, 120),
            (800, 150, 120),
        ]
        for x, y, w in platforms_data:
            self.platforms.append(Platform(self.physics, x, y, w))

        # Optional high platform for bonus collectibles
        self.platforms.append(Platform(self.physics, 300, 350, 100))
        self.platforms.append(Platform(self.physics, 500, 450, 100))

        # Main path collectibles (easy to get)
        self.collectibles.append(Collectible(self.physics, 200, 200, value=1))
        self.collectibles.append(Collectible(self.physics, 400, 200, value=1))
        self.collectibles.append(Collectible(self.physics, 600, 200, value=1))

        # Bonus collectibles (require detour/platforming)
        self.collectibles.append(Collectible(self.physics, 300, 400, value=5))
        self.collectibles.append(Collectible(self.physics, 500, 500, value=5))

        # Goal at end
        self.goals.append(Goal(self.physics, 800, 220))

    def _setup_dynamics_feature_level(self, feature: str) -> None:
        """Level for testing dynamics variants with clear platforming challenges."""
        from .dynamics import (
            CubicDynamics, FloatyDynamics, AsymmetricDynamics,
            VelocityDynamics, ImpulseDynamics, DragDynamics,
        )

        dynamics_map = {
            "dynamics-cubic": CubicDynamics,
            "dynamics-floaty": FloatyDynamics,
            "dynamics-asymmetric": AsymmetricDynamics,
            "dynamics-velocity": VelocityDynamics,
            "dynamics-impulse": ImpulseDynamics,
            "dynamics-drag": DragDynamics,
        }

        if feature not in dynamics_map:
            raise ValueError(f"Unknown dynamics feature: {feature}")

        # Set the dynamics model
        cls = dynamics_map[feature]
        self.dynamics_model = cls(physics_config=self.config.physics)

        # Platforms to test vertical + horizontal feel
        platforms_data = [
            (200, 150, 150),
            (450, 250, 120),
            (700, 350, 100),
            (400, 450, 200),
            (150, 350, 80),
        ]
        for x, y, w in platforms_data:
            self.platforms.append(Platform(self.physics, x, y, w))

        # Goal at the top
        self.goals.append(Goal(self.physics, 400, 520))

    def load_generated_level(self, seed: Optional[int] = None) -> LevelSpec:
        """Generate and load a level from config parameters.

        Uses the full GameConfig (physics, layout, dynamics, objectives)
        to generate a playable level.

        Args:
            seed: Random seed for reproducible generation.

        Returns:
            LevelSpec describing the generated level.
        """
        # Clear any existing level
        self._clear_level()

        # Generate level from config
        generator = LevelGenerator.from_config(self.config)
        spec = generator.generate(seed=seed)

        # Build entities from spec
        (self.platforms, self.goals, self.hazards,
         self.timed_hazards, self.flashing_zones,
         self.springs, self.collectibles) = build_level(self.physics, spec)

        # Create player at start position
        self.player = Player(
            self.physics,
            spec.player_start,
            physics_config=self.config.physics,
            dynamics_model=self.dynamics_model,
        )

        self._settle_player()

        # Store for reset
        self._player_start = spec.player_start
        self._current_spec = spec
        self._use_generated_levels = True

        return spec

    def _clear_level(self) -> None:
        """Remove all entities from the level."""
        # Remove player (has its own body)
        if self.player:
            self.physics.remove_body(self.player.body)
            self.player = None

        # Remove platforms (static shapes on space.static_body)
        for plat in self.platforms:
            self.physics.remove_shape(plat.shape)
        self.platforms = []

        # Remove goals and hazards (also static shapes)
        for goal in self.goals:
            self.physics.remove_shape(goal.shape)
        self.goals = []

        for hazard in self.hazards:
            if hazard.body:
                self.physics.remove_body(hazard.body)
            else:
                self.physics.remove_shape(hazard.shape)
        self.hazards = []

        # Remove new entity types
        for th in self.timed_hazards:
            self.physics.remove_shape(th.shape)
        self.timed_hazards = []

        for fz in self.flashing_zones:
            self.physics.remove_shape(fz.shape)
        self.flashing_zones = []

        for spring in self.springs:
            self.physics.remove_shape(spring.shape)
        self.springs = []

        for collectible in self.collectibles:
            self.physics.remove_shape(collectible.shape)
        self.collectibles = []

    def reset(self) -> None:
        """Reset level - generates new level if using procedural generation."""
        # If annotation overlay is active and not done, finish it first (skip)
        if self._annotator and self._annotation_state in ("physics_feel", "layout_playable", "level_rating"):
            self._finish_annotation()

        # Reset state flags
        self.level_complete = False
        self.player_dead = False
        self.death_cause = None
        self.score = 0
        self.episode_steps = 0

        # Randomize full config if enabled
        if self.randomize_on_reset:
            from .config import GameConfig
            new_config = GameConfig.sample_full(ensure_features=self.ensure_features)
            self.config.physics = new_config.physics
            self.config.layout = new_config.layout
            self.config.dynamics = new_config.dynamics
            self.config.objectives = new_config.objectives
            # Update gravity in physics world
            self.physics.set_gravity(self.config.physics.gravity)
            p = self.config.physics
            d = self.config.dynamics
            o = self.config.objectives
            print(f"NEW CONFIG: jump={p.jump_height:.0f}px air={p.air_control:.2f} "
                  f"hazards={d.hazard_density:.2f} springs={d.spring_density:.2f} "
                  f"collectibles={o.collectibles}")

        # Randomize dynamics model if enabled
        if self.randomize_dynamics_on_reset:
            import random as _rng
            from .dynamics import create_dynamics, VerticalModel, HorizontalModel
            v = _rng.choice(list(VerticalModel))
            h = _rng.choice(list(HorizontalModel))
            self.dynamics_model = create_dynamics(
                vertical=v, horizontal=h,
                physics_config=self.config.physics,
            )
            print(f"NEW DYNAMICS: {v.name.lower()}/{h.name.lower()}")

        # If using generated levels, create a completely new level
        if self._use_generated_levels:
            spec = self.load_generated_level(seed=None)  # New random seed
            # Show platform positions for debugging
            plat_positions = [(int(p.x), int(p.y)) for p in self.platforms[1:4]]  # Skip ground, show first 3
            print(f"NEW LEVEL: {len(spec.platforms)} platforms | first 3: {plat_positions}")
            self._begin_annotation_episode()
            return

        # Otherwise just reset player position
        if self.player:
            self.physics.remove_body(self.player.body)
            self.player = None

        # Get player start position
        player_start = getattr(self, '_player_start', DEFAULT_PLAYER_START)

        # Recreate player with physics config and dynamics model
        self.player = Player(
            self.physics,
            player_start,
            physics_config=self.config.physics,
            dynamics_model=self.dynamics_model,
        )

        self._settle_player()

        # Reset goals
        for goal in self.goals:
            goal.reached = False

        # Reset hazard triggers
        self.physics.clear_hazard_triggers()

        self._begin_annotation_episode()

    def handle_events(self) -> None:
        """Process pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                # Annotation overlay intercepts keys when active
                if self._annotator and self._annotation_state in ("physics_feel", "layout_playable", "level_rating"):
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    else:
                        self._handle_annotation_key(event.key)
                    continue

                self._keys_pressed[event.key] = True
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    self.reset()
            elif event.type == pygame.KEYUP:
                self._keys_pressed[event.key] = False

    def handle_input(self) -> None:
        """Process keyboard input for player control."""
        if not self.player or self.level_complete or self.player_dead:
            return

        # Movement
        if self._keys_pressed.get(pygame.K_LEFT) or self._keys_pressed.get(pygame.K_a):
            self.player.move_left()
        if self._keys_pressed.get(pygame.K_RIGHT) or self._keys_pressed.get(pygame.K_d):
            self.player.move_right()

        # Jump
        if self._keys_pressed.get(pygame.K_SPACE) or self._keys_pressed.get(pygame.K_w):
            self.player.jump()

    def update(self, dt: float) -> None:
        """Update game state.

        Args:
            dt: Time step in seconds.
        """
        if self.player:
            self.player.update()

        # Update timed entities
        for hazard in self.hazards:
            hazard.update(dt)

        for th in self.timed_hazards:
            th.update(dt)

        for fz in self.flashing_zones:
            fz.update(dt)

        for spring in self.springs:
            spring.reset_trigger()

        # Step physics
        self.physics.step(dt)

        # Increment step counter (before checking win/lose so final step counts)
        if not self.level_complete and not self.player_dead:
            self.episode_steps += 1

        # Check collectibles
        for collectible in self.collectibles:
            if collectible.collected:
                self.score += collectible.value
                # Remove from physics so it doesn't trigger again
                self.physics.remove_shape(collectible.shape)
        # Remove collected items from list
        self.collectibles = [c for c in self.collectibles if not c.collected]

        # Check win/lose conditions
        for goal in self.goals:
            if goal.reached:
                self.level_complete = True

        for hazard in self.hazards:
            if hazard.triggered:
                self.player_dead = True
                if not self.death_cause:
                    self.death_cause = "hazard"

        for th in self.timed_hazards:
            if th.triggered:
                self.player_dead = True
                if not self.death_cause:
                    self.death_cause = "timed_hazard"

        for fz in self.flashing_zones:
            if fz.triggered:
                self.player_dead = True
                if not self.death_cause:
                    self.death_cause = "flashing_zone"

        # Check if player fell off screen
        if self.player:
            _, y = self.player.position
            if y < -100:
                self.player_dead = True
                if not self.death_cause:
                    self.death_cause = "fall"

        # Trigger annotation overlay on level end
        if (self.level_complete or self.player_dead) and self._annotator:
            self._on_level_end()

        # Update camera to follow player
        self._update_camera()

    def _update_camera(self) -> None:
        """Update camera position to follow player."""
        if not self.player:
            return

        player_x, _ = self.player.position

        # Camera follows player, keeping them at camera_offset from left
        target_x = player_x - self.config.screen_width * self.camera_offset

        # Smooth camera follow (lerp)
        self.camera_x += (target_x - self.camera_x) * 0.1

        # Don't scroll past left edge
        self.camera_x = max(0, self.camera_x)

    def _world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates.

        Pymunk uses bottom-left origin, pygame uses top-left.
        Accounts for camera position for scrolling.
        """
        screen_x = int(x - self.camera_x)
        screen_y = int(self.config.screen_height - y)
        return screen_x, screen_y

    def render(self) -> None:
        """Render current game state."""
        self.screen.fill(COLOR_BG)

        # Draw platforms
        for plat in self.platforms:
            left, bottom, right, top = plat.bounds
            screen_left, screen_top = self._world_to_screen(left, top)
            width = int(right - left)
            height = int(top - bottom)
            pygame.draw.rect(
                self.screen, COLOR_PLATFORM,
                (screen_left, screen_top, width, height)
            )

        # Draw goals
        for goal in self.goals:
            left, bottom, right, top = goal.bounds
            screen_left, screen_top = self._world_to_screen(left, top)
            width = int(right - left)
            height = int(top - bottom)
            pygame.draw.rect(
                self.screen, COLOR_GOAL,
                (screen_left, screen_top, width, height)
            )

        # Draw hazards
        for hazard in self.hazards:
            left, bottom, right, top = hazard.bounds
            screen_left, screen_top = self._world_to_screen(left, top)
            width = int(right - left)
            height = int(top - bottom)
            pygame.draw.rect(
                self.screen, COLOR_HAZARD,
                (screen_left, screen_top, width, height)
            )

        # Draw timed hazards (dimmed when inactive)
        for th in self.timed_hazards:
            left, bottom, right, top = th.bounds
            screen_left, screen_top = self._world_to_screen(left, top)
            width = int(right - left)
            height = int(top - bottom)
            color = COLOR_HAZARD if th.active else COLOR_HAZARD_INACTIVE
            pygame.draw.rect(
                self.screen, color,
                (screen_left, screen_top, width, height)
            )

        # Draw flashing zones (green when safe, red when deadly, always red border)
        for fz in self.flashing_zones:
            left, bottom, right, top = fz.bounds
            screen_left, screen_top = self._world_to_screen(left, top)
            width = int(right - left)
            height = int(top - bottom)
            # Fill with current state color
            color = COLOR_FLASHING_SAFE if fz.safe else COLOR_FLASHING_DEADLY
            pygame.draw.rect(
                self.screen, color,
                (screen_left, screen_top, width, height)
            )
            # Always draw red border to indicate danger
            pygame.draw.rect(
                self.screen, COLOR_HAZARD,
                (screen_left, screen_top, width, height),
                width=3  # Border thickness
            )

        # Draw springs
        for spring in self.springs:
            left, bottom, right, top = spring.bounds
            screen_left, screen_top = self._world_to_screen(left, top)
            width = int(right - left)
            height = int(top - bottom)
            pygame.draw.rect(
                self.screen, COLOR_SPRING,
                (screen_left, screen_top, width, height)
            )

        # Draw collectibles
        for collectible in self.collectibles:
            left, bottom, right, top = collectible.bounds
            screen_left, screen_top = self._world_to_screen(left, top)
            width = int(right - left)
            height = int(top - bottom)
            pygame.draw.rect(
                self.screen, COLOR_COLLECTIBLE,
                (screen_left, screen_top, width, height)
            )

        # Draw player
        if self.player:
            px, py = self.player.position
            half_w = self.player.config.width / 2
            half_h = self.player.config.height / 2
            screen_x, screen_y = self._world_to_screen(px - half_w, py + half_h)
            pygame.draw.rect(
                self.screen, COLOR_PLAYER,
                (screen_x, screen_y, int(self.player.config.width), int(self.player.config.height))
            )

        # Draw status text (show simpler message when annotation overlay will appear)
        if self._annotator and self._annotation_state in ("physics_feel", "layout_playable", "level_rating"):
            # Annotation overlay handles the message
            pass
        elif self.level_complete:
            self._draw_text("LEVEL COMPLETE! Press R to restart", (255, 255, 255))
        elif self.player_dead:
            self._draw_text("GAME OVER! Press R to restart", COLOR_HAZARD)

        # Draw score and timer (top right)
        font = pygame.font.Font(None, 28)
        time_seconds = self.episode_steps / self.config.fps
        score_text = f"Score: {self.score}  |  Time: {time_seconds:.1f}s"
        score_surface = font.render(score_text, True, COLOR_COLLECTIBLE)
        score_rect = score_surface.get_rect(topright=(self.config.screen_width - 10, 10))
        self.screen.blit(score_surface, score_rect)

        # Draw debug info - show behavioral params, not raw physics
        if self.player:
            vx, vy = self.player.velocity
            grounded = "Yes" if self.player.is_grounded else "No"
            p = self.config.physics
            dyn_name = self.player.dynamics_model.vertical.name.lower() if self.dynamics_model else "standard"
            debug_text = f"Vel: ({vx:.0f}, {vy:.0f}) | Grounded: {grounded} | Jump: {p.jump_height:.0f}px/{p.jump_duration:.2f}s | Dyn: {dyn_name}"
            debug_font = pygame.font.Font(None, 24)
            text_surface = debug_font.render(debug_text, True, (200, 200, 200))
            self.screen.blit(text_surface, (10, 10))

        # Annotation overlay (drawn on top of everything)
        if self._annotator:
            self._render_annotation_overlay()

        pygame.display.flip()

    def _draw_text(self, text: str, color: Tuple[int, int, int]) -> None:
        """Draw centered text on screen."""
        font = pygame.font.Font(None, 48)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(
            center=(self.config.screen_width // 2, self.config.screen_height // 2)
        )
        self.screen.blit(text_surface, text_rect)

    def run(self) -> None:
        """Main game loop."""
        self.running = True
        dt = 1.0 / self.config.fps

        # Start annotation tracking for the initial level
        self._begin_annotation_episode()

        while self.running:
            self.handle_events()
            self.handle_input()
            self.update(dt)
            self.render()
            self.clock.tick(self.config.fps)

        # Save any in-progress annotation on quit
        if self._annotator and self._annotator.recording:
            self._annotator.end_episode(
                outcome="quit", score=self.score, steps=self.episode_steps
            )
            self._annotator.save()

        pygame.quit()

    def get_state(self) -> Dict[str, Any]:
        """Get current game state for observation/logging.

        Returns:
            Dictionary with player position, velocity, and flags.
        """
        state = {
            "level_complete": self.level_complete,
            "player_dead": self.player_dead,
            "death_cause": self.death_cause,
        }
        if self.player:
            state["player_position"] = self.player.position
            state["player_velocity"] = self.player.velocity
            state["player_grounded"] = self.player.is_grounded
        return state
