"""Gymnasium environment wrapper for the platformer.

Provides standard Gym API for RL training and data collection.
Observations include both RGB frames and a structured state vector.
"""

import numpy as np
import gymnasium
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple

import pygame

from .config import GameConfig
from .physics import PhysicsWorld, PhysicsParams
from .entities import (
    Player, Platform, Goal, Hazard,
    TimedHazard, FlashingZone, Spring, Collectible,
)
from .level_gen import LevelGenerator, LevelSpec, build_level
from .dynamics import (
    create_dynamics, VerticalModel, HorizontalModel, DynamicsModel,
)
from .calibration import calibrate, BehavioralProfile


# Semantic colors (same as engine.py)
_COLOR_BG = (40, 44, 52)
_COLOR_PLAYER = (97, 175, 239)
_COLOR_PLATFORM = (152, 195, 121)
_COLOR_GOAL = (229, 192, 123)
_COLOR_HAZARD = (224, 108, 117)
_COLOR_HAZARD_INACTIVE = (100, 180, 100)
_COLOR_FLASHING_SAFE = (100, 180, 100)
_COLOR_FLASHING_DEADLY = (224, 108, 117)
_COLOR_SPRING = (255, 200, 100)
_COLOR_COLLECTIBLE = (255, 215, 0)


class PlatformerEnv(gymnasium.Env):
    """Gymnasium wrapper for the platformer.

    Observation space (Dict):
        'rgb': uint8 array of shape (H, W, 3) - rendered frame
        'state': float32 array of shape (16,) - state vector containing:
            [0-1] player position (x, y)
            [2-3] player velocity (vx, vy)
            [4]   player grounded (0/1)
            [5-9] physics config (jump_height, jump_duration, move_speed, air_control, ground_friction)
            [10]  score
            [11]  episode progress (steps / max_steps)
            [12]  level complete (0/1)
            [13]  player dead (0/1)
            [14]  dynamics type id (0-15, encodes vertical*4 + horizontal)
            [15]  vertical model type (0-3)

    Action space (Dict):
        'move_x': float in [-1, 1] - horizontal movement intent
        'jump':   int in {0, 1} - jump trigger

    Reward = weighted sum of raw signals (stored in info['reward_signals']):
        goal:     1.0 when goal reached
        progress: delta_x (rightward movement in pixels)
        death:    1.0 when player dies
        step:     1.0 every step
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        config: Optional[GameConfig] = None,
        render_mode: Optional[str] = None,
        obs_resolution: Tuple[int, int] = (256, 256),
        max_episode_steps: int = 1000,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        self.config = config or GameConfig()
        self.render_mode = render_mode
        self.obs_height, self.obs_width = obs_resolution
        self.max_episode_steps = max_episode_steps

        self.reward_weights = reward_weights or {
            "goal": 100.0,
            "progress": 1.0,
            "death": -50.0,
            "step": -0.1,
        }

        # Action space: hybrid continuous + discrete
        self.action_space = spaces.Dict({
            "move_x": spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32,
            ),
            "jump": spaces.Discrete(2),
        })

        # Observation space
        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(
                low=0, high=255,
                shape=(self.obs_height, self.obs_width, 3),
                dtype=np.uint8,
            ),
            "state": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(16,),
                dtype=np.float32,
            ),
        })

        # Initialize pygame (caller sets SDL_VIDEODRIVER for headless)
        if not pygame.get_init():
            pygame.init()

        # Offscreen render surface (native resolution)
        self._surface = pygame.Surface(
            (self.config.screen_width, self.config.screen_height)
        )

        # Display for human render mode
        self._display = None
        if render_mode == "human":
            self._display = pygame.display.set_mode(
                (self.config.screen_width, self.config.screen_height)
            )
            pygame.display.set_caption("PlatformerEnv")

        # Game state (populated on reset)
        self._physics: Optional[PhysicsWorld] = None
        self._player: Optional[Player] = None
        self._platforms = []
        self._goals = []
        self._hazards = []
        self._timed_hazards = []
        self._flashing_zones = []
        self._springs = []
        self._collectibles = []
        self._score = 0
        self._episode_steps = 0
        self._level_complete = False
        self._player_dead = False
        self._prev_player_x = 0.0
        self._camera_x = 0.0
        self._dynamics_model: Optional[DynamicsModel] = None
        self._behavioral_profile: Optional[BehavioralProfile] = None
        self._level_spec: Optional[LevelSpec] = None
        self._level_seed: int = 0

        self._dt = 1.0 / self.config.fps

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Clean up stale collectible registry entries
        Collectible._registry.clear()

        # Create dynamics model from config (needed for calibration before level gen)
        v_name = self.config.dynamics.vertical_model
        h_name = self.config.dynamics.horizontal_model
        vertical = VerticalModel[v_name.upper()]
        horizontal = HorizontalModel[h_name.upper()]
        self._dynamics_model = create_dynamics(
            vertical=vertical,
            horizontal=horizontal,
            physics_config=self.config.physics,
        )

        # Run behavioral calibration to get measured reachability
        self._behavioral_profile = calibrate(
            self.config.physics, self._dynamics_model
        )

        # Fresh physics world
        physics_params = PhysicsParams(gravity=self.config.physics.gravity)
        self._physics = PhysicsWorld(physics_params)

        # Generate level using measured behavioral references
        generator = LevelGenerator.from_config(
            self.config, behavioral_profile=self._behavioral_profile
        )
        level_seed = int(self.np_random.integers(0, 2**31))
        self._level_spec = spec = generator.generate(seed=level_seed)
        self._level_seed = level_seed

        # Build entities
        (self._platforms, self._goals, self._hazards,
         self._timed_hazards, self._flashing_zones,
         self._springs, self._collectibles) = build_level(self._physics, spec)

        # Create player
        self._player = Player(
            self._physics,
            spec.player_start,
            physics_config=self.config.physics,
            dynamics_model=self._dynamics_model,
        )

        # Settle loop: player needs to land on ground before episode starts.
        # Multiple steps needed because floaty/cubic have zero gravity at airtime=0;
        # player.update() increments airtime so gravity builds up.
        for _ in range(30):
            self._player.update(self._dt)
            self._physics.step(self._dt)
            if self._player.is_grounded:
                break
        self._player._airtime = 0.0

        # Reset episode state
        self._score = 0
        self._episode_steps = 0
        self._level_complete = False
        self._player_dead = False
        self._prev_player_x = spec.player_start[0]
        self._camera_x = 0.0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        assert self._player is not None, "Must call reset() before step()"

        # Apply action
        self._apply_action(action)

        # Update game state
        self._update(self._dt)
        self._episode_steps += 1

        # Compute reward
        reward_signals = self._compute_rewards()
        reward = sum(
            self.reward_weights.get(k, 0.0) * v
            for k, v in reward_signals.items()
        )

        terminated = self._level_complete or self._player_dead
        truncated = self._episode_steps >= self.max_episode_steps

        obs = self._get_obs()
        info = self._get_info()
        info["reward_signals"] = reward_signals

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Action handling
    # ------------------------------------------------------------------

    def _apply_action(self, action):
        if self._level_complete or self._player_dead:
            return

        move_x = action["move_x"]
        if isinstance(move_x, np.ndarray):
            move_x = float(move_x.item())
        move_x = float(move_x)

        jump = action["jump"]
        if isinstance(jump, np.ndarray):
            jump = int(jump.item())
        jump = int(jump)

        # Continuous horizontal force
        if abs(move_x) > 0.01:
            self._player._apply_horizontal_force(move_x)

        if jump:
            self._player.jump()

    # ------------------------------------------------------------------
    # Game state update (mirrors engine.update)
    # ------------------------------------------------------------------

    def _update(self, dt):
        if self._player:
            self._player.update()

        for hazard in self._hazards:
            hazard.update(dt)
        for th in self._timed_hazards:
            th.update(dt)
        for fz in self._flashing_zones:
            fz.update(dt)
        for spring in self._springs:
            spring.reset_trigger()

        self._physics.step(dt)

        # Collectibles
        for c in self._collectibles:
            if c.collected:
                self._score += c.value
                self._physics.remove_shape(c.shape)
        self._collectibles = [c for c in self._collectibles if not c.collected]

        # Win / lose checks
        for goal in self._goals:
            if goal.reached:
                self._level_complete = True

        for hazard in self._hazards:
            if hazard.triggered:
                self._player_dead = True
        for th in self._timed_hazards:
            if th.triggered:
                self._player_dead = True
        for fz in self._flashing_zones:
            if fz.triggered:
                self._player_dead = True

        if self._player:
            _, y = self._player.position
            if y < -100:
                self._player_dead = True

        self._update_camera()

    def _update_camera(self):
        if not self._player:
            return
        px, _ = self._player.position
        target_x = px - self.config.screen_width * 0.3
        self._camera_x += (target_x - self._camera_x) * 0.1
        self._camera_x = max(0, self._camera_x)

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_rewards(self):
        signals = {}

        if self._player:
            px, _ = self._player.position
            signals["progress"] = px - self._prev_player_x
            self._prev_player_x = px
        else:
            signals["progress"] = 0.0

        signals["goal"] = 1.0 if self._level_complete else 0.0
        signals["death"] = 1.0 if self._player_dead else 0.0
        signals["step"] = 1.0

        return signals

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_obs(self):
        # Only render RGB when someone will actually use it.
        # During RL training, StateOnlyWrapper strips RGB immediately,
        # so rendering 256x256 every step is pure waste (~50% of step cost).
        # render_mode="rgb_array" signals that the caller wants frames.
        if self.render_mode in ("rgb_array", "human"):
            rgb = self._render_frame()
        else:
            rgb = np.zeros(
                (self.obs_height, self.obs_width, 3), dtype=np.uint8
            )
        state = self._get_state_vector()
        return {"rgb": rgb, "state": state}

    def _get_state_vector(self):
        state = np.zeros(16, dtype=np.float32)

        if self._player:
            px, py = self._player.position
            vx, vy = self._player.velocity
            state[0] = px
            state[1] = py
            state[2] = vx
            state[3] = vy
            state[4] = float(self._player.is_grounded)

        p = self.config.physics
        state[5] = p.jump_height
        state[6] = p.jump_duration
        state[7] = p.move_speed
        state[8] = p.air_control
        state[9] = p.ground_friction

        state[10] = float(self._score)
        state[11] = float(self._episode_steps) / max(self.max_episode_steps, 1)
        state[12] = float(self._level_complete)
        state[13] = float(self._player_dead)

        # Dynamics model type encoding
        if self._dynamics_model:
            state[14] = float(self._dynamics_model.get_type_id())
            state[15] = float(list(VerticalModel).index(self._dynamics_model.vertical))

        return state

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _world_to_screen(self, x, y):
        screen_x = int(x - self._camera_x)
        screen_y = int(self.config.screen_height - y)
        return screen_x, screen_y

    def _render_frame(self):
        """Render current state to numpy array (H, W, 3) uint8."""
        self._surface.fill(_COLOR_BG)

        # Platforms
        for plat in self._platforms:
            left, bottom, right, top = plat.bounds
            sx, sy = self._world_to_screen(left, top)
            w, h = int(right - left), int(top - bottom)
            pygame.draw.rect(self._surface, _COLOR_PLATFORM, (sx, sy, w, h))

        # Goals
        for goal in self._goals:
            left, bottom, right, top = goal.bounds
            sx, sy = self._world_to_screen(left, top)
            w, h = int(right - left), int(top - bottom)
            pygame.draw.rect(self._surface, _COLOR_GOAL, (sx, sy, w, h))

        # Static hazards
        for hazard in self._hazards:
            left, bottom, right, top = hazard.bounds
            sx, sy = self._world_to_screen(left, top)
            w, h = int(right - left), int(top - bottom)
            pygame.draw.rect(self._surface, _COLOR_HAZARD, (sx, sy, w, h))

        # Timed hazards
        for th in self._timed_hazards:
            left, bottom, right, top = th.bounds
            sx, sy = self._world_to_screen(left, top)
            w, h = int(right - left), int(top - bottom)
            color = _COLOR_HAZARD if th.active else _COLOR_HAZARD_INACTIVE
            pygame.draw.rect(self._surface, color, (sx, sy, w, h))

        # Flashing zones
        for fz in self._flashing_zones:
            left, bottom, right, top = fz.bounds
            sx, sy = self._world_to_screen(left, top)
            w, h = int(right - left), int(top - bottom)
            color = _COLOR_FLASHING_SAFE if fz.safe else _COLOR_FLASHING_DEADLY
            pygame.draw.rect(self._surface, color, (sx, sy, w, h))

        # Springs
        for spring in self._springs:
            left, bottom, right, top = spring.bounds
            sx, sy = self._world_to_screen(left, top)
            w, h = int(right - left), int(top - bottom)
            pygame.draw.rect(self._surface, _COLOR_SPRING, (sx, sy, w, h))

        # Collectibles
        for c in self._collectibles:
            left, bottom, right, top = c.bounds
            sx, sy = self._world_to_screen(left, top)
            w, h = int(right - left), int(top - bottom)
            pygame.draw.rect(self._surface, _COLOR_COLLECTIBLE, (sx, sy, w, h))

        # Player
        if self._player:
            px, py = self._player.position
            half_w = self._player.config.width / 2
            half_h = self._player.config.height / 2
            sx, sy = self._world_to_screen(px - half_w, py + half_h)
            pygame.draw.rect(
                self._surface, _COLOR_PLAYER,
                (sx, sy, int(self._player.config.width), int(self._player.config.height)),
            )

        # Scale to observation resolution
        scaled = pygame.transform.scale(
            self._surface, (self.obs_width, self.obs_height)
        )
        # surfarray gives (W, H, 3); transpose to (H, W, 3)
        array = pygame.surfarray.array3d(scaled)
        return np.transpose(array, (1, 0, 2)).astype(np.uint8)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human" and self._display:
            # Render at native resolution to the display
            self._render_frame()  # updates self._surface
            self._display.blit(self._surface, (0, 0))
            # Draw HUD on display only (not in observation RGB)
            self._draw_hud()
            pygame.display.flip()

    def _draw_hud(self):
        """Draw score/time/status text on the display surface."""
        font = pygame.font.Font(None, 28)
        time_seconds = self._episode_steps / self.config.fps
        score_text = f"Score: {self._score}  |  Time: {time_seconds:.1f}s"
        score_surface = font.render(score_text, True, _COLOR_COLLECTIBLE)
        score_rect = score_surface.get_rect(
            topright=(self.config.screen_width - 10, 10)
        )
        self._display.blit(score_surface, score_rect)

        if self._level_complete:
            self._draw_centered_text("LEVEL COMPLETE!", (255, 255, 255))
        elif self._player_dead:
            self._draw_centered_text("GAME OVER!", _COLOR_HAZARD)

    def _draw_centered_text(self, text, color):
        font = pygame.font.Font(None, 48)
        surface = font.render(text, True, color)
        rect = surface.get_rect(
            center=(self.config.screen_width // 2, self.config.screen_height // 2)
        )
        self._display.blit(surface, rect)

    def _get_info(self):
        info = {
            "score": self._score,
            "episode_steps": self._episode_steps,
            "level_complete": self._level_complete,
            "player_dead": self._player_dead,
        }
        if self._player:
            info["player_position"] = self._player.position
        if self._dynamics_model:
            info["dynamics_type"] = self._dynamics_model.get_metadata()
        if self._behavioral_profile:
            info["behavioral_profile"] = self._behavioral_profile.to_dict()
        if self._level_spec:
            info["level_geometry"] = {
                "platforms": self._level_spec.platforms,
                "collectibles": self._level_spec.collectibles,
                "hazards": self._level_spec.hazards,
                "goals": self._level_spec.goals,
            }
        info["level_seed"] = self._level_seed
        return info

    def close(self):
        if self._display:
            pygame.display.quit()
            self._display = None
