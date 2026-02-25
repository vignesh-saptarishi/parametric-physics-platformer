"""parametric-physics-platformer — 2D platformer with swappable physics equations.

A Gymnasium-compatible platformer environment where both the physics parameters
(gravity, jump height, move speed, friction) and the dynamics equations themselves
(4 vertical models x 4 horizontal models = 16 combinations) can be varied per
episode. Built for domain randomization, physics generalization, and world model
research.
"""

from .config import PhysicsConfig, GameConfig, LayoutConfig, DynamicsConfig, ObjectiveConfig, CONFIGS
from .physics import PhysicsWorld, PhysicsParams
from .entities import Player, Platform, Goal, Hazard
from .engine import PlatformerEngine
from .level_gen import LevelGenerator, LevelSpec, build_level
from .constraints import ParameterConstraints, ConstrainedSampler, ConstraintResult, ConstraintViolation

__all__ = [
    "PhysicsConfig",
    "GameConfig",
    "LayoutConfig",
    "DynamicsConfig",
    "ObjectiveConfig",
    "CONFIGS",
    "PhysicsWorld",
    "PhysicsParams",
    "Player",
    "Platform",
    "Goal",
    "Hazard",
    "PlatformerEngine",
    "LevelGenerator",
    "LevelSpec",
    "build_level",
    "ParameterConstraints",
    "ConstrainedSampler",
    "ConstraintResult",
    "ConstraintViolation",
]
