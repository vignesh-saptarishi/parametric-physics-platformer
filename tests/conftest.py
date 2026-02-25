"""Pytest configuration and shared fixtures."""

import os

# Ensure headless pygame for all tests
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import pytest

from parametric_physics_platformer.physics import PhysicsWorld
from parametric_physics_platformer.config import GameConfig


@pytest.fixture
def physics():
    """Fresh physics world for each test."""
    return PhysicsWorld()


@pytest.fixture
def game_config():
    """Default game configuration."""
    return GameConfig()
