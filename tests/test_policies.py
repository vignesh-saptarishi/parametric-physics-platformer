"""Tests for scripted policies."""

import os
import numpy as np
import pytest

os.environ['SDL_VIDEODRIVER'] = 'dummy'

from parametric_physics_platformer.policies import RandomPolicy, RushPolicy, CautiousPolicy, ExplorerPolicy, POLICIES
from parametric_physics_platformer.gym_env import PlatformerEnv


@pytest.fixture
def env():
    e = PlatformerEnv(max_episode_steps=100)
    yield e
    e.close()


@pytest.fixture
def obs(env):
    o, _ = env.reset(seed=42)
    return o


class TestRandomPolicy:
    def test_returns_valid_action(self, obs):
        policy = RandomPolicy(rng=np.random.default_rng(0))
        action = policy(obs)
        assert "move_x" in action
        assert "jump" in action
        assert -1.0 <= action["move_x"][0] <= 1.0
        assert action["jump"] in (0, 1)

    def test_varies_actions(self, obs):
        policy = RandomPolicy(rng=np.random.default_rng(0))
        actions = [policy(obs)["move_x"][0] for _ in range(20)]
        assert len(set(round(a, 3) for a in actions)) > 1


class TestRushPolicy:
    def test_always_moves_right(self, obs):
        policy = RushPolicy()
        for _ in range(10):
            action = policy(obs)
            assert action["move_x"][0] == pytest.approx(1.0)

    def test_jumps_periodically(self, env):
        policy = RushPolicy(jump_interval=5)
        policy.reset()
        obs, _ = env.reset(seed=42)
        jumps = []
        for _ in range(50):
            action = policy(obs)
            jumps.append(action["jump"])
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        assert sum(jumps) > 0


class TestCautiousPolicy:
    def test_moves_slowly(self, obs):
        policy = CautiousPolicy()
        action = policy(obs)
        assert 0 < action["move_x"][0] <= 0.5

    def test_runs_without_crash(self, env):
        policy = CautiousPolicy()
        obs, _ = env.reset(seed=42)
        for _ in range(50):
            action = policy(obs)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break


class TestExplorerPolicy:
    def test_changes_direction(self, obs):
        policy = ExplorerPolicy(rng=np.random.default_rng(0))
        directions = []
        for _ in range(200):
            action = policy(obs)
            directions.append(action["move_x"][0])
        # Should have both positive and negative movement
        assert any(d > 0 for d in directions)
        assert any(d < 0 for d in directions)

    def test_runs_without_crash(self, env):
        policy = ExplorerPolicy(rng=np.random.default_rng(0))
        obs, _ = env.reset(seed=42)
        for _ in range(50):
            action = policy(obs)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break


class TestPoliciesRegistry:
    def test_all_policies_registered(self):
        assert "random" in POLICIES
        assert "rush" in POLICIES
        assert "cautious" in POLICIES
        assert "explorer" in POLICIES

    def test_all_policies_work_with_env(self, env):
        """Run each policy for 20 steps in the env."""
        for name, PolicyCls in POLICIES.items():
            if name == "random":
                policy = PolicyCls(rng=np.random.default_rng(0))
            elif name == "explorer":
                policy = PolicyCls(rng=np.random.default_rng(0))
            else:
                policy = PolicyCls()

            policy.reset()
            obs, _ = env.reset(seed=42)
            for _ in range(20):
                action = policy(obs)
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break
