"""Tests for annotation collector."""

import os
import json
import tempfile
import numpy as np
import pytest

os.environ['SDL_VIDEODRIVER'] = 'dummy'

from parametric_physics_platformer.annotations import AnnotationCollector
from parametric_physics_platformer.engine import PlatformerEngine
from parametric_physics_platformer.config import GameConfig


class TestAnnotationCollector:
    def test_begin_and_end_episode(self, tmp_path):
        path = tmp_path / "annot.json"
        collector = AnnotationCollector(output_path=str(path))

        collector.begin_episode(physics_config={"jump_height": 100})
        assert collector.recording

        collector.end_episode(outcome="goal", score=3, steps=100)
        assert not collector.recording
        assert collector.episode_count == 1

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "annot.json"
        collector = AnnotationCollector(output_path=str(path))

        collector.begin_episode(physics_config={"jump_height": 100})
        collector.annotate(physics_feel="good", level_rating=4)
        collector.end_episode(outcome="goal", score=5, steps=200)
        collector.save()

        # Read back
        with open(path) as f:
            data = json.load(f)
        assert data["total_episodes"] == 1
        ep = data["episodes"][0]
        assert ep["physics_feel"] == "good"
        assert ep["level_rating"] == 4
        assert ep["outcome"] == "goal"
        assert ep["score"] == 5
        assert ep["steps"] == 200

    def test_log_death(self, tmp_path):
        path = tmp_path / "annot.json"
        collector = AnnotationCollector(output_path=str(path))

        collector.begin_episode()
        collector.log_death(position=(300.0, 50.0), timestep=145, cause="hazard")
        collector.log_death(position=(500.0, -100.0), timestep=200, cause="fall")
        collector.end_episode(outcome="death")
        collector.save()

        with open(path) as f:
            data = json.load(f)
        deaths = data["episodes"][0]["deaths"]
        assert len(deaths) == 2
        assert deaths[0]["cause"] == "hazard"
        assert deaths[0]["position"] == [300.0, 50.0]
        assert deaths[0]["timestep"] == 145
        assert deaths[1]["cause"] == "fall"

    def test_skip_annotation(self, tmp_path):
        path = tmp_path / "annot.json"
        collector = AnnotationCollector(output_path=str(path))

        collector.begin_episode()
        # Don't annotate — skipped
        collector.end_episode(outcome="death")
        collector.save()

        with open(path) as f:
            data = json.load(f)
        ep = data["episodes"][0]
        assert ep["physics_feel"] is None
        assert ep["level_rating"] is None

    def test_multiple_episodes(self, tmp_path):
        path = tmp_path / "annot.json"
        collector = AnnotationCollector(output_path=str(path))

        for i in range(3):
            collector.begin_episode()
            collector.annotate(physics_feel="good" if i % 2 == 0 else "bad")
            collector.end_episode(outcome="goal")

        assert collector.episode_count == 3
        collector.save()

        with open(path) as f:
            data = json.load(f)
        assert data["total_episodes"] == 3

    def test_loads_existing_file(self, tmp_path):
        path = tmp_path / "annot.json"
        # First session
        c1 = AnnotationCollector(output_path=str(path))
        c1.begin_episode()
        c1.end_episode(outcome="goal")
        c1.save()

        # Second session loads existing
        c2 = AnnotationCollector(output_path=str(path))
        assert c2.episode_count == 1
        c2.begin_episode()
        c2.end_episode(outcome="death")
        c2.save()

        with open(path) as f:
            data = json.load(f)
        assert data["total_episodes"] == 2

    def test_summary(self, tmp_path):
        path = tmp_path / "annot.json"
        collector = AnnotationCollector(output_path=str(path))

        collector.begin_episode()
        collector.annotate(physics_feel="good", level_rating=4)
        collector.log_death(position=(100, 50), timestep=50, cause="hazard")
        collector.end_episode(outcome="death")

        collector.begin_episode()
        collector.annotate(physics_feel="bad", level_rating=2)
        collector.end_episode(outcome="goal")

        summary = collector.summary()
        assert summary["episodes"] == 2
        assert summary["annotated"] == 2
        assert summary["good_feel"] == 1
        assert summary["bad_feel"] == 1
        assert summary["avg_rating"] == pytest.approx(3.0)
        assert summary["total_death_events"] == 1

    def test_no_recording_ignores_calls(self, tmp_path):
        path = tmp_path / "annot.json"
        collector = AnnotationCollector(output_path=str(path))

        # These should not crash when not recording
        collector.log_death(position=(0, 0), timestep=0, cause="fall")
        collector.annotate(physics_feel="good")
        result = collector.end_episode()
        assert result is None


class TestEngineDeathCause:
    def test_death_cause_initially_none(self):
        engine = PlatformerEngine()
        assert engine.death_cause is None

    def test_death_cause_reset_on_reset(self):
        engine = PlatformerEngine()
        engine.load_test_level()
        engine.death_cause = "hazard"
        engine.player_dead = True
        engine.reset()
        assert engine.death_cause is None

    def test_fall_sets_death_cause(self):
        engine = PlatformerEngine()
        engine.load_test_level()
        # Move player way below screen
        engine.player.body.position = (100, -200)
        engine.update(1.0 / 60)
        assert engine.player_dead is True
        assert engine.death_cause == "fall"
