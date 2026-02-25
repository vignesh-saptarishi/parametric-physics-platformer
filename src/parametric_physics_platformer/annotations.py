"""Annotation collector for human playtesting sessions.

Records end-of-level annotations (physics feel, level rating) and
auto-logs deaths with position, timestep, and cause.
Saves all data to a JSON file for later analysis.
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List


class AnnotationCollector:
    """Collects human annotations during play sessions.

    Tracks per-episode:
    - Death events (position, timestep, cause) — auto-logged
    - Physics feel (good/bad) — from user prompt
    - Level rating (1-5) — from user prompt
    - Physics config used
    - Episode outcome (goal/death/quit)

    Usage:
        collector = AnnotationCollector(output_path="data/annotations.json")
        collector.begin_episode(physics_config={...})

        # Auto-log a death
        collector.log_death(position=(300, 50), timestep=145, cause="hazard")

        # Record end-of-level annotation
        collector.annotate(physics_feel="good", level_rating=4)

        collector.end_episode(outcome="goal", score=3)
        collector.save()
    """

    def __init__(self, output_path: str = "data/annotations.json"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self._episodes: List[Dict[str, Any]] = []
        self._current: Optional[Dict[str, Any]] = None

        # Load existing annotations if file exists
        if self.output_path.exists():
            with open(self.output_path) as f:
                data = json.load(f)
                self._episodes = data.get("episodes", [])

    def begin_episode(self, physics_config: Optional[Dict[str, Any]] = None):
        """Start tracking a new episode."""
        self._current = {
            "episode_index": len(self._episodes),
            "timestamp": time.time(),
            "physics_config": physics_config or {},
            "deaths": [],
            "physics_feel": None,  # "good" or "bad"
            "layout_playable": None,  # "yes" or "no"
            "level_rating": None,  # 1-5
            "outcome": None,       # "goal", "death", "quit"
            "score": 0,
            "steps": 0,
        }

    def log_death(
        self,
        position: tuple,
        timestep: int,
        cause: Optional[str] = None,
    ):
        """Auto-log a death event.

        Args:
            position: (x, y) world position at death.
            timestep: Episode step count at death.
            cause: What killed the player (hazard, timed_hazard, flashing_zone, fall).
        """
        if self._current is None:
            return

        self._current["deaths"].append({
            "position": list(position),
            "timestep": timestep,
            "cause": cause or "unknown",
        })

    def annotate(
        self,
        physics_feel: Optional[str] = None,
        layout_playable: Optional[str] = None,
        level_rating: Optional[int] = None,
    ):
        """Record end-of-level annotation from user prompt.

        Args:
            physics_feel: "good" or "bad" (or None if skipped).
            layout_playable: "yes" or "no" (or None if skipped).
            level_rating: 1-5 (or None if skipped).
        """
        if self._current is None:
            return

        if physics_feel is not None:
            self._current["physics_feel"] = physics_feel
        if layout_playable is not None:
            self._current["layout_playable"] = layout_playable
        if level_rating is not None:
            self._current["level_rating"] = level_rating

    def end_episode(
        self,
        outcome: Optional[str] = None,
        score: int = 0,
        steps: int = 0,
    ):
        """Finish the current episode and add to collection.

        Args:
            outcome: "goal", "death", or "quit".
            score: Final score.
            steps: Total episode steps.
        """
        if self._current is None:
            return

        self._current["outcome"] = outcome
        self._current["score"] = score
        self._current["steps"] = steps
        self._episodes.append(self._current)
        self._current = None

    def save(self):
        """Write all annotations to JSON file."""
        data = {
            "version": 1,
            "session_time": time.time(),
            "total_episodes": len(self._episodes),
            "episodes": self._episodes,
        }
        with open(self.output_path, "w") as f:
            json.dump(data, f, indent=2)

    @property
    def episode_count(self) -> int:
        return len(self._episodes)

    @property
    def recording(self) -> bool:
        return self._current is not None

    def summary(self) -> Dict[str, Any]:
        """Return summary stats of collected annotations."""
        if not self._episodes:
            return {"episodes": 0}

        feels = [e["physics_feel"] for e in self._episodes if e["physics_feel"]]
        layouts = [e.get("layout_playable") for e in self._episodes if e.get("layout_playable")]
        ratings = [e["level_rating"] for e in self._episodes if e["level_rating"]]
        outcomes = [e["outcome"] for e in self._episodes if e["outcome"]]
        total_deaths = sum(len(e["deaths"]) for e in self._episodes)

        return {
            "episodes": len(self._episodes),
            "annotated": len(feels),
            "skipped": len(self._episodes) - len(feels),
            "good_feel": feels.count("good"),
            "bad_feel": feels.count("bad"),
            "layout_playable": layouts.count("yes"),
            "layout_unplayable": layouts.count("no"),
            "avg_rating": sum(ratings) / len(ratings) if ratings else None,
            "goals": outcomes.count("goal"),
            "deaths_episodes": outcomes.count("death"),
            "total_death_events": total_deaths,
        }
