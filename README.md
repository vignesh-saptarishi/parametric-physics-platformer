# parametric-physics-platformer

A 2D platformer with swappable physics equations, built as a Gymnasium environment for RL and world model research. Vary both continuous physics parameters (gravity, jump height, move speed, friction) and the dynamics equations themselves (4 vertical x 4 horizontal = 16 combinations) per episode.

**Status:** Experimental. The core game engine, dynamics models, calibration system, and Gymnasium wrapper are fully implemented and tested (~270 tests). The missing piece is the action-to-force translation layer — currently actions map to velocity/impulse directly rather than through a force-based interface. This will likely be addressed in future work. The environment is usable as-is for research on physics generalization and domain randomization.

## What makes this different

Standard platformer environments have fixed physics. This one lets you change the equations of motion themselves, not just tune parameters:

**4 vertical models** (how jumps work):
- **Parabolic** — constant gravity, standard `y ~ t^2` arcs
- **Cubic** — gravity increases with time, `y ~ t^3` trajectories
- **Floaty** — `tanh`-shaped gravity, float-then-snap behavior
- **Asymmetric** — different gravity for rising vs falling

**4 horizontal models** (how movement works):
- **Force** — input maps to acceleration (inertia-based)
- **Velocity** — input maps directly to speed (instant response)
- **Impulse** — burst-based movement (tap to dash)
- **Drag** — acceleration with speed-dependent drag cap

That's 16 qualitatively distinct physics "worlds". On top of that, continuous parameters (jump height, move speed, air control, ground friction) vary within each dynamics type.

## Physics parameters

| Parameter | Default | Range | What it controls |
|---|---|---|---|
| `jump_height` | 120.0 | [60, 200] | Peak height of a jump (pixels) |
| `jump_duration` | 0.4 | [0.2, 0.8] | Time to reach apex (seconds) |
| `move_speed` | 200.0 | [100, 400] | Horizontal movement speed (px/s) |
| `air_control` | 0.3 | [0.0, 1.0] | Aerial steering authority (0=none, 1=full) |
| `ground_friction` | 0.8 | [0.0, 1.0] | Ground deceleration factor |

Under parabolic dynamics, these map directly to behavioral outcomes. Under other dynamics models, actual behavior is measured by the calibration system.

## Key components

- **PlatformerEngine** — pygame-based game loop with rendering, collision handling, entity management
- **PlatformerEnv** — Gymnasium wrapper with Dict observation (RGB + 16D state vector) and hybrid action space
- **DynamicsModel** — pluggable equations of motion (4 vertical x 4 horizontal)
- **Calibration** — headless side-simulation measuring actual jump apex, max speed, reach for any config + dynamics combo
- **LevelGenerator** — procedural platform placement using measured behavioral profiles
- **RL wrappers** — domain randomization, state-only projection, dynamics-blind, continuous action, history stacking
- **Constraint system** — validates that configs produce playable levels
- **Scripted policies** — random, rush, cautious, explorer for automated data collection
- **Trajectory collector** — records episodes to `.npz` for offline training

## Structure

```
src/parametric_physics_platformer/
├── config.py          # PhysicsConfig, GameConfig, LayoutConfig, DynamicsConfig
├── physics.py         # pymunk world setup, collision types
├── dynamics.py        # 4x4 dynamics model grid (equations of motion)
├── entities.py        # Player, Platform, Goal, Hazard, Spring, Collectible
├── engine.py          # pygame game loop + rendering
├── gym_env.py         # Gymnasium environment wrapper
├── level_gen.py       # Procedural level generation
├── calibration.py     # Behavioral profiling (measure actual physics outcomes)
├── rl_wrappers.py     # Domain randomization, state projection, factory
├── constraints.py     # Parameter validation + constrained sampling
├── data_collector.py  # Trajectory recording to .npz
├── policies.py        # Scripted policies for data collection
├── annotations.py     # Human playtesting annotation collector
└── analysis/          # Trajectory metrics + diversity analysis
```

## License

MIT
