# Reinforcement Learning for Racing in CARLA (Monza)

This repository is a fork of the [ROAR_PY_RL](https://github.com/augcog/ROAR_PY_RL) RL code provided by Berkeley and is focused on training an RL policy to drive the Monza track (walls on both sides) as fast as possible without collisions, using CARLA via the ROAR_PY interface.

# Performance
![](./turning_1.gif)
![](./turning_2.gif)
![](./turning_3.gif)

[Click here for a longer video (7 mins)](https://www.youtube.com/watch?v=VjO94m9O1PA)

## Changes Made
- Custom spawn points with a variety of locations before, during, after corners, as well as varying distances to walls to improve robustness.
- 360-degree LiDAR wrapper converted to beams normalized to [0, 1].
- Observation filtering + normalization + previous action (111-dim flat).
- Added reward components (progress, step penalty, collision, stalling, reverse, steering deadzone, heading penalty, speed x heading penalty).
- Parallel SB3 SAC training (SubprocVecEnv) with W&B logging and optional video.
- Rendering toggle: CARLA `no_rendering_mode` for fast training; camera only when enabled.
- Spawn point control via `spawn_point_index` / `EVAL_SPAWN_POINT_INDEX`.

## Setup
1. Follow the ROAR setup tutorial: https://roar.gitbook.io/roar_py_rl-documentation/installation
2. Use my ROAR_PY fork for custom spawn points:
   https://github.com/simnlee/roar_py
3. Download the latest maps from ROAR:
   https://roar.berkeley.edu/berkeley-major-map/
4. Copy `example.env` to `.env`, then set at least:
   - `CARLA_EXE` (path to your CARLA executable)
   - `PROJECT_NAME`, `RUN_NAME` (W&B)

## Configuration (.env)
Key knobs used by `training/train_online.py` and `training/eval_agent.py`:
- `N_ENVS`, `BASE_PORT`, `PORT_STRIDE`: parallel CARLA instances/ports.
- `RUN_FPS`, `SUBSTEPS_PER_STEP`, `TIME_LIMIT`: control + physics rates and episode length.
- `ENABLE_RENDERING`: `false` for training speed, `true` for videos/manual debug.
- `RESUME_CHECKPOINT`: resume training from a model path.
- Reward params: `PROGRESS_SCALE`, `STEP_PENALTY`, `COLLISION_*`, `STALL_*`,
  `REVERSE_PENALTY`, `STEERING_*`, `HEADING_*`, `SPEED_HEADING_PENALTY_SCALE`.
- Observation: `NUM_LIDAR_BEAMS`, `LIDAR_MAX_DISTANCE`.
- Eval: `EVAL_SPAWN_POINT_INDEX`.

See `example.env` for the full list and defaults.

## Environment details

### Observation space (active)
Observation is filtered to these keys in `training/env_util.py`:
- `gyroscope` (3)
- `accelerometer` (3, converted to local frame)
- `local_velocimeter` (3)
- `waypoints_information` (10 x 4 = 40)
- `lidar` (60)
- `previous_action` (2)

All values are normalized in `NormalizeObservationWrapper`. When flattened for
SB3, the total observation size is 111.

### Action space (active)
Only throttle + steer are exposed to the agent:
- `throttle`: Box(-1, 1) (negative values become brake)
- `steer`: Box(-1, 1)

The wrapper maps this to full CARLA controls in `SimplifyCarlaActionFilter`.

### Reward and termination
Reward lives in `roar_py_rl/environment/sim_env.py`:
- Continuous progress reward (`progress_scale * delta_distance`)
- Step penalty ("hot water")
- Collision penalty (applied before termination)
- Stalling penalty
- Reverse penalty
- Steering deadzone reward
- Optional heading and speed x heading penalties

Episodes terminate on collision when impulse exceeds `COLLISION_THRESHOLD`. All reward/termination parameters are configured via `.env`.

### Rendering
`enable_rendering` toggles CARLA `no_rendering_mode`. Camera sensors are only attached when rendering is enabled (for training speed).

### Spawn points
`initialize_roar_env` spawns at `spawn_points[1]` by default. If
`spawn_point_index` is provided, both the initial spawn and resets use that fixed point. Otherwise, resets are randomized.

## Train
From repo root:
```bash
cd training
python train_online.py
```

Artifacts (relative to `training/`):
- Tensorboard: `runs/`
- Models: `models/`
- Monitor logs: `logs/`
- Videos (if enabled): `videos/`

## Eval
```bash
python training/eval_agent.py
```
Set `EVAL_SPAWN_POINT_INDEX` and keep reward/observation parameters identical to training for best results.

## Manual play and debugging
```bash
python training/start_play.py
python tests/inspect_observation_space.py
python tests/manual_control_lidar.py
```

## LiDAR note
CARLA LiDAR conversion requires a name-mangling patch. Keep the helper in any script that uses LiDAR (see `training/train_online.py`, `training/eval_agent.py`, and `tests/inspect_observation_space.py`).

## References
- https://arxiv.org/abs/1801.01290
- https://arxiv.org/abs/1910.07207
- https://arxiv.org/pdf/1812.05905.pdf
- https://arxiv.org/abs/1707.06347
- https://arxiv.org/abs/1312.5602
- https://arxiv.org/abs/2201.12417
- https://arxiv.org/abs/2204.05186
- https://arxiv.org/abs/2212.04407
- https://arxiv.org/abs/2106.01345
- https://arxiv.org/abs/2309.04459
- https://www.nature.com/articles/s41586-023-06419-4
