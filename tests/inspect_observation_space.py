import asyncio
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from roar_py_carla.sensors import carla_lidar_sensor

_lidar_converter = getattr(carla_lidar_sensor, "__convert_carla_lidar_raw_to_roar_py", None)
if _lidar_converter is not None and not hasattr(
    carla_lidar_sensor, "_RoarPyCarlaLiDARSensor__convert_carla_lidar_raw_to_roar_py"
):
    setattr(
        carla_lidar_sensor,
        "_RoarPyCarlaLiDARSensor__convert_carla_lidar_raw_to_roar_py",
        _lidar_converter,
    )

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.env_util import initialize_roar_env

RUN_FPS = 25
SUBSTEPS_PER_STEP = 5


def describe_space(space: gym.Space, indent: int = 0) -> None:
    prefix = " " * indent
    if isinstance(space, gym.spaces.Dict):
        print(f"{prefix}Dict({len(space.spaces)} keys)")
        for key, subspace in space.spaces.items():
            print(f"{prefix}- {key}:")
            describe_space(subspace, indent + 2)
        return

    if hasattr(space, "shape"):
        dtype = getattr(space, "dtype", None)
        print(f"{prefix}{space.__class__.__name__} shape={space.shape} dtype={dtype}")
    else:
        print(f"{prefix}{space.__class__.__name__}")


def describe_observation(obs, indent: int = 0) -> None:
    prefix = " " * indent
    if isinstance(obs, dict):
        for key, value in obs.items():
            print(f"{prefix}- {key}:")
            describe_observation(value, indent + 2)
        return

    if isinstance(obs, np.ndarray):
        summary = f"shape={obs.shape} dtype={obs.dtype}"
        if obs.size <= 12:
            print(f"{prefix}{summary} values={np.array2string(obs, precision=3)}")
        else:
            print(
                f"{prefix}{summary} min={obs.min():.3f} max={obs.max():.3f} mean={obs.mean():.3f}"
            )
        return

    print(f"{prefix}{type(obs).__name__}: {obs}")


def main() -> None:
    np.set_printoptions(precision=3, suppress=True)
    control_timestep = 1.0 / RUN_FPS
    physics_timestep = control_timestep / SUBSTEPS_PER_STEP

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        env = loop.run_until_complete(
            initialize_roar_env(
                control_timestep=control_timestep,
                physics_timestep=physics_timestep,
                enable_rendering=False,
            )
        )
        obs, info = env.reset()

        print("Observation space (structured):")
        describe_space(env.observation_space)
        print("\nObservation (structured):")
        describe_observation(obs)

        flat_space = gym.spaces.flatten_space(env.observation_space)
        flat_obs = gym.spaces.flatten(env.observation_space, obs)
        print("\nObservation space (flattened):")
        describe_space(flat_space)
        print(
            f"\nObservation (flattened): shape={flat_obs.shape} dtype={flat_obs.dtype} "
            f"sample={flat_obs[: min(10, flat_obs.size)]}"
        )
    finally:
        loop.close()


if __name__ == "__main__":
    main()
