import gymnasium as gym
import os
import asyncio
from pathlib import Path
from typing import Optional, Dict, Callable

import numpy as np
import torch as th
import wandb
import nest_asyncio
from dotenv import load_dotenv, find_dotenv
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EveryNTimesteps,
    CallbackList,
)
from stable_baselines3.common.utils import set_random_seed
from wandb.integration.sb3 import WandbCallback

from roar_py_carla.sensors import carla_lidar_sensor
from env_util import initialize_roar_env
from roar_py_rl_carla import FlattenActionWrapper

# Load environment variables from .env file (walk up from CWD)
load_dotenv(find_dotenv(usecwd=True))

def _patch_carla_lidar_name_mangling() -> None:
    converter = getattr(carla_lidar_sensor, "__convert_carla_lidar_raw_to_roar_py", None)
    if converter is None:
        return
    mangled_name = "_RoarPyCarlaLiDARSensor__convert_carla_lidar_raw_to_roar_py"
    if not hasattr(carla_lidar_sensor, mangled_name):
        setattr(carla_lidar_sensor, mangled_name, converter)

_patch_carla_lidar_name_mangling()

class NaNCheckWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, name: str) -> None:
        super().__init__(env)
        self._name = name
        self._episode = 0
        self._step = 0
        self._skipped_keys = set()

    def _check_array(
        self,
        value,
        label: str,
        where: str,
        key: Optional[str] = None,
        allow_non_numeric: bool = False,
    ) -> None:
        arr = np.asarray(value)
        if not np.issubdtype(arr.dtype, np.number):
            if allow_non_numeric:
                if key is not None and key not in self._skipped_keys:
                    print(
                        f"[NaNCheck:{self._name}] skip non-numeric {label} key={key} "
                        f"(episode={self._episode}, step={self._step})"
                    )
                    self._skipped_keys.add(key)
                return
            key_msg = f" key={key}" if key is not None else ""
            raise ValueError(
                f"[NaNCheck:{self._name}] non-numeric {label} in {where} "
                f"(episode={self._episode}, step={self._step}){key_msg}"
            )
        if not np.all(np.isfinite(arr)):
            key_msg = f" key={key}" if key is not None else ""
            raise ValueError(
                f"[NaNCheck:{self._name}] non-finite {label} in {where} "
                f"(episode={self._episode}, step={self._step}){key_msg}"
            )

    def _check_obs(self, obs, where: str) -> None:
        if isinstance(obs, dict):
            for key, value in obs.items():
                if value is None:
                    continue
                self._check_array(
                    value,
                    "observation",
                    where,
                    key=key,
                    allow_non_numeric=True,
                )
            return
        self._check_array(obs, "observation", where)

    def _check_action(self, action, where: str) -> None:
        if isinstance(action, dict):
            for key, value in action.items():
                if value is None:
                    continue
                self._check_array(value, "action", where, key=key)
            return
        self._check_array(action, "action", where)

    def reset(self, **kwargs):
        self._episode += 1
        self._step = 0
        obs, info = self.env.reset(**kwargs)
        self._check_obs(obs, "reset")
        return obs, info

    def step(self, action):
        self._check_action(action, "step")
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step += 1
        self._check_obs(obs, "step")
        if not np.isfinite(reward):
            raise ValueError(
                f"[NaNCheck:{self._name}] non-finite reward in step "
                f"(episode={self._episode}, step={self._step}): {reward}"
            )
        return obs, reward, terminated, truncated, info


class MinSpeedPenaltyLogger(BaseCallback):
    def __init__(
        self,
        info_key: str = "reward_min_speed_penalty",
        log_key: str = "rollout/ep_speed_pen_mean",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.info_key = info_key
        self.log_key = log_key
        self._episode_sums = None
        self._episode_lengths = None

    def _init_callback(self) -> None:
        n_envs = self.training_env.num_envs
        self._episode_sums = np.zeros(n_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(n_envs, dtype=np.int32)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones")
        if dones is None:
            return True
        for idx, info in enumerate(infos):
            penalty = info.get(self.info_key, 0.0)
            self._episode_sums[idx] += float(penalty)
            self._episode_lengths[idx] += 1
        if np.any(dones):
            done_indices = np.where(dones)[0]
            means = []
            for idx in done_indices:
                length = max(int(self._episode_lengths[idx]), 1)
                means.append(float(self._episode_sums[idx]) / length)
                self._episode_sums[idx] = 0.0
                self._episode_lengths[idx] = 0
            if means:
                self.logger.record(self.log_key, float(np.mean(means)))
        return True

RUN_FPS = int(os.getenv("RUN_FPS", "25"))
SUBSTEPS_PER_STEP = int(os.getenv("SUBSTEPS_PER_STEP", "5"))
MODEL_SAVE_FREQ = int(os.getenv("MODEL_SAVE_FREQ", "50000"))
TIME_LIMIT = int(os.getenv("TIME_LIMIT", str(RUN_FPS * 2 * 60)))
PROJECT_NAME = os.getenv("PROJECT_NAME", "CARLA_RL_parallel")
RUN_NAME = os.getenv("RUN_NAME", "Run4")
ENABLE_RENDERING = os.getenv("ENABLE_RENDERING", "false") == "true"
VIDEO_SAVE_FREQ = int(os.getenv("VIDEO_SAVE_FREQ", "20000"))

# Parallel training config (loaded from .env)
N_ENVS = int(os.getenv("N_ENVS", "2"))
BASE_PORT = int(os.getenv("BASE_PORT", "2000"))
PORT_STRIDE = int(os.getenv("PORT_STRIDE", "2"))
SEED = int(os.getenv("SEED", "1"))
RESUME_CHECKPOINT = os.getenv("RESUME_CHECKPOINT", "").strip()

# Reward configuration
PROGRESS_SCALE = float(os.getenv("PROGRESS_SCALE", "1.0"))
TIME_PENALTY = float(os.getenv("TIME_PENALTY", "0.1"))
SPEED_BONUS_SCALE = float(os.getenv("SPEED_BONUS_SCALE", "0.0"))
COLLISION_THRESHOLD = float(os.getenv("COLLISION_THRESHOLD", "1.0"))
WALL_PENALTY_SCALE = float(os.getenv("WALL_PENALTY_SCALE", "0.01"))

# Slip penalty configuration (GT Sophy-style)
SLIP_PENALTY_SCALE = float(os.getenv("SLIP_PENALTY_SCALE", "0.01"))
SLIP_THRESHOLD = float(os.getenv("SLIP_THRESHOLD", "8.0"))

# Minimum speed penalty configuration
MIN_SPEED_THRESHOLD = float(os.getenv("MIN_SPEED_THRESHOLD", "15.0"))
MIN_SPEED_PENALTY_SCALE = float(os.getenv("MIN_SPEED_PENALTY_SCALE", "0.1"))

# Observation configuration
NUM_LIDAR_BEAMS = int(os.getenv("NUM_LIDAR_BEAMS", "60"))
LIDAR_MAX_DISTANCE = float(os.getenv("LIDAR_MAX_DISTANCE", "50.0"))

# SAC training parameters
training_params = dict(
    learning_rate=1e-4,
    batch_size=512,
    gamma=0.995,
    ent_coef="auto",
    target_entropy="auto",
    use_sde=True,
    sde_sample_freq=RUN_FPS * 2,
    tensorboard_log=f"./runs/{RUN_NAME}",
    verbose=1,
    seed=SEED,
    device=th.device("cuda" if th.cuda.is_available() else "cpu"),
    gradient_steps=4,
    learning_starts=10000,
    policy_kwargs=dict(
        net_arch=[512, 512, 256],
        use_sde=True,
    ),
)

def find_latest_model(root_path: Path) -> Optional[Path]:
    """
        Find the path of latest model if exists.
    """
    logs_path = (os.path.join(root_path, "logs"))
    if os.path.exists(logs_path) is False:
        print(f"No previous record found in {logs_path}")
        return None
    print(f"logs_path: {logs_path}")
    files = os.listdir(logs_path)
    paths = sorted(files)
    paths_dict: Dict[int, Path] = {int(path.split("_")[2]): path for path in paths}
    if len(paths_dict) == 0:
        return None
    latest_model_file_path: Optional[Path] = Path(os.path.join(logs_path, paths_dict[max(paths_dict.keys())]))
    return latest_model_file_path

def _create_single_env(rank: int, run_name: str, run_id: str) -> gym.Env:
    """
    Create a single environment instance. Called in subprocess for parallel training.

    Args:
        rank: Environment index (0, 1, 2, ...) - determines CARLA port
        run_name: W&B run name for logging
        run_id: W&B run ID for logging
    """
    # Apply nest_asyncio in subprocess (needed for asyncio.run in spawned process)
    nest_asyncio.apply()

    # Apply lidar name mangling patch in subprocess
    _patch_carla_lidar_name_mangling()

    # Calculate port for this env
    carla_port = BASE_PORT + (rank * PORT_STRIDE)

    env = asyncio.run(
        initialize_roar_env(
            carla_port=carla_port,
            control_timestep=1.0 / RUN_FPS,
            physics_timestep=1.0 / (RUN_FPS * SUBSTEPS_PER_STEP),
            enable_rendering=ENABLE_RENDERING,
            # Lidar config
            num_lidar_beams=NUM_LIDAR_BEAMS,
            lidar_max_distance=LIDAR_MAX_DISTANCE,
            # Reward config
            progress_scale=PROGRESS_SCALE,
            time_penalty=TIME_PENALTY,
            speed_bonus_scale=SPEED_BONUS_SCALE,
            collision_threshold=COLLISION_THRESHOLD,
            wall_penalty_scale=WALL_PENALTY_SCALE,
            # Slip penalty config
            slip_penalty_scale=SLIP_PENALTY_SCALE,
            slip_threshold=SLIP_THRESHOLD,
            # Minimum speed penalty config
            min_speed_threshold=MIN_SPEED_THRESHOLD,
            min_speed_penalty_scale=MIN_SPEED_PENALTY_SCALE,
        )
    )
    env = NaNCheckWrapper(env, name=f"env_{rank}")
    env = gym.wrappers.FlattenObservation(env)
    env = FlattenActionWrapper(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=TIME_LIMIT)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if ENABLE_RENDERING and VIDEO_SAVE_FREQ > 0 and rank == 0:
        video_dir = Path("videos") / f"{run_name}_{run_id}"
        video_dir.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_dir.as_posix(),
            step_trigger=lambda step: step % VIDEO_SAVE_FREQ == 0,
            name_prefix="train",
        )

    # Per-env Monitor logs to separate directories
    log_dir = f"logs/{run_name}_{run_id}/env_{rank}"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir, allow_early_resets=True)

    # Seed for reproducibility (different seed per env)
    env.reset(seed=SEED + rank)

    return env


def make_env(rank: int, run_name: str, run_id: str) -> Callable[[], gym.Env]:
    """
    Factory function that returns a callable to create an environment.
    Must be defined at module scope for Windows multiprocessing (pickling).

    Args:
        rank: Environment index
        run_name: W&B run name
        run_id: W&B run ID

    Returns:
        A callable that creates the environment
    """
    def _init() -> gym.Env:
        set_random_seed(SEED + rank)
        return _create_single_env(rank, run_name, run_id)
    return _init

def main():
    wandb_run = wandb.init(
        project=PROJECT_NAME,
        name=RUN_NAME,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        config={
            "n_envs": N_ENVS,
            "base_port": BASE_PORT,
            "port_stride": PORT_STRIDE,
            "seed": SEED,
            "run_fps": RUN_FPS,
            # Reward config
            "progress_scale": PROGRESS_SCALE,
            "time_penalty": TIME_PENALTY,
            "speed_bonus_scale": SPEED_BONUS_SCALE,
            "collision_threshold": COLLISION_THRESHOLD,
            "wall_penalty_scale": WALL_PENALTY_SCALE,
            # Slip penalty config
            "slip_penalty_scale": SLIP_PENALTY_SCALE,
            "slip_threshold": SLIP_THRESHOLD,
            # Minimum speed penalty config
            "min_speed_threshold": MIN_SPEED_THRESHOLD,
            "min_speed_penalty_scale": MIN_SPEED_PENALTY_SCALE,
            # Observation config
            "num_lidar_beams": NUM_LIDAR_BEAMS,
            "lidar_max_distance": LIDAR_MAX_DISTANCE,
            # Training hyperparameters
            "learning_rate": training_params["learning_rate"],
            "batch_size": training_params["batch_size"],
            "gamma": training_params["gamma"],
            "ent_coef": training_params["ent_coef"],
            "gradient_steps": training_params["gradient_steps"],
            "learning_starts": training_params["learning_starts"],
            "net_arch": training_params["policy_kwargs"]["net_arch"],
        }
    )

    # Create vectorized environment
    print(f"Creating {N_ENVS} parallel environment(s)...")
    env_fns = [make_env(rank, wandb_run.name, wandb_run.id) for rank in range(N_ENVS)]

    if N_ENVS == 1:
        # Single env: use DummyVecEnv (simpler, no subprocess overhead)
        vec_env = DummyVecEnv(env_fns)
    else:
        # Multiple envs: use SubprocVecEnv for true parallelism
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")

    # Wrap with VecMonitor for aggregated metrics
    vec_env = VecMonitor(vec_env, f"logs/{wandb_run.name}_{wandb_run.id}")

    models_path = f"models/{wandb_run.name}"
    if RESUME_CHECKPOINT:
        latest_model_path = Path(RESUME_CHECKPOINT)
    else:
        latest_model_path = find_latest_model(Path(models_path))

    if latest_model_path is None:
        print("No previous model found, starting fresh training...\n")
        model = SAC(
            "MlpPolicy",
            vec_env,
            replay_buffer_kwargs={"handle_timeout_termination": True},
            **training_params
        )
    else:
        print(f"Reloading from {latest_model_path}\n")
        model = SAC.load(
            latest_model_path,
            env=vec_env,
            **training_params
        )

    wandb_callback = WandbCallback(
        gradient_save_freq=MODEL_SAVE_FREQ,
        model_save_path=f"models/{wandb_run.name}",
        verbose=2,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=MODEL_SAVE_FREQ,
        verbose=2,
        save_path=f"{models_path}/logs"
    )
    min_speed_penalty_callback = MinSpeedPenaltyLogger()
    event_callback = EveryNTimesteps(
        n_steps=MODEL_SAVE_FREQ,
        callback=checkpoint_callback
    )

    callbacks = CallbackList([
        wandb_callback,
        min_speed_penalty_callback,
        checkpoint_callback,
        event_callback
    ])

    model.learn(
        total_timesteps=1e7,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    # Clean up
    vec_env.close()


if __name__ == "__main__":
    # Only apply nest_asyncio in main process for DummyVecEnv case
    # SubprocVecEnv spawns new processes that apply it themselves
    if N_ENVS == 1:
        nest_asyncio.apply()
    main()
