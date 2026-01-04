import gymnasium as gym
from gymnasium.core import Env
import numpy as np
from stable_baselines3 import SAC 
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
import asyncio
import nest_asyncio
import os
from pathlib import Path
from typing import Optional, Dict
import torch as th
from typing import Dict, SupportsFloat, Union
from env_util import initialize_roar_env
from roar_py_rl_carla import FlattenActionWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps, CallbackList, BaseCallback

RUN_FPS=25
SUBSTEPS_PER_STEP = 5
MODEL_SAVE_FREQ = 50_000
VIDEO_SAVE_FREQ = 20_000
TIME_LIMIT = RUN_FPS * 2 * 60
PROJECT_NAME = "CARLA_RL"
RUN_NAME = "Denser_Waypoints_And_Collision_Detection"
ENABLE_RENDERING = False

training_params = dict(
    learning_rate = 1e-5,  # be smaller 2.5e-4
    #n_steps = 256 * RUN_FPS, #1024
    batch_size=256,  # mini_batch_size = 256?
    # n_epochs=10,
    gamma=0.9997,  # rec range .9 - .99 0.999997
    ent_coef="auto",
    target_entropy="auto",
    # gae_lambda=0.95,
    # clip_range_vf=None,
    # vf_coef=0.5,
    # max_grad_norm=0.5,
    use_sde=True,
    sde_sample_freq = RUN_FPS * 2,
    # target_kl=None,
    tensorboard_log=f"./runs/{RUN_NAME}",  # Enable tensorboard logging for wandb sync
    # create_eval_env=False,
    # policy_kwargs=None,
    verbose=1,
    seed=1,
    device=th.device('cuda' if th.cuda.is_available() else 'cpu'),
    # _init_setup_model=True,
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

def get_env(wandb_run) -> gym.Env:
    env = asyncio.run(
        initialize_roar_env(
            control_timestep=1.0 / RUN_FPS,
            physics_timestep=1.0 / (RUN_FPS * SUBSTEPS_PER_STEP),
            enable_rendering=ENABLE_RENDERING
        )
    )
    env = gym.wrappers.FlattenObservation(env)
    env = FlattenActionWrapper(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps = TIME_LIMIT)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if ENABLE_RENDERING:
        env = gym.wrappers.RecordVideo(env, f"videos/{wandb_run.name}", step_trigger=lambda x: x % VIDEO_SAVE_FREQ == 0)
    env = Monitor(env, f"logs/{wandb_run.name}_{wandb_run.id}", allow_early_resets=True)
    return env

def main():
    wandb_run = wandb.init(
        project=PROJECT_NAME,
        name=RUN_NAME,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True
    )
    
    env = get_env(wandb_run)

    models_path = f"models/{wandb_run.name}"
    latest_model_path = find_latest_model(Path(models_path))
    
    if latest_model_path is None:
        # create new models
        model = SAC(
            "MlpPolicy",
            env,
            # optimize_memory_usage=True,
            replay_buffer_kwargs={"handle_timeout_termination": True},
            **training_params
        )
    else:
        # Load the model
        print(f"reloading from {type(latest_model_path)} {latest_model_path}\n\n\n\n")
        model = SAC.load( 
            latest_model_path,
            env=env,
            # optimize_memory_usage=True,
            # replay_buffer_kwargs={"handle_timeout_termination": False}
            **training_params
        )

    wandb_callback=WandbCallback(
        gradient_save_freq = MODEL_SAVE_FREQ,
        model_save_path = f"models/{wandb_run.name}",
        verbose = 2,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq = MODEL_SAVE_FREQ,
        verbose = 2,
        save_path = f"{models_path}/logs"
    )
    event_callback = EveryNTimesteps(
        n_steps = MODEL_SAVE_FREQ,
        callback=checkpoint_callback
    )

    callbacks = CallbackList([
        wandb_callback,
        checkpoint_callback, 
        event_callback
    ])

    model.learn(
        total_timesteps=1e7,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=False,
    )

if __name__ == "__main__":
    nest_asyncio.apply()
    main()
