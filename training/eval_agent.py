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
import tqdm

RUN_FPS= 25
SUBSTEPS_PER_STEP = 5
MODEL_SAVE_FREQ = 20_000
VIDEO_SAVE_FREQ = 10_000
training_params = dict(
    learning_rate = 1e-5,  # be smaller 2.5e-4
    #n_steps = 256 * RUN_FPS, #1024
    batch_size=256,  # mini_batch_size = 256?
    # n_epochs=10,
    gamma=0.97,  # rec range .9 - .99 0.999997
    ent_coef="auto",
    target_entropy=-10.0,
    # gae_lambda=0.95,
    # clip_range_vf=None,
    # vf_coef=0.5,
    # max_grad_norm=0.5,
    use_sde=True,
    sde_sample_freq=RUN_FPS * 2,
    # target_kl=None,
    # tensorboard_log=(Path(misc_params["model_directory"]) / "tensorboard").as_posix(),
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

def get_env(wandb_run, video_dir: Path) -> gym.Env:
    env = asyncio.run(initialize_roar_env(
        control_timestep=1.0/RUN_FPS, 
        physics_timestep=1.0/(RUN_FPS*SUBSTEPS_PER_STEP),
        image_width=1920,
        image_height=1080
    ))
    env = gym.wrappers.FlattenObservation(env)
    env = FlattenActionWrapper(env)
    # env = gym.wrappers.TimeLimit(env, max_episode_steps=6000)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(env, video_dir.as_posix())
    env = Monitor(env, f"logs/{wandb_run.name}_{wandb_run.id}", allow_early_resets=True)
    return env

def log_recorded_videos(wandb_run, video_dir: Path) -> None:
    if not video_dir.exists():
        return
    video_paths = sorted(video_dir.glob("*.mp4"))
    if not video_paths:
        return
    for video_path in video_paths:
        wandb_run.log({f"video/{video_path.stem}": wandb.Video(str(video_path))})

def main():
    wandb_run = wandb.init(
        project="ROAR_PY_RL",
        name="Denser_Waypoints_And_Collision_Detection",
        sync_tensorboard=True,
        monitor_gym=False,
    ) 
    
    video_dir = Path("videos") / f"{wandb_run.name}_eval_{wandb_run.id}"
    env = get_env(wandb_run, video_dir)

    models_path = f"models/{wandb_run.name}"
    latest_model_path = find_latest_model(Path(models_path))
    #latest_model_path = Path(os.path.join(models_path, "logs", "rl_model_1099946_steps"))
    if latest_model_path is None:
        print("no model found!")
        exit()
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

    obs, info = env.reset()
    try:
        for i in tqdm.trange(3600*RUN_FPS):
            action, _state = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:    
                print("mission failed")
                break
    finally:
        env.close()
        log_recorded_videos(wandb_run, video_dir)
        return

if __name__ == "__main__":
    nest_asyncio.apply()
    main()
