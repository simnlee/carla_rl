import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pygame
import roar_py_interface
from roar_py_carla.sensors import carla_lidar_sensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.env_util import initialize_roar_env

RUN_FPS = 25
SUBSTEPS_PER_STEP = 5


def _patch_carla_lidar_name_mangling() -> None:
    converter = getattr(carla_lidar_sensor, "__convert_carla_lidar_raw_to_roar_py", None)
    if converter is None:
        return
    mangled_name = "_RoarPyCarlaLiDARSensor__convert_carla_lidar_raw_to_roar_py"
    if not hasattr(carla_lidar_sensor, mangled_name):
        setattr(carla_lidar_sensor, mangled_name, converter)


def _world_to_local(accel_world: np.ndarray, roll_pitch_yaw: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = roll_pitch_yaw
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,               cp * cr],
    ], dtype=np.float32)
    return rotation.T @ accel_world


class ManualControlViewer:
    def __init__(self) -> None:
        self.screen = None
        self.clock = None

    def init_pygame(self, width: int, height: int) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("RoarPy Manual Control Viewer (Accelerometer)")
        pygame.key.set_repeat()
        self.clock = pygame.time.Clock()

    def render(
        self,
        image: roar_py_interface.RoarPyCameraSensorData,
    ) -> Optional[Dict[str, Any]]:
        image_pil = image.get_image()
        if self.screen is None:
            self.init_pygame(image_pil.width, image_pil.height)

        action = {
            "throttle": 0.0,
            "steer": 0.0,
        }

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        pressed_keys = pygame.key.get_pressed()
        if pressed_keys[pygame.K_UP]:
            action["throttle"] = 1
        if pressed_keys[pygame.K_DOWN]:
            action["throttle"] = -0.4
        if pressed_keys[pygame.K_LEFT]:
            action["steer"] = -0.2
        if pressed_keys[pygame.K_RIGHT]:
            action["steer"] = 0.2

        image_surface = pygame.image.fromstring(image_pil.tobytes(), image_pil.size, image_pil.mode).convert()
        self.screen.fill((0, 0, 0))
        self.screen.blit(image_surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(RUN_FPS)
        return action


def main() -> None:
    np.set_printoptions(precision=2, suppress=True)
    control_timestep = 1.0 / RUN_FPS
    physics_timestep = control_timestep / SUBSTEPS_PER_STEP

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _patch_carla_lidar_name_mangling()

    env = loop.run_until_complete(
        initialize_roar_env(
            control_timestep=control_timestep,
            physics_timestep=physics_timestep,
            enable_rendering=True,
        )
    )

    vehicle = env.unwrapped.roar_py_actor
    camera = vehicle.attach_camera_sensor(
        roar_py_interface.RoarPyCameraSensorDataRGB,
        np.array([-2.0 * vehicle.bounding_box.extent[0], 0.0, 3.0 * vehicle.bounding_box.extent[2]]),
        np.array([0.0, 10 / 180.0 * np.pi, 0.0]),
        control_timestep=control_timestep,
        name="viewer_camera",
    )
    if camera is None:
        raise RuntimeError("Failed to attach camera sensor")

    viewer = ManualControlViewer()
    print("Use arrow keys to drive. Close the window to exit.")
    obs, _ = env.reset()
    img = loop.run_until_complete(camera.receive_observation())
    last_print_time = time.monotonic()

    try:
        while True:
            control = viewer.render(img)
            if control is None:
                break
            obs, _, terminated, truncated, _ = env.step(control)
            img = loop.run_until_complete(camera.receive_observation())
            now = time.monotonic()
            if now - last_print_time >= 1.0:
                accel_norm = np.asarray(obs.get("accelerometer"), dtype=np.float32)
                raw_obs = env.unwrapped.roar_py_actor.get_last_gym_observation()
                accel_world = np.asarray(raw_obs.get("accelerometer"), dtype=np.float32)
                rpy = np.asarray(env.unwrapped.roll_pitch_yaw_sensor.get_last_gym_observation(), dtype=np.float32)
                accel_local = _world_to_local(accel_world, rpy)
                print(f"accelerometer(norm): {accel_norm} accel_local_mps2: {accel_local}")
                last_print_time = now
            if terminated or truncated:
                obs, _ = env.reset()
                img = loop.run_until_complete(camera.receive_observation())
    finally:
        env.close()
        pygame.quit()
        loop.close()


if __name__ == "__main__":
    main()
