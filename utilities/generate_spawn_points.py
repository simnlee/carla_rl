import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import carla
import numpy as np
import pygame
import roar_py_carla
import roar_py_interface

RUN_FPS = 25
SUBSTEPS_PER_STEP = 5

FORWARD_THROTTLE = 1.0
REVERSE_THROTTLE = 0.4
BRAKE_FORCE = 0.6
STEER_AMOUNT = 0.25
REVERSE_SPEED_THRESHOLD = 0.5


class ManualControlViewer:
    def __init__(self) -> None:
        self.screen = None
        self.clock = None
        self.capture_requested = False

    def init_pygame(self, width: int, height: int) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("RoarPy Spawn Point Generator")
        pygame.key.set_repeat()
        self.clock = pygame.time.Clock()

    def _build_control(self, forward_speed: float) -> Dict[str, Any]:
        control = {
            "throttle": 0.0,
            "steer": 0.0,
            "brake": 0.0,
            "hand_brake": np.array([0]),
            "reverse": np.array([0]),
        }

        pressed_keys = pygame.key.get_pressed()
        if pressed_keys[pygame.K_UP]:
            control["throttle"] = FORWARD_THROTTLE

        if pressed_keys[pygame.K_DOWN]:
            if forward_speed > REVERSE_SPEED_THRESHOLD:
                control["throttle"] = 0.0
                control["brake"] = BRAKE_FORCE
                control["reverse"] = np.array([0])
            else:
                control["reverse"] = np.array([1])
                control["throttle"] = REVERSE_THROTTLE

        if pressed_keys[pygame.K_LEFT]:
            control["steer"] = -STEER_AMOUNT
        if pressed_keys[pygame.K_RIGHT]:
            control["steer"] = STEER_AMOUNT

        return control

    def render(
        self,
        image: roar_py_interface.RoarPyCameraSensorData,
        forward_speed: float,
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        image_pil = image.get_image()
        if self.screen is None:
            self.init_pygame(image_pil.width, image_pil.height)

        self.capture_requested = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None, False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return None, False
                if event.key == pygame.K_p:
                    self.capture_requested = True

        control = self._build_control(forward_speed)

        image_surface = pygame.image.fromstring(image_pil.tobytes(), image_pil.size, image_pil.mode).convert()
        self.screen.fill((0, 0, 0))
        self.screen.blit(image_surface, (0, 0))
        pygame.display.flip()
        if self.clock is not None:
            self.clock.tick(RUN_FPS)

        return control, self.capture_requested


def _format_spawn_point(location: np.ndarray, rpy: np.ndarray) -> str:
    location_list = np.asarray(location, dtype=np.float32).tolist()
    rpy_list = np.asarray(rpy, dtype=np.float32).tolist()
    return f"(np.array({location_list}, dtype=np.float32), np.array({rpy_list}, dtype=np.float32))"


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drive manually and capture CARLA spawn points with the P key."
    )
    parser.add_argument("--host", default="localhost", help="CARLA host (default: localhost).")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port (default: 2000).")
    parser.add_argument("--timeout", type=float, default=10.0, help="CARLA client timeout seconds.")
    args = parser.parse_args()

    np.set_printoptions(precision=3, suppress=True)
    control_timestep = 1.0 / RUN_FPS
    physics_timestep = control_timestep / SUBSTEPS_PER_STEP

    carla_client = carla.Client(args.host, args.port)
    carla_client.set_timeout(args.timeout)
    roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)

    spawn_points: List[Tuple[np.ndarray, np.ndarray]] = []
    assets_dir = Path(__file__).resolve().parents[1] / "assets"

    try:
        world = roar_py_instance.world
        world.set_control_steps(control_timestep, physics_timestep)
        world.set_asynchronous(False)

        carla_world = world.carla_world
        settings = carla_world.get_settings()
        settings.no_rendering_mode = False
        carla_world.apply_settings(settings)

        await world.step()
        roar_py_instance.clean_actors_not_registered(["vehicle.*", "sensor.*"])

        spawn_point = world.spawn_points[0]
        vehicle = world.spawn_vehicle(
            "vehicle.tesla.model3",
            spawn_point[0] + np.array([0, 0, 2.0]),
            spawn_point[1],
            True,
            "vehicle",
        )
        if vehicle is None:
            raise RuntimeError("Failed to spawn vehicle")

        camera = vehicle.attach_camera_sensor(
            roar_py_interface.RoarPyCameraSensorDataRGB,
            np.array([-2.0 * vehicle.bounding_box.extent[0], 0.0, 3.0 * vehicle.bounding_box.extent[2]]),
            np.array([0.0, 10 / 180.0 * np.pi, 0.0]),
            control_timestep=control_timestep,
            name="viewer_camera",
        )
        if camera is None:
            raise RuntimeError("Failed to attach camera sensor")

        location_sensor = vehicle.attach_location_in_world_sensor("location")
        rpy_sensor = vehicle.attach_roll_pitch_yaw_sensor("roll_pitch_yaw")
        local_vel_sensor = vehicle.attach_local_velocimeter_sensor("local_velocimeter")

        if location_sensor is None or rpy_sensor is None or local_vel_sensor is None:
            raise RuntimeError("Failed to attach required sensors")

        viewer = ManualControlViewer()
        print("Use arrow keys to drive. Press P to capture a spawn point. Esc or close window to exit.")

        await world.step()
        img = await camera.receive_observation()
        await location_sensor.receive_observation()
        await rpy_sensor.receive_observation()
        await local_vel_sensor.receive_observation()

        while True:
            local_vel = local_vel_sensor.get_last_gym_observation()
            forward_speed = float(local_vel[0]) if local_vel is not None else 0.0
            control, capture = viewer.render(img, forward_speed)
            if control is None:
                break

            if capture:
                location = location_sensor.get_last_gym_observation()
                rpy = rpy_sensor.get_last_gym_observation()
                if location is None or rpy is None:
                    print("Spawn point capture skipped; sensor data not ready.")
                else:
                    location = np.asarray(location, dtype=np.float32).copy()
                    rpy = np.asarray(rpy, dtype=np.float32).copy()
                    spawn_points.append((location, rpy))
                    print(f"Spawn {len(spawn_points) - 1}: location {location} rpy {rpy}")

            await vehicle.apply_action(control)
            await world.step()
            img = await camera.receive_observation()
            await location_sensor.receive_observation()
            await rpy_sensor.receive_observation()
            await local_vel_sensor.receive_observation()
    finally:
        if spawn_points:
            assets_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = assets_dir / f"{timestamp}.npz"
            locations = np.stack([point[0] for point in spawn_points], axis=0)
            rpy = np.stack([point[1] for point in spawn_points], axis=0)
            np.savez(output_path, locations=locations, rpy=rpy)
            print(f"Saved {len(spawn_points)} spawn points to {output_path}")
            print("Captured spawn points (copy-ready):")
            for idx, (location, rpy) in enumerate(spawn_points):
                print(f"{idx}: {_format_spawn_point(location, rpy)}")
        roar_py_instance.close()
        pygame.quit()


if __name__ == "__main__":
    asyncio.run(main())
