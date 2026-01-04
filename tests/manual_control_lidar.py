import asyncio
import time
from typing import Any, Dict, Optional

import carla
import numpy as np
import pygame
import roar_py_carla
import roar_py_interface
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

RUN_FPS = 25
SUBSTEPS_PER_STEP = 5
NUM_BEAMS = 20
MAX_DISTANCE = 50.0
POINTS_PER_SECOND = RUN_FPS * NUM_BEAMS


class ManualControlViewer:
    def __init__(self) -> None:
        self.screen = None
        self.clock = None
        self.last_control = {
            "throttle": 0.0,
            "steer": 0.0,
            "brake": 0.0,
            "hand_brake": np.array([0]),
            "reverse": np.array([0]),
        }

    def init_pygame(self, width: int, height: int) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("RoarPy Manual Control Viewer (LiDAR)")
        pygame.key.set_repeat()
        self.clock = pygame.time.Clock()

    def render(
        self,
        image: roar_py_interface.RoarPyCameraSensorData,
    ) -> Optional[Dict[str, Any]]:
        image_pil = image.get_image()
        if self.screen is None:
            self.init_pygame(image_pil.width, image_pil.height)

        new_control = {
            "throttle": 0.0,
            "steer": 0.0,
            "brake": 0.0,
            "hand_brake": np.array([0]),
            "reverse": np.array([0]),
        }

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        pressed_keys = pygame.key.get_pressed()
        if pressed_keys[pygame.K_UP]:
            new_control["throttle"] = 0.4
        if pressed_keys[pygame.K_DOWN]:
            new_control["brake"] = 0.2
        if pressed_keys[pygame.K_LEFT]:
            new_control["steer"] = -0.2
        if pressed_keys[pygame.K_RIGHT]:
            new_control["steer"] = 0.2

        image_surface = pygame.image.fromstring(image_pil.tobytes(), image_pil.size, image_pil.mode).convert()
        self.screen.fill((0, 0, 0))
        self.screen.blit(image_surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(RUN_FPS)
        self.last_control = new_control
        return new_control


def process_lidar(points: np.ndarray, num_beams: int, max_distance: float) -> np.ndarray:
    beams = np.full(num_beams, max_distance, dtype=np.float32)
    if points is None:
        return beams
    points = np.asarray(points)
    if points.size == 0:
        return beams
    if points.ndim == 1:
        points = points.reshape(1, -1)

    xyz = points[:, :3]
    dists = np.linalg.norm(xyz, axis=1)
    angles = np.arctan2(xyz[:, 1], xyz[:, 0])
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    bin_ids = (angles / (2 * np.pi) * num_beams).astype(np.int64)
    bin_ids = np.clip(bin_ids, 0, num_beams - 1)

    for bin_id, dist in zip(bin_ids, dists):
        dist = min(float(dist), max_distance)
        if dist < beams[bin_id]:
            beams[bin_id] = dist

    return beams


async def main() -> None:
    np.set_printoptions(precision=2, suppress=True)
    control_timestep = 1.0 / RUN_FPS
    physics_timestep = control_timestep / SUBSTEPS_PER_STEP

    carla_client = carla.Client("localhost", 2000)
    carla_client.set_timeout(5.0)
    roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)

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
        name="camera",
    )
    if camera is None:
        raise RuntimeError("Failed to attach camera sensor")

    lidar_sensor = vehicle.attach_lidar_sensor(
        np.array([0.0, 0.0, vehicle.bounding_box.extent[2] + 0.2]),
        np.array([0.0, 0.0, 0.0]),
        num_lasers=1,
        max_distance=MAX_DISTANCE,
        points_per_second=POINTS_PER_SECOND,
        rotation_frequency=RUN_FPS,
        upper_fov=0.0,
        lower_fov=0.0,
        horizontal_fov=360.0,
        control_timestep=control_timestep,
        name="lidar",
    )
    if lidar_sensor is None:
        raise RuntimeError("Failed to attach lidar sensor")

    viewer = ManualControlViewer()
    print("Use arrow keys to drive. Close the window to exit.")
    last_print_time = time.monotonic()

    try:
        await world.step()
        img = await camera.receive_observation()
        lidar_data = await lidar_sensor.receive_observation()
        while True:
            control = viewer.render(img)
            if control is None:
                break
            await vehicle.apply_action(control)
            await world.step()
            img = await camera.receive_observation()
            lidar_data = await lidar_sensor.receive_observation()
            distances = process_lidar(lidar_data.lidar_points_data, NUM_BEAMS, MAX_DISTANCE)
            now = time.monotonic()
            if now - last_print_time >= 1.0:
                print(f"lidar_20: {distances}")
                last_print_time = now
    finally:
        roar_py_instance.close()


if __name__ == "__main__":
    asyncio.run(main())
