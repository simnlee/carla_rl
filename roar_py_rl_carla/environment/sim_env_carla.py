from typing import List, Optional
from gymnasium.core import Env
from roar_py_interface import RoarPyActor, RoarPyWaypoint, RoarPyWorld, RoarPyLocationInWorldSensor, RoarPyCollisionSensor, RoarPyVelocimeterSensor
from roar_py_rl import RoarRLEnv, RoarRLSimEnv
from roar_py_carla import RoarPyCarlaVehicle, RoarPyCarlaWorld
from typing import Any, Dict, SupportsFloat, Tuple, Optional
import gymnasium as gym
import numpy as np
import asyncio
import transforms3d as tr3d

class FlattenActionWrapper(gym.ActionWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.action_space = gym.spaces.flatten_space(self.env.action_space)

    def action(self, action: Any) -> Any:
        return gym.spaces.unflatten(self.env.action_space, action)
    
class RoarRLCarlaSimEnv(RoarRLSimEnv):
    def __init__(
        self,
        actor,
        manuverable_waypoints,
        location_sensor,
        roll_pitch_yaw_sensor,
        velocimeter_sensor,
        collision_sensor,
        collision_threshold: float = 30.0,
        waypoint_information_distances=set([]),
        world=None,
        render_mode="rgb_array",
        progress_scale: float = 1.0,
        time_penalty: float = 0.1,
        speed_bonus_scale: float = 0.0,
        wall_penalty_scale: float = 0.01,
    ):
        super().__init__(
            actor,
            manuverable_waypoints,
            location_sensor,
            roll_pitch_yaw_sensor,
            velocimeter_sensor,
            collision_sensor,
            collision_threshold=collision_threshold,
            waypoint_information_distances=waypoint_information_distances,
            world=world,
            render_mode=render_mode,
            progress_scale=progress_scale,
            time_penalty=time_penalty,
            speed_bonus_scale=speed_bonus_scale,
            wall_penalty_scale=wall_penalty_scale,
        )

    def reset_vehicle(self) -> None:
        # assert isinstance(self.roar_py_actor, RoarPyCarlaVehicle)
        # assert isinstance(self.roar_py_world, RoarPyCarlaWorld)
        vehicle : RoarPyCarlaVehicle = self.roar_py_actor

        spawn_points = self.roar_py_world.spawn_points
        next_spawn_loc, next_spawn_rpy = spawn_points[np.random.randint(len(spawn_points))]
        next_spawn_loc, next_spawn_rpy = next_spawn_loc.copy(), next_spawn_rpy.copy()

        rotated_extent = vehicle.bounding_box.extent.copy()
        rotated_extent = np.linalg.inv(tr3d.euler.euler2mat(*vehicle.get_roll_pitch_yaw())) @ rotated_extent
        next_spawn_loc += np.array([0,0,rotated_extent[2]+0.2])

        print(f"Resetting vehicle to {next_spawn_loc} {next_spawn_rpy}")

        # next_spawn_wp = self.manuverable_waypoints[np.random.randint(len(self.manuverable_waypoints))]
        # next_spawn_loc, next_spawn_rpy = next_spawn_wp.location, next_spawn_wp.roll_pitch_yaw
        # next_spawn_loc, next_spawn_rpy = next_spawn_loc.copy(), next_spawn_rpy.copy()
        # next_spawn_loc += np.array([0, 0, 2.0])
        
        brake_action = {
            "throttle": 0.0,
            "steer": 0.0,
            "brake": 1.0,
            "hand_brake": 1.0,
            "reverse": False
        }

        async def wait_for_world_ticks(spawn_ticks : int, wait_ticks : int) -> bool:
            for _ in range(spawn_ticks):
                self.roar_py_actor.set_transform(next_spawn_loc, next_spawn_rpy)
                self.roar_py_actor.set_linear_3d_velocity(np.zeros(3))
                self.roar_py_actor.set_angular_velocity(np.zeros(3))
                await self.roar_py_actor.apply_action(brake_action)
                await self.roar_py_world.step()
                observation_task_async = asyncio.gather(
                    self.roar_py_actor.receive_observation(),
                    *[sensor.receive_observation() for sensor in self.sensors_to_update],
                    *[sensor.receive_observation() for sensor in self.additional_sensors]
                )
                await observation_task_async
            for _ in range(wait_ticks):
                await self.roar_py_actor.apply_action(brake_action)
                await self.roar_py_world.step()
                observation_task_async = asyncio.gather(
                    self.roar_py_actor.receive_observation(),
                    *[sensor.receive_observation() for sensor in self.sensors_to_update],
                    *[sensor.receive_observation() for sensor in self.additional_sensors]
                )
                await observation_task_async
            
            await self.roar_py_actor.apply_action(brake_action)
            collision_impulse = np.linalg.norm(self.collision_sensor.get_last_gym_observation())
            if (collision_impulse > self.collision_threshold):
                return False
            else:
                return True
        
        reset_completed = asyncio.get_event_loop().run_until_complete(
            wait_for_world_ticks(5, int(1.0 / self.roar_py_world.control_timestep))
        )
        assert reset_completed
