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
        local_velocimeter_sensor,
        collision_sensor,
        collision_threshold: float = 1.0,
        waypoint_information_distances=set([]),
        world=None,
        render_mode="rgb_array",
        # ROAR Berkeley style reward parameters (adapted to continuous progress)
        progress_scale: float = 15.0,
        step_penalty: float = 1.0,
        collision_penalty: float = 25.0,
        stall_frames_threshold: int = 10,
        stall_penalty: float = 25.0,
        reverse_penalty: float = 25.0,
        steering_deadzone: float = 0.01,
        steering_deadzone_reward: float = 0.1,
        heading_penalty_scale: float = 0.1,
        heading_penalty_threshold: float = 0.4,
        heading_lookahead: float = 10.0,
        speed_heading_penalty_scale: float = 0.0,
        fixed_spawn_point_index: Optional[int] = None,
    ):
        super().__init__(
            actor,
            manuverable_waypoints,
            location_sensor,
            roll_pitch_yaw_sensor,
            local_velocimeter_sensor,
            collision_sensor,
            collision_threshold=collision_threshold,
            waypoint_information_distances=waypoint_information_distances,
            world=world,
            render_mode=render_mode,
            progress_scale=progress_scale,
            step_penalty=step_penalty,
            collision_penalty=collision_penalty,
            stall_frames_threshold=stall_frames_threshold,
            stall_penalty=stall_penalty,
            reverse_penalty=reverse_penalty,
            steering_deadzone=steering_deadzone,
            steering_deadzone_reward=steering_deadzone_reward,
            heading_penalty_scale=heading_penalty_scale,
            heading_penalty_threshold=heading_penalty_threshold,
            heading_lookahead=heading_lookahead,
            speed_heading_penalty_scale=speed_heading_penalty_scale,
        )
        self.fixed_spawn_point_index = fixed_spawn_point_index

    def reset_vehicle(self) -> None:
        # assert isinstance(self.roar_py_actor, RoarPyCarlaVehicle)
        # assert isinstance(self.roar_py_world, RoarPyCarlaWorld)
        vehicle : RoarPyCarlaVehicle = self.roar_py_actor

        spawn_points = self.roar_py_world.spawn_points
        if self.fixed_spawn_point_index is None:
            spawn_index = np.random.randint(len(spawn_points))
        else:
            spawn_index = self.fixed_spawn_point_index
            if spawn_index < 0 or spawn_index >= len(spawn_points):
                raise ValueError(
                    f"fixed_spawn_point_index {spawn_index} out of range (0..{len(spawn_points) - 1})"
                )
        next_spawn_loc, next_spawn_rpy = spawn_points[spawn_index]
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
