from typing import List, Optional
from roar_py_interface import RoarPyActor, RoarPySensor, RoarPyWaypoint, RoarPyWorld, RoarPyLocationInWorldSensor, RoarPyCollisionSensor, RoarPyVelocimeterSensor, RoarPyRollPitchYawSensor, RoarPyWaypointsTracker, RoarPyWaypointsProjection
from .base_env import RoarRLEnv
from typing import Any, Dict, SupportsFloat, Tuple, Optional, Set
import gymnasium as gym
import numpy as np
from shapely import Polygon, Point
from collections import OrderedDict

def distance_to_waypoint_polygon(
    waypoint_1: RoarPyWaypoint,
    waypoint_2: RoarPyWaypoint,
    point: np.ndarray
):
    p1, p2 = waypoint_1.line_representation
    p3, p4 = waypoint_2.line_representation
    polygon = Polygon([p1, p2, p4, p3])
    return polygon.distance(Point(point))

def global_to_local(
    global_point: np.ndarray,
    local_origin: np.ndarray,
    local_yaw: float
) -> np.ndarray:
    delta_global = global_point - local_origin
    delta_local = np.array([
        np.cos(local_yaw) * delta_global[0] + np.sin(local_yaw) * delta_global[1],
        -np.sin(local_yaw) * delta_global[0] + np.cos(local_yaw) * delta_global[1]
    ])
    return delta_local

def normalize_rad(rad : float) -> float:
    return (rad % (2 * np.pi) + 3 * np.pi) % (2 * np.pi) - np.pi

class RoarRLSimEnv(RoarRLEnv):
    def __init__(
            self,
            actor: RoarPyActor,
            manuverable_waypoints: List[RoarPyWaypoint],
            location_sensor : RoarPyLocationInWorldSensor,
            roll_pitch_yaw_sensor : RoarPyRollPitchYawSensor,
            local_velocimeter_sensor : RoarPyVelocimeterSensor,
            collision_sensor : RoarPyCollisionSensor,
            collision_threshold : float = 1.0,
            waypoint_information_distances : Set[float] = set([]),
            world: Optional[RoarPyWorld] = None,
            render_mode="rgb_array",
            # ROAR Berkeley style reward parameters
            checkpoint_reward: float = 1.0,           # Reward per checkpoint crossed
            step_penalty: float = 1.0,                # "Hot water" penalty per frame
            collision_penalty: float = 25.0,          # Explicit crash penalty
            stall_frames_threshold: int = 10,         # Frames before stalling penalty
            stall_penalty: float = 25.0,              # Penalty when stuck
            reverse_penalty: float = 25.0,            # Penalty for going backward
            steering_deadzone: float = 0.01,          # Threshold for deadzone
            steering_deadzone_reward: float = 0.1,    # Reward for staying in deadzone
        ) -> None:
        super().__init__(actor, manuverable_waypoints, world, render_mode)
        self.location_sensor = location_sensor
        self.roll_pitch_yaw_sensor = roll_pitch_yaw_sensor
        self.local_velocimeter_sensor = local_velocimeter_sensor
        self.collision_sensor = collision_sensor
        self.collision_threshold = collision_threshold
        self.waypoint_information_distances = waypoint_information_distances

        # ROAR Berkeley style reward parameters
        self.checkpoint_reward = checkpoint_reward
        self.step_penalty = step_penalty
        self.collision_penalty = collision_penalty
        self.stall_frames_threshold = stall_frames_threshold
        self.stall_penalty = stall_penalty
        self.reverse_penalty = reverse_penalty
        self.steering_deadzone = steering_deadzone
        self.steering_deadzone_reward = steering_deadzone_reward

        self.waypoints_tracer = RoarPyWaypointsTracker(manuverable_waypoints)
        self._traced_projection : RoarPyWaypointsProjection = RoarPyWaypointsProjection(0,0.0)
        self._delta_distance_travelled = 0.0

        # State variables for new reward components
        self._stall_counter = 0
        self._last_waypoint_idx = 0

    @property
    def observation_space(self) -> gym.Space:
        space = super().observation_space
        if len(self.waypoint_information_distances) > 0:
            waypoints_info_space_dict = OrderedDict()
            for dist in sorted(self.waypoint_information_distances):
                waypoints_info_space_dict[f"waypoint_{dist}"] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(4,), # x, y, yaw, lane_width
                    dtype=np.float32
                )
            space["waypoints_information"] = gym.spaces.Dict(waypoints_info_space_dict)
        
        return space

    def observation(self, info_dict : Dict[str, Any]) -> Dict[str, Any]:
        obs = super().observation(info_dict)

        location = self.location_sensor.get_last_gym_observation()
        yaw = self.roll_pitch_yaw_sensor.get_last_gym_observation()[2]

        if len(self.waypoint_information_distances) > 0:
            waypoint_info = {}
            for trace_dist in self.waypoint_information_distances:
                traced_projection = self.waypoints_tracer.trace_forward_projection(self._traced_projection, trace_dist)
                traced_projection_wp = self.waypoints_tracer.get_interpolated_waypoint(traced_projection)
                waypoint_info[f"waypoint_{trace_dist}"] = np.concatenate([
                    global_to_local(traced_projection_wp.location, location, yaw),
                    np.array([normalize_rad(traced_projection_wp.roll_pitch_yaw[2] - yaw)]),
                    np.array([traced_projection_wp.lane_width])
                ])

            obs["waypoints_information"] = waypoint_info
        info_dict["delta_distance_travelled"] = self._delta_distance_travelled
        return obs

    def reset_vehicle(self) -> None:
        return NotImplementedError

    @property
    def sensors_to_update(self) -> List[RoarPySensor]:
        sensors = [self.location_sensor, self.roll_pitch_yaw_sensor, self.local_velocimeter_sensor, self.collision_sensor]
        return [
            sensor for sensor in sensors
            if sensor not in self.roar_py_actor.get_sensors()
        ]

    def get_reward(self, observation : Any, action : Any, info_dict : Dict[str, Any]) -> SupportsFloat:
        """
        ROAR Berkeley style reward function.

        Components:
        1. Checkpoint reward - discrete progress signal when crossing waypoints
        2. Step penalty ("hot water") - constant negative pressure to encourage speed
        3. Collision penalty - explicit penalty before termination
        4. Stalling penalty - penalize being stuck
        5. Reverse penalty - penalize going backward
        6. Steering deadzone reward - encourage stable steering
        """
        speed = np.linalg.norm(self.local_velocimeter_sensor.get_last_gym_observation())
        collision_impulse : np.ndarray = self.collision_sensor.get_last_gym_observation()
        collision_impulse_norm = np.linalg.norm(collision_impulse)

        # Component 1: Checkpoint reward (ROAR Berkeley style)
        # Reward given when crossing to next waypoint
        checkpoint_reward = 0.0
        current_waypoint_idx = self._traced_projection.waypoint_idx
        if current_waypoint_idx > self._last_waypoint_idx:
            # Crossed checkpoint(s) - give reward for each one crossed
            checkpoints_crossed = current_waypoint_idx - self._last_waypoint_idx
            checkpoint_reward = self.checkpoint_reward * checkpoints_crossed
            self._last_waypoint_idx = current_waypoint_idx
        elif current_waypoint_idx < self._last_waypoint_idx:
            # Handle lap wrap-around (waypoint index resets)
            if self._last_waypoint_idx > len(self.manuverable_waypoints) - 10:
                # Likely wrapped around to start of track
                checkpoints_crossed = (len(self.manuverable_waypoints) - self._last_waypoint_idx) + current_waypoint_idx
                checkpoint_reward = self.checkpoint_reward * checkpoints_crossed
            self._last_waypoint_idx = current_waypoint_idx

        # Component 2: Step penalty ("hot water")
        # Creates constant negative pressure - only way to positive reward is progress
        step_penalty_reward = -self.step_penalty

        # Component 3: Collision penalty (applied before termination)
        collision_reward = 0.0
        if collision_impulse_norm > self.collision_threshold:
            collision_reward = -self.collision_penalty
            info_dict["collision_speed_mps"] = speed

        # Component 4: Stalling penalty
        # Penalize being stuck (speed < 1 m/s for too long)
        stalling_penalty = 0.0
        if speed < 1.0:
            self._stall_counter += 1
            if self._stall_counter > self.stall_frames_threshold:
                stalling_penalty = -self.stall_penalty
        else:
            self._stall_counter = 0

        # Component 5: Reverse progress penalty
        # Penalize going backward through checkpoints
        reverse_penalty = 0.0
        if self._delta_distance_travelled < -1.0:
            reverse_penalty = -self.reverse_penalty

        # Component 6: Steering deadzone reward
        # Encourage stable steering (reduces wobbling at high speeds)
        deadzone_reward = 0.0
        # Extract steering from action (handle both dict and array formats)
        if isinstance(action, dict):
            steering = action.get("steer", 0.0)
        elif hasattr(action, '__getitem__'):
            steering = action[1] if len(action) > 1 else 0.0
        else:
            steering = 0.0
        if abs(steering) < self.steering_deadzone:
            deadzone_reward = self.steering_deadzone_reward

        # Combine all components
        total_reward = (
            checkpoint_reward
            + step_penalty_reward
            + collision_reward
            + stalling_penalty
            + reverse_penalty
            + deadzone_reward
        )

        # Log all components for debugging (visible in info_dict and wandb)
        info_dict["reward_checkpoint"] = checkpoint_reward
        info_dict["reward_step_penalty"] = step_penalty_reward
        info_dict["reward_collision"] = collision_reward
        info_dict["reward_stalling"] = stalling_penalty
        info_dict["reward_reverse"] = reverse_penalty
        info_dict["reward_deadzone"] = deadzone_reward
        info_dict["speed_mps"] = speed
        info_dict["stall_counter"] = self._stall_counter
        info_dict["waypoint_idx"] = current_waypoint_idx

        return total_reward
    
    def _perform_waypoint_trace(self, location: Optional[np.ndarray] = None) -> None:
        if location is None:
            location = self.location_sensor.get_last_gym_observation()
        _last_traced_projection = self._traced_projection
        self._traced_projection = self.waypoints_tracer.trace_point(location, _last_traced_projection.waypoint_idx)
        self._traced_projection_point = self.waypoints_tracer.get_interpolated_waypoint(self._traced_projection)
        self._delta_distance_travelled = self.waypoints_tracer.delta_distance_projection(_last_traced_projection, self._traced_projection)

    def _step(self, action: Any) -> None:
        self._perform_waypoint_trace()

    def _reset(self) -> None:
        self._perform_waypoint_trace()
        self._delta_distance_travelled = 0.0
        self._stall_counter = 0
        self._last_waypoint_idx = self._traced_projection.waypoint_idx

    def is_terminated(self, observation : Any, action : Any, info_dict : Dict[str, Any]) -> bool:
        collision_impulse : np.ndarray = self.collision_sensor.get_last_gym_observation()
        collision_impulse_norm = np.linalg.norm(collision_impulse)
        if collision_impulse_norm > 0:
            print(f"Collision detected with impulse {collision_impulse_norm}", self.collision_sensor.get_last_observation().impulse_normal)
        if collision_impulse_norm > self.collision_threshold:
            print("Terminated due to collision")
            return True

    def is_truncated(self, observation : Any, action : Any, info_dict : Dict[str, Any]) -> bool:
        return False
