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
            velocimeter_sensor : RoarPyVelocimeterSensor,
            collision_sensor : RoarPyCollisionSensor,
            collision_threshold : float = 30.0,
            waypoint_information_distances : Set[float] = set([]),
            world: Optional[RoarPyWorld] = None,
            render_mode="rgb_array"
        ) -> None:
        super().__init__(actor, manuverable_waypoints, world, render_mode)
        self.location_sensor = location_sensor
        self.roll_pitch_yaw_sensor = roll_pitch_yaw_sensor
        self.velocimeter_sensor = velocimeter_sensor
        self.collision_sensor = collision_sensor
        self.collision_threshold = collision_threshold
        self.waypoint_information_distances = waypoint_information_distances
        
        self.waypoints_tracer = RoarPyWaypointsTracker(manuverable_waypoints)
        self._traced_projection : RoarPyWaypointsProjection = RoarPyWaypointsProjection(0,0.0)
        self._delta_distance_travelled = 0.0

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
        return [
            sensor for sensor in
            [self.location_sensor, self.roll_pitch_yaw_sensor, self.velocimeter_sensor, self.collision_sensor]
            if sensor not in self.roar_py_actor.get_sensors()
        ]

    def get_reward(self, observation : Any, action : Any, info_dict : Dict[str, Any]) -> SupportsFloat:
        collision_impulse : np.ndarray = self.collision_sensor.get_last_gym_observation()
        collision_impulse_norm = np.linalg.norm(collision_impulse)
        
        
        collision_penalty = collision_impulse_norm / 10.0

        # Previous centerline-shaped reward (kept for reference):
        # if collision_impulse_norm > self.collision_threshold:
        #     # Proportional penalty
        #     return -collision_penalty
        # dist_to_projection = np.linalg.norm(self.location_sensor.get_last_gym_observation() - self._traced_projection_point.location)
        # if self._delta_distance_travelled <= 0:
        #     normalized_rew = self._delta_distance_travelled * 10.0 * (0.2 * dist_to_projection + 1.0)
        # else:
        #     normalized_rew = self._delta_distance_travelled * 10.0 / (0.2 * dist_to_projection + 1.0)
        # return normalized_rew - collision_penalty

        # Progress reward with continuous collision penalty (no threshold).
        progress_reward = self._delta_distance_travelled * 10.0
        return progress_reward - collision_penalty
    
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

    def is_terminated(self, observation : Any, action : Any, info_dict : Dict[str, Any]) -> bool:
        collision_impulse : np.ndarray = self.collision_sensor.get_last_gym_observation()
        collision_impulse_norm = np.linalg.norm(collision_impulse)
        if collision_impulse_norm > 0:
            print(f"Collision detected with impulse {collision_impulse_norm}", self.collision_sensor.get_last_observation().impulse_normal)
        if collision_impulse_norm > self.collision_threshold:
            print("Terminated due to collision")
            return True
        
        return False

    def is_truncated(self, observation : Any, action : Any, info_dict : Dict[str, Any]) -> bool:
        return False
