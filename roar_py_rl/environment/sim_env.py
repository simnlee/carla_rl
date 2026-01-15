from typing import List, Optional
from roar_py_interface import RoarPyActor, RoarPySensor, RoarPyWaypoint, RoarPyWorld, RoarPyLocationInWorldSensor, RoarPyCollisionSensor, RoarPyVelocimeterSensor, RoarPyRollPitchYawSensor, RoarPyAccelerometerSensor, RoarPyWaypointsTracker, RoarPyWaypointsProjection
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
            collision_threshold : float = 30.0,
            waypoint_information_distances : Set[float] = set([]),
            world: Optional[RoarPyWorld] = None,
            render_mode="rgb_array",
            progress_scale: float = 1.0,
            time_penalty: float = 0.1,
            speed_bonus_scale: float = 0.0,
            wall_penalty_scale: float = 0.01,
            accelerometer_sensor: Optional[RoarPyAccelerometerSensor] = None,
            slip_penalty_scale: float = 0.01,
            slip_threshold: float = 8.0,
            min_speed_threshold: float = 15.0,
            min_speed_penalty_scale: float = 0.1,
            # Heading penalty parameters
            heading_penalty_scale: float = 0.3,
            heading_penalty_threshold: float = 0.15,
        ) -> None:
        super().__init__(actor, manuverable_waypoints, world, render_mode)
        self.location_sensor = location_sensor
        self.roll_pitch_yaw_sensor = roll_pitch_yaw_sensor
        self.local_velocimeter_sensor = local_velocimeter_sensor
        self.collision_sensor = collision_sensor
        self.collision_threshold = collision_threshold
        self.waypoint_information_distances = waypoint_information_distances

        # Reward parameters
        self.progress_scale = progress_scale
        self.time_penalty = time_penalty
        self.speed_bonus_scale = speed_bonus_scale
        self.wall_penalty_scale = wall_penalty_scale

        # Slip penalty parameters (GT Sophy-style)
        self.accelerometer_sensor = accelerometer_sensor
        self.slip_penalty_scale = slip_penalty_scale
        self.slip_threshold = slip_threshold
        self.min_speed_threshold = min_speed_threshold
        self.min_speed_penalty_scale = min_speed_penalty_scale

        # Heading penalty parameters
        self.heading_penalty_scale = heading_penalty_scale
        self.heading_penalty_threshold = heading_penalty_threshold

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
        sensors = [self.location_sensor, self.roll_pitch_yaw_sensor, self.local_velocimeter_sensor, self.collision_sensor]
        if self.accelerometer_sensor is not None:
            sensors.append(self.accelerometer_sensor)
        return [
            sensor for sensor in sensors
            if sensor not in self.roar_py_actor.get_sensors()
        ]

    def get_reward(self, observation : Any, action : Any, info_dict : Dict[str, Any]) -> SupportsFloat:
        # Get collision data for potential penalty
        collision_impulse : np.ndarray = self.collision_sensor.get_last_gym_observation()
        collision_impulse_norm = np.linalg.norm(collision_impulse)
        speed = np.linalg.norm(self.local_velocimeter_sensor.get_last_gym_observation())

        # Component 1: Progress reward
        # Reward forward progress along the track
        progress_reward = self.progress_scale * self._delta_distance_travelled

        # Component 2: Time penalty
        # Makes standing still costly, prevents "wait out the clock" strategy
        time_penalty_reward = -self.time_penalty

        # Component 3: Speed bonus (optional)
        # Direct reward for going fast, off by default
        speed_bonus_reward = 0.0
        if self.speed_bonus_scale > 0:
            speed_bonus_reward = self.speed_bonus_scale * speed

        # Component 4: Collision penalty
        collision_reward = 0.0
        if collision_impulse_norm > self.collision_threshold:
            collision_reward = -self.wall_penalty_scale * (speed ** 2)
            info_dict["collision_speed_mps"] = speed

        # Component 5: Slip penalty (lateral acceleration proxy)
        # Penalizes high lateral acceleration indicating tire slip / loss of grip
        slip_penalty = 0.0
        if self.accelerometer_sensor is not None:
            accel_world = self.accelerometer_sensor.get_last_gym_observation()

            # Convert world-frame acceleration to local-frame using vehicle orientation
            roll_pitch_yaw = self.roll_pitch_yaw_sensor.get_last_gym_observation()
            roll, pitch, yaw = roll_pitch_yaw
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)
            # Rotation matrix from local to world (Z-Y-X / yaw-pitch-roll)
            # We need the transpose to go from world to local
            rotation = np.array([
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp,     cp * sr,                cp * cr],
            ], dtype=np.float32)
            accel_local = rotation.T @ accel_world

            # In local frame: [forward, lateral, vertical]
            # Lateral acceleration (y component) indicates sideways force / tire slip
            lateral_accel = abs(accel_local[1])

            # Only penalize above threshold (normal cornering is fine)
            if lateral_accel > self.slip_threshold:
                excess = lateral_accel - self.slip_threshold
                slip_penalty = -self.slip_penalty_scale * (excess ** 2)

            info_dict["lateral_accel_mps2"] = lateral_accel
            info_dict["reward_slip_penalty"] = slip_penalty

        # Component 6: Minimum speed penalty
        # Forces agent to maintain speed, then slip penalty teaches corner limits
        min_speed_penalty = 0.0
        if speed < self.min_speed_threshold:
            deficit = self.min_speed_threshold - speed
            min_speed_penalty = -self.min_speed_penalty_scale * deficit
        info_dict["reward_min_speed_penalty"] = min_speed_penalty

        # Component 7: Heading penalty
        # Penalizes when car heading differs from track heading
        # This teaches "follow the track direction" before crashes happen
        heading_penalty = 0.0
        if self.heading_penalty_scale > 0 and len(self.waypoint_information_distances) > 0:
            # Get the nearest waypoint distance for heading reference
            nearest_dist = min(self.waypoint_information_distances)
            traced_forward = self.waypoints_tracer.trace_forward_projection(
                self._traced_projection, nearest_dist
            )
            traced_wp = self.waypoints_tracer.get_interpolated_waypoint(traced_forward)

            # Compute heading error: difference between track heading and car heading
            car_yaw = self.roll_pitch_yaw_sensor.get_last_gym_observation()[2]
            track_yaw = traced_wp.roll_pitch_yaw[2]
            heading_error = abs(normalize_rad(track_yaw - car_yaw))

            # Only penalize above threshold (small deviations are fine)
            if heading_error > self.heading_penalty_threshold:
                excess = heading_error - self.heading_penalty_threshold
                heading_penalty = -self.heading_penalty_scale * excess

            info_dict["heading_error_rad"] = heading_error
        info_dict["reward_heading_penalty"] = heading_penalty

        # Combine components
        total_reward = (
            progress_reward
            + time_penalty_reward
            + speed_bonus_reward
            + collision_reward
            + slip_penalty
            + min_speed_penalty
            + heading_penalty
        )

        # Log components for debugging (visible in info_dict)
        info_dict["reward_progress"] = progress_reward
        info_dict["reward_time_penalty"] = time_penalty_reward
        info_dict["reward_speed_bonus"] = speed_bonus_reward
        info_dict["reward_collision"] = collision_reward
        info_dict["speed_mps"] = speed

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
