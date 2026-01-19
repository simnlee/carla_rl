import gymnasium as gym
import numpy as np
from typing import Any, List, Optional, SupportsFloat, Tuple, Dict, Union
import roar_py_rl_carla
import roar_py_carla
import roar_py_interface
import carla

# =============================================================================
# Observation Normalization Constants
# =============================================================================

# Velocity (m/s) - [forward, lateral, vertical]
# Forward max updated for higher top speed; lateral/vertical remain conservative.
LOCAL_VELOCITY_MAX = np.array([90.0, 15.0, 10.0], dtype=np.float32)

# Gyroscope (rad/s) - [roll_rate, pitch_rate, yaw_rate]
GYROSCOPE_MAX = np.array([np.pi, np.pi, np.pi], dtype=np.float32)

# Accelerometer (m/s^2) - [x, y, z] in local frame
# Max ~2g for aggressive driving (braking, cornering)
ACCELEROMETER_MAX = np.array([20.0, 20.0, 20.0], dtype=np.float32)

# Waypoints
WAYPOINT_X_MARGIN = 1.5  # multiply by waypoint distance for x normalization
WAYPOINT_Y_MAX = 20.0  # meters
WAYPOINT_YAW_MAX = np.pi  # radians
WAYPOINT_LANE_WIDTH_MAX = 12.0  # meters (exact value from Monza track)

# Waypoint distances for x normalization (extended for high-speed racing)
# Provides ~6+ seconds lookahead at racing speed (200m / 30 m/s ≈ 6.7s)
WAYPOINT_DISTANCES = [2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0, 120.0, 160.0, 200.0]

class SimplifyCarlaActionFilter(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._action_space = gym.spaces.Dict({
            "throttle": gym.spaces.Box(-1.0, 1.0, (1,), np.float32),
            "steer": gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        })
    
    def action(self, action: Dict[str, Union[SupportsFloat, float]]) -> Dict[str, Union[SupportsFloat, float]]:
        """Returns a modified action before :meth:`env.step` is called.

        Args:
            action: The original :meth:`step` actions

        Returns:
            The modified actions
        """
        # action = {
        #     "throttle": [-1.0, 1.0],
        #     "steer": [-1.0, 1.0]
        # }
        # if isinstance(self.env.unwrapped, roar_py_rl_carla.RoarRLCarlaSimEnv):
        #     velocity = self.env.unwrapped.roar_py_actor.get_linear_3d_velocity()
        #     velocity = np.linalg.norm(velocity)
        
        real_action = {
            "throttle": np.clip(action["throttle"], 0.0, 1.0),
            "brake": np.clip(-action["throttle"], 0.0, 1.0),
            "steer": action["steer"],
            "hand_brake": 0.0,
            "reverse": 0.0
        }
        return real_action

class LidarObservationWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        lidar_key: str = "lidar",
        num_beams: int = 60,
        max_distance: float = 50.0,
    ):
        super().__init__(env)
        self.lidar_key = lidar_key
        self.num_beams = num_beams
        self.max_distance = max_distance
        self.observation_space = self._build_observation_space()

    def _build_observation_space(self) -> gym.spaces.Dict:
        base_space = self.env.observation_space
        if not isinstance(base_space, gym.spaces.Dict):
            raise TypeError("LidarObservationWrapper expects a Dict observation space")
        spaces = base_space.spaces.copy()
        spaces.pop(self.lidar_key, None)
        spaces["lidar"] = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_beams,),
            dtype=np.float32
        )
        return gym.spaces.Dict(spaces)

    def observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if self.lidar_key not in observation:
            return observation
        points = observation[self.lidar_key]
        beams = np.full(self.num_beams, self.max_distance, dtype=np.float32)

        if points is not None and len(points) > 0:
            points = np.asarray(points)
            if points.ndim == 1:
                points = points.reshape(1, -1)
            xyz = points[:, :3]
            dists = np.linalg.norm(xyz, axis=1)
            angles = np.arctan2(xyz[:, 1], xyz[:, 0])
            angles = (angles + 2 * np.pi) % (2 * np.pi)
            bin_ids = (angles / (2 * np.pi) * self.num_beams).astype(np.int64)
            for bin_id, dist in zip(bin_ids, dists):
                if dist < beams[bin_id]:
                    beams[bin_id] = dist

        beams = np.clip(beams / self.max_distance, 0.0, 1.0).astype(np.float32)
        obs = dict(observation)
        obs.pop(self.lidar_key, None)
        obs["lidar"] = beams
        return obs


class NormalizeObservationWrapper(gym.ObservationWrapper):
    """Normalizes observations to [-1, 1] or [0, 1] ranges using fixed constants.

    This wrapper normalizes:
    - local_velocimeter: divided by LOCAL_VELOCITY_MAX, clipped to [-1, 1]
    - gyroscope: divided by GYROSCOPE_MAX, clipped to [-1, 1]
    - accelerometer: divided by ACCELEROMETER_MAX, clipped to [-1, 1]
    - waypoints_information: each waypoint's [x, y, yaw, lane_width] normalized
      - x: forward distance to waypoint in vehicle frame (positive = ahead)
      - y: lateral distance to waypoint in vehicle frame (perpendicular to forward)
      - yaw: relative heading difference
      - lane_width: track width at waypoint
    - lidar: already normalized by LidarObservationWrapper, passed through
    - previous_action: already in [-1, 1] range, passed through
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = self._build_observation_space()

        # Precompute waypoint normalization constants for each distance
        # Keys are formatted as "waypoint_{dist}" (e.g., "waypoint_2.0")
        self._waypoint_norm = {}
        for dist in WAYPOINT_DISTANCES:
            key = f"waypoint_{dist}"
            self._waypoint_norm[key] = np.array([
                WAYPOINT_X_MARGIN * dist,  # x max (forward distance)
                WAYPOINT_Y_MAX,            # y max (lateral distance)
                WAYPOINT_YAW_MAX,          # yaw max
                WAYPOINT_LANE_WIDTH_MAX    # lane_width max
            ], dtype=np.float32)

    def _build_observation_space(self) -> gym.spaces.Dict:
        base_space = self.env.observation_space
        if not isinstance(base_space, gym.spaces.Dict):
            raise TypeError("NormalizeObservationWrapper expects a Dict observation space")

        spaces = {}

        # local_velocimeter: normalized to [-1, 1]
        if "local_velocimeter" in base_space.spaces:
            spaces["local_velocimeter"] = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(3,), dtype=np.float32
            )

        # gyroscope: normalized to [-1, 1]
        if "gyroscope" in base_space.spaces:
            spaces["gyroscope"] = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(3,), dtype=np.float32
            )

        # accelerometer: normalized to [-1, 1]
        if "accelerometer" in base_space.spaces:
            spaces["accelerometer"] = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(3,), dtype=np.float32
            )

        # lidar: already normalized to [0, 1], pass through
        if "lidar" in base_space.spaces:
            spaces["lidar"] = base_space.spaces["lidar"]

        # previous_action: already in [-1, 1], pass through
        if "previous_action" in base_space.spaces:
            spaces["previous_action"] = base_space.spaces["previous_action"]

        # waypoints_information: dict of waypoints, each normalized
        if "waypoints_information" in base_space.spaces:
            wp_base = base_space.spaces["waypoints_information"]
            if isinstance(wp_base, gym.spaces.Dict):
                wp_spaces = {}
                for key in wp_base.spaces:
                    # x, y, yaw normalized to [-1, 1], lane_width to [0, 1]
                    # Combined box with x,y,yaw in [-1,1] and lane_width in [0,1]
                    wp_spaces[key] = gym.spaces.Box(
                        low=np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32),
                        high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                        dtype=np.float32
                    )
                spaces["waypoints_information"] = gym.spaces.Dict(wp_spaces)

        return gym.spaces.Dict(spaces)

    def observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        obs = {}

        # Normalize local_velocimeter
        if "local_velocimeter" in observation:
            vel = np.asarray(observation["local_velocimeter"], dtype=np.float32)
            obs["local_velocimeter"] = np.clip(vel / LOCAL_VELOCITY_MAX, -1.0, 1.0)

        # Normalize gyroscope
        if "gyroscope" in observation:
            gyro = np.asarray(observation["gyroscope"], dtype=np.float32)
            obs["gyroscope"] = np.clip(gyro / GYROSCOPE_MAX, -1.0, 1.0)

        # Normalize accelerometer
        if "accelerometer" in observation:
            accel = np.asarray(observation["accelerometer"], dtype=np.float32)
            obs["accelerometer"] = np.clip(accel / ACCELEROMETER_MAX, -1.0, 1.0)

        # Pass through lidar (already normalized)
        if "lidar" in observation:
            obs["lidar"] = observation["lidar"]

        # Pass through previous_action (already in [-1, 1])
        if "previous_action" in observation:
            obs["previous_action"] = observation["previous_action"]

        # Normalize waypoints_information
        if "waypoints_information" in observation:
            wp_obs = {}
            for key, wp_data in observation["waypoints_information"].items():
                wp = np.asarray(wp_data, dtype=np.float32)
                norm_const = self._waypoint_norm[key]

                # Normalize: x (forward), y (lateral), yaw to [-1, 1], lane_width to [0, 1]
                normalized = wp / norm_const
                normalized[:3] = np.clip(normalized[:3], -1.0, 1.0)  # x, y, yaw
                normalized[3] = np.clip(normalized[3], 0.0, 1.0)     # lane_width
                wp_obs[key] = normalized
            obs["waypoints_information"] = wp_obs

        return obs


class LocalAccelerometerWrapper(gym.ObservationWrapper):
    """Converts accelerometer from world to local coordinates using roll/pitch/yaw."""

    def observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if "accelerometer" not in observation or "roll_pitch_yaw" not in observation:
            return observation

        accel_world = np.asarray(observation["accelerometer"], dtype=np.float32)
        roll_pitch_yaw = np.asarray(observation["roll_pitch_yaw"], dtype=np.float32)
        if accel_world.shape != (3,) or roll_pitch_yaw.shape != (3,):
            return observation

        roll, pitch, yaw = roll_pitch_yaw
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        # Rotation from local to world (Z-Y-X / yaw-pitch-roll).
        rotation = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,               cp * cr],
        ], dtype=np.float32)
        accel_local = rotation.T @ accel_world

        obs = dict(observation)
        obs["accelerometer"] = accel_local
        return obs


class PreviousActionWrapper(gym.Wrapper):
    """
    Adds the previous action to the observation space.

    This helps the agent reason about momentum and enables smoother control,
    similar to GT Sophy's "action feedback" observation.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # Get the action space shape (should be 2: throttle, steer after SimplifyCarlaActionFilter)
        # After FlattenActionWrapper, action is a flat array
        if isinstance(env.action_space, gym.spaces.Box):
            self.action_shape = env.action_space.shape
        else:
            # For Dict action space, we'll store throttle and steer
            self.action_shape = (2,)

        self._previous_action = np.zeros(self.action_shape, dtype=np.float32)

        # Modify observation space to include previous action
        base_obs_space = env.observation_space
        if isinstance(base_obs_space, gym.spaces.Dict):
            spaces = dict(base_obs_space.spaces)
            spaces["previous_action"] = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=self.action_shape,
                dtype=np.float32
            )
            self.observation_space = gym.spaces.Dict(spaces)
        else:
            # Handle Box observation space (after flattening)
            raise TypeError("PreviousActionWrapper expects Dict observation space. Apply before FlattenObservation.")

    def _add_previous_action(self, obs):
        if isinstance(obs, dict):
            obs = dict(obs)  # Copy to avoid mutating original
            obs["previous_action"] = self._previous_action.copy()
        return obs

    def reset(self, **kwargs):
        self._previous_action = np.zeros(self.action_shape, dtype=np.float32)
        obs, info = self.env.reset(**kwargs)
        return self._add_previous_action(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Store action for next step
        # Handle both dict and array actions
        if isinstance(action, dict):
            self._previous_action = np.array([
                action.get("throttle", 0.0),
                action.get("steer", 0.0)
            ], dtype=np.float32).flatten()
        else:
            self._previous_action = np.array(action, dtype=np.float32).flatten()

        return self._add_previous_action(obs), reward, terminated, truncated, info


async def initialize_roar_env(
    carla_host : str = "localhost",
    carla_port : int = 2000,
    control_timestep : float = 0.05,
    physics_timestep : float = 0.01,
    waypoint_information_distances : list = [2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0, 120.0, 160.0, 200.0],
    image_width : int = 400,
    image_height : int = 200,
    enable_rendering : bool = True,
    # Lidar config
    num_lidar_beams: int = 60,
    lidar_max_distance: float = 50.0,
    # ROAR Berkeley style reward config
    collision_threshold: float = 1.0,
    checkpoint_reward: float = 1.0,
    step_penalty: float = 1.0,
    collision_penalty: float = 25.0,
    stall_frames_threshold: int = 10,
    stall_penalty: float = 25.0,
    reverse_penalty: float = 25.0,
    steering_deadzone: float = 0.01,
    steering_deadzone_reward: float = 0.1,
):
    carla_client = carla.Client(carla_host, carla_port)
    carla_client.set_timeout(15.0)
    
    roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
    world = roar_py_instance.world
    world.set_control_steps(control_timestep, physics_timestep)
    world.set_asynchronous(False)
    
    # Toggle CARLA render pipeline for training speed.
    carla_world = roar_py_instance.world.carla_world
    settings = carla_world.get_settings()
    settings.no_rendering_mode = not enable_rendering
    carla_world.apply_settings(settings)
    print(f"CARLA rendering enabled: {enable_rendering}")
    await world.step()
    roar_py_instance.clean_actors_not_registered(["vehicle.*", "sensor.*"])

    spawn_point = world.spawn_points[0]

    vehicle = world.spawn_vehicle(
        "vehicle.tesla.model3",
        spawn_point[0] + np.array([0, 0, 2.0]),
        spawn_point[1],
        True,
        "vehicle"
    )
    assert vehicle is not None, "Failed to spawn vehicle"
    collision_sensor = vehicle.attach_collision_sensor(
        np.zeros(3),
        np.zeros(3),
        name="collision_sensor"
    )
    assert collision_sensor is not None, "Failed to attach collision sensor"
    
    #TODO: Attach next waypoint to observation
    local_velocimeter_sensor = vehicle.attach_local_velocimeter_sensor("local_velocimeter")

    location_sensor = vehicle.attach_location_in_world_sensor("location")
    rpy_sensor = vehicle.attach_roll_pitch_yaw_sensor("roll_pitch_yaw")
    gyroscope_sensor = vehicle.attach_gyroscope_sensor("gyroscope")
    accelerometer_sensor = vehicle.attach_accelerometer_sensor("accelerometer")
    if control_timestep > 0:
        # Scale points_per_second with num_lidar_beams (was 20 for 20 beams)
        lidar_points_per_second = max(1, int(round((1.0 / control_timestep) * num_lidar_beams)))
        lidar_rotation_frequency = 1.0 / control_timestep
    else:
        lidar_points_per_second = num_lidar_beams
        lidar_rotation_frequency = 10.0
    lidar_sensor = vehicle.attach_lidar_sensor(
        np.array([0.0, 0.0, vehicle.bounding_box.extent[2] + 0.2]),
        np.array([0.0, 0.0, 0.0]),
        num_lasers=1,
        max_distance=lidar_max_distance,
        points_per_second=lidar_points_per_second,
        rotation_frequency=lidar_rotation_frequency,
        upper_fov=0.0,
        lower_fov=0.0,
        horizontal_fov=360.0,
        control_timestep=control_timestep,
        name="lidar"
    )
    assert lidar_sensor is not None, "Failed to attach lidar sensor"
    if enable_rendering:
        camera = vehicle.attach_camera_sensor(
            roar_py_interface.RoarPyCameraSensorDataRGB, # Specify what kind of data you want to receive
            np.array([-2.0 * vehicle.bounding_box.extent[0], 0.0, 3.0 * vehicle.bounding_box.extent[2]]), # relative position
            np.array([0, 10/180.0*np.pi, 0]), # relative rotation
            image_width=image_width,
            image_height=image_height
        )
    # occupancy_map_sensor = vehicle.attach_occupancy_map_sensor(
    #     50,
    #     50,
    #     4.0,
    #     4.0,
    #     name="occupancy_map"
    # )

    await world.step()
    await vehicle.receive_observation()
    env = roar_py_rl_carla.RoarRLCarlaSimEnv(
        vehicle,
        world.maneuverable_waypoints,
        location_sensor,
        rpy_sensor,
        local_velocimeter_sensor,
        collision_sensor,
        waypoint_information_distances=set(waypoint_information_distances),
        world=world,
        collision_threshold=collision_threshold,
        checkpoint_reward=checkpoint_reward,
        step_penalty=step_penalty,
        collision_penalty=collision_penalty,
        stall_frames_threshold=stall_frames_threshold,
        stall_penalty=stall_penalty,
        reverse_penalty=reverse_penalty,
        steering_deadzone=steering_deadzone,
        steering_deadzone_reward=steering_deadzone_reward,
    )
    env = SimplifyCarlaActionFilter(env)
    env = LidarObservationWrapper(env, lidar_key="lidar", num_beams=num_lidar_beams, max_distance=lidar_max_distance)
    env = PreviousActionWrapper(env)
    env = LocalAccelerometerWrapper(env)
    env = gym.wrappers.FilterObservation(env, ["gyroscope", "accelerometer", "waypoints_information", "local_velocimeter", "lidar", "previous_action"])
    env = NormalizeObservationWrapper(env)
    return env
