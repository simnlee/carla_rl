import argparse
import asyncio
from typing import Iterable, Optional, Sequence

import carla
import matplotlib.pyplot as plt
import numpy as np
import roar_py_carla

DEFAULT_CONTROL_TIMESTEP = 0.05


def _looks_like_vector(value: object) -> bool:
    return isinstance(value, (list, tuple, np.ndarray))


def _to_location_array(value: object) -> np.ndarray:
    if isinstance(value, (list, tuple)) and len(value) == 2 and _looks_like_vector(value[0]):
        value = value[0]
    if hasattr(value, "location"):
        value = value.location
    if hasattr(value, "x") and hasattr(value, "y"):
        z_value = value.z if hasattr(value, "z") else 0.0
        return np.array([value.x, value.y, z_value], dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _waypoints_xy(waypoints: Iterable[object]) -> np.ndarray:
    points = []
    for waypoint in waypoints:
        location = waypoint.location if hasattr(waypoint, "location") else waypoint
        location = _to_location_array(location)
        if location.shape[0] < 2:
            raise ValueError("Waypoint location is missing XY coordinates.")
        points.append(location[:2])
    if not points:
        raise ValueError("No waypoints found in the CARLA world.")
    return np.vstack(points)


def _spawn_points_xy(spawn_points: Sequence[object]) -> np.ndarray:
    points = []
    for spawn_point in spawn_points:
        location = _to_location_array(spawn_point)
        if location.shape[0] < 2:
            raise ValueError("Spawn point is missing XY coordinates.")
        points.append(location[:2])
    if not points:
        raise ValueError("No spawn points found in the CARLA world.")
    return np.vstack(points)


def _plot_track(
    waypoint_xy: np.ndarray,
    spawn_xy: np.ndarray,
    highlight_spawn: Optional[int],
    label_spawns: bool,
    close_track: bool,
    title: str,
) -> None:
    if close_track and waypoint_xy.shape[0] > 1:
        waypoint_xy = np.vstack([waypoint_xy, waypoint_xy[0]])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(
        waypoint_xy[:, 0],
        waypoint_xy[:, 1],
        color="#1f77b4",
        linewidth=2.0,
        label="Waypoints arc",
        zorder=1,
    )
    ax.scatter(
        spawn_xy[:, 0],
        spawn_xy[:, 1],
        s=50,
        color="#d62728",
        edgecolors="white",
        linewidths=0.6,
        label="Spawn points",
        zorder=3,
    )

    if highlight_spawn is not None and 0 <= highlight_spawn < spawn_xy.shape[0]:
        ax.scatter(
            spawn_xy[highlight_spawn, 0],
            spawn_xy[highlight_spawn, 1],
            s=140,
            color="#ffcc00",
            edgecolors="black",
            linewidths=0.8,
            marker="*",
            label=f"Spawn {highlight_spawn}",
            zorder=4,
        )

    if label_spawns:
        for idx, (x, y) in enumerate(spawn_xy):
            ax.text(x, y, str(idx), fontsize=7, color="black", ha="center", va="center", zorder=5)

    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", "box")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot track waypoints and spawn points from a CARLA world."
    )
    parser.add_argument("--host", default="localhost", help="CARLA host (default: localhost).")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port (default: 2000).")
    parser.add_argument("--timeout", type=float, default=10.0, help="CARLA client timeout seconds.")
    parser.add_argument(
        "--highlight-spawn",
        type=int,
        default=0,
        help="Spawn point index to highlight (default: 0).",
    )
    parser.add_argument(
        "--label-spawns",
        dest="label_spawns",
        action="store_true",
        default=True,
        help="Annotate spawn points with their indices (default).",
    )
    parser.add_argument(
        "--no-label-spawns",
        dest="label_spawns",
        action="store_false",
        help="Disable spawn point index labels.",
    )
    parser.add_argument(
        "--no-close-track",
        action="store_true",
        help="Do not close the waypoint arc.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save the plot image instead of showing it.",
    )
    args = parser.parse_args()

    carla_client = carla.Client(args.host, args.port)
    carla_client.set_timeout(args.timeout)
    roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)

    try:
        world = roar_py_instance.world
        world.set_asynchronous(False)
        if world.control_timestep is None or world.control_timestep <= 0:
            world.control_timestep = DEFAULT_CONTROL_TIMESTEP
        await world.step()

        waypoints = getattr(world, "maneuverable_waypoints", None)
        if waypoints is None:
            waypoints = getattr(world, "manuverable_waypoints", None)
        if waypoints is None:
            raise AttributeError("World does not expose maneuverable waypoints.")

        spawn_points = world.spawn_points
        waypoint_xy = _waypoints_xy(waypoints)
        spawn_xy = _spawn_points_xy(spawn_points)

        _plot_track(
            waypoint_xy=waypoint_xy,
            spawn_xy=spawn_xy,
            highlight_spawn=args.highlight_spawn,
            label_spawns=args.label_spawns,
            close_track=not args.no_close_track,
            title="CARLA Track Waypoints + Spawn Points",
        )
    finally:
        roar_py_instance.close()

    if args.save:
        plt.savefig(args.save, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    asyncio.run(main())
