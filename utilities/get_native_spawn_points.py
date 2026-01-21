import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple

import carla
import numpy as np
from roar_py_carla.utils import transform_from_carla


def _to_numpy_spawn(transform: carla.Transform) -> Tuple[np.ndarray, np.ndarray]:
    loc, rotations = transform_from_carla(transform)
    return np.asarray(loc, dtype=np.float32), np.asarray(rotations, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine CARLA map spawn points with a saved spawn npz file."
    )
    parser.add_argument("--host", default="localhost", help="CARLA host (default: localhost).")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port (default: 2000).")
    parser.add_argument("--timeout", type=float, default=10.0, help="CARLA client timeout seconds.")
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parents[1] / "assets" / "20260118_201501.npz"),
        help="Input npz file with spawn points (default: assets/20260118_201501.npz).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output npz path (default: assets/combined_spawn_points_<timestamp>.npz).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input npz not found: {input_path}")

    assets_dir = Path(__file__).resolve().parents[1] / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = assets_dir / f"combined_spawn_points_{timestamp}.npz"
    else:
        output_path = Path(args.output)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    carla_map = world.get_map()
    native_transforms = carla_map.get_spawn_points()
    if not native_transforms:
        raise RuntimeError("No spawn points returned from CARLA map.")

    native_locations = []
    native_rotations = []
    for transform in native_transforms:
        loc, rotations = _to_numpy_spawn(transform)
        native_locations.append(loc)
        native_rotations.append(rotations)

    native_locations = np.stack(native_locations, axis=0)
    native_rotations = np.stack(native_rotations, axis=0)

    with np.load(input_path) as data:
        if "locations" not in data:
            raise KeyError("Input npz must contain a 'locations' array.")
        saved_locations = np.asarray(data["locations"], dtype=np.float32)
        if "rotations" in data:
            saved_rotations = np.asarray(data["rotations"], dtype=np.float32)
        elif "rpy" in data:
            saved_rotations = np.asarray(data["rpy"], dtype=np.float32)
        else:
            raise KeyError("Input npz must contain 'rotations' (preferred) or 'rpy' arrays.")

    if saved_locations.ndim != 2 or saved_locations.shape[1] != 3:
        raise ValueError("Saved locations array must have shape (N, 3).")
    if saved_rotations.ndim != 2 or saved_rotations.shape[1] != 3:
        raise ValueError("Saved rotations array must have shape (N, 3).")

    combined_locations = np.concatenate([native_locations, saved_locations], axis=0)
    combined_rotations = np.concatenate([native_rotations, saved_rotations], axis=0)

    np.savez(output_path, locations=combined_locations, rotations=combined_rotations)
    print(
        "Saved combined spawn points:",
        f"native={native_locations.shape[0]}",
        f"saved={saved_locations.shape[0]}",
        f"total={combined_locations.shape[0]}",
        f"path={output_path}",
    )


if __name__ == "__main__":
    main()
