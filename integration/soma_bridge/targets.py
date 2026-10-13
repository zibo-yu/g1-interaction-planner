"""Convert MPC hand/task-space JSON into discrete KiMoDo time-indexed targets.

This module deliberately depends only on NumPy and SciPy. It does NOT emit
native KiMoDo constraints.json (those require a SOMA skeletal reconstruction).
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

REQUIRED = (
    "pelvis_position_m", "pelvis_orientation_xyzw",
    "right_wrist_position_m", "right_wrist_orientation_xyzw",
    "right_hand_tip_position_m",
)


def load_plan(path):
    data = json.loads(Path(path).read_text(encoding="utf8"))
    if data.get("schema_version") != "handshake_task_space/v1":
        raise ValueError("Expected handshake_task_space/v1 JSON, not native KiMoDo constraints")
    convention = data.get("coordinate_convention", {})
    required = {"up_axis": "Y", "initial_forward_axis": "+Z", "length_unit": "m", "quaternion_order": "xyzw"}
    for k, v in required.items():
        if convention.get(k) != v:
            raise ValueError(f"Unexpected coordinate convention {k}: {convention.get(k)!r}; expected {v!r}")
    t = np.asarray(data.get("relative_times_s"), dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("relative_times_s must be a strictly increasing finite array")
    if abs(float(t[0])) > 1e-7:
        raise ValueError("relative_times_s must start at 0")
    trajectory = {}
    for name in REQUIRED:
        arr = np.asarray(data["trajectories"][name], dtype=float)
        d = 4 if name.endswith("xyzw") else 3
        if arr.shape != (len(t), d) or not np.all(np.isfinite(arr)):
            raise ValueError(f"Invalid {name}: expected {(len(t), d)}, got {arr.shape}")
        if d == 4 and np.any(np.linalg.norm(arr, axis=1) < 1e-7):
            raise ValueError(f"Zero-length quaternion in {name}")
        trajectory[name] = arr
    return t, trajectory, convention


def _sample(times, values, grid):
    if values.shape[1] == 4:
        return Slerp(times, Rotation.from_quat(values))(grid).as_quat()
    return np.column_stack([np.interp(grid, times, values[:, j]) for j in range(3)])


def prepare(plan_path, output_path, fps=30.0, start_s=0.0, duration_s=4.0,
            hand_stride=12, max_frames=300):
    if fps <= 0 or duration_s <= 0 or not math.isfinite(fps) or not math.isfinite(duration_s):
        raise ValueError("FPS and duration must be positive and finite")
    if hand_stride < 1:
        raise ValueError("hand_stride must be positive")
    times, traj, conv = load_plan(plan_path)
    num_frames = int(round(duration_s * fps))
    if not (2 <= num_frames <= max_frames):
        raise ValueError(f"Requested {num_frames} frames; allowed 2..{max_frames}")
    grid = float(start_s) + np.arange(num_frames) / float(fps)
    if grid[0] < times[0] - 1e-7 or grid[-1] > times[-1] + 1e-7:
        raise ValueError(f"Requested [{grid[0]}, {grid[-1]}] s, outside MPC record [{times[0]}, {times[-1]}]")
    tr = {name: _sample(times, val, grid).tolist() for name, val in traj.items()}
    inds = sorted(set([0, num_frames - 1] + list(range(0, num_frames, int(hand_stride)))))
    obj = {
        "schema_version": "soma_task_targets/v1",
        "source_plan": str(plan_path),
        "coordinate_convention": conv,
        "fps": float(fps),
        "num_frames": num_frames,
        "start_in_source_s": float(start_s),
        "time_s": (np.arange(num_frames) / float(fps)).tolist(),
        "right_hand_frame_indices": inds,
        "trajectories": tr,
        "notes": [
            "Positions and quaternions are intermediate targets, NOT native KiMoDo JSON.",
            "Planned hand-tip offset is not automatically the SOMA RightHandMiddleEnd offset.",
        ],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf8")
    return obj


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=4.0)
    p.add_argument("--hand-stride", type=int, default=12)
    a = p.parse_args()
    res = prepare(a.plan, a.out, a.fps, a.start, a.duration, a.hand_stride)
    print(f"Prepared {res['num_frames']} frames, {len(res['right_hand_frame_indices'])} right-hand keyframes: {a.out}")

if __name__ == "__main__":
    main()
