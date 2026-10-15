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
            hand_stride=12, max_frames=300, hand_gate_threshold=0.6,
            hand_start_s=None, root_stride=1, no_hand=False):
    """Resample executed MPC trajectory and gate hand constraints from reach_gain.

    For legacy V1 files without `scheduling.reach_gain`, retain legacy all-clip
    hand constraints unless --hand-start-s or --no-hand is specified.
    """
    if fps <= 0 or duration_s <= 0 or not math.isfinite(fps) or not math.isfinite(duration_s):
        raise ValueError("FPS and duration must be positive and finite")
    if hand_stride < 1 or root_stride < 1:
        raise ValueError("hand_stride/root_stride must be positive")
    if not (0. <= hand_gate_threshold <= 1.):
        raise ValueError("hand_gate_threshold must be inside [0, 1]")
    times, traj, conv = load_plan(plan_path)
    num_frames = int(round(duration_s * fps))
    if not (2 <= num_frames <= max_frames):
        raise ValueError(f"Requested {num_frames} frames; allowed 2..{max_frames}")
    grid = float(start_s) + np.arange(num_frames) / float(fps)
    if grid[0] < times[0] - 1e-7 or grid[-1] > times[-1] + 1e-7:
        raise ValueError(f"Requested [{grid[0]}, {grid[-1]}] s, outside MPC record [{times[0]}, {times[-1]}]")
    tr = {name: _sample(times, val, grid).tolist() for name, val in traj.items()}
    source = json.loads(Path(plan_path).read_text(encoding="utf8"))
    schedule = source.get("scheduling", {})
    gains = None
    if "reach_gain" in schedule:
        values = np.asarray(schedule["reach_gain"], dtype=float)
        if values.shape != times.shape or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("scheduling.reach_gain must be finite [0,1] per source sample")
        gains = np.interp(grid, times, values)
    if no_hand:
        start_index = num_frames
    elif hand_start_s is not None:
        if hand_start_s < 0 or not math.isfinite(hand_start_s):
            raise ValueError("hand_start_s must be finite nonnegative")
        start_index = min(num_frames, int(math.ceil((hand_start_s - 1e-8) * fps)))
    elif gains is not None:
        active = np.flatnonzero(gains >= hand_gate_threshold)
        # Activation latches after first threshold crossing, even if vision jitters.
        start_index = int(active[0]) if len(active) else num_frames
    else:
        start_index = 0
    if start_index < num_frames:
        inds = sorted(set([start_index, num_frames - 1] +
                          list(range(start_index, num_frames, int(hand_stride)))))
    else:
        inds = []
    root_inds = sorted(set([0, num_frames-1] + list(range(0, num_frames, int(root_stride)))))
    if start_index < num_frames:
        root_inds = sorted(set(root_inds + [start_index]))
    obj = {
        "schema_version": "soma_task_targets/v1",
        "source_plan": str(plan_path),
        "coordinate_convention": conv,
        "fps": float(fps),
        "num_frames": num_frames,
        "start_in_source_s": float(start_s),
        "time_s": (np.arange(num_frames) / float(fps)).tolist(),
        "right_hand_frame_indices": inds,
        "root_frame_indices": root_inds,
        "phase_split_frame": (start_index if 0 < start_index < num_frames else None),
        "reach_gain": gains.tolist() if gains is not None else None,
        "trajectories": tr,
        "notes": [
            "Positions and quaternions are intermediate targets, NOT native KiMoDo JSON.",
            "No hand constraints are supplied before the first activated frame.",
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
    p.add_argument("--root-stride", type=int, default=1, help="1=dense root, 15=one root waypoint per 0.5s at 30Hz")
    p.add_argument("--hand-gate-threshold", type=float, default=0.6)
    p.add_argument("--hand-start-s", type=float, default=None, help="Override scheduling activation relative to exported clip start")
    p.add_argument("--no-hand", action="store_true", help="Pure root-only walking ablation")
    a = p.parse_args()
    res = prepare(a.plan, a.out, a.fps, a.start, a.duration, a.hand_stride,
                  hand_gate_threshold=a.hand_gate_threshold, hand_start_s=a.hand_start_s,
                  root_stride=a.root_stride, no_hand=a.no_hand)
    print(f"Prepared {res['num_frames']} frames, {len(res['root_frame_indices'])} root waypoints, "
          f"{len(res['right_hand_frame_indices'])} right-hand keyframes, "
          f"phase split={res['phase_split_frame']}: {a.out}")

if __name__ == "__main__":
    main()
