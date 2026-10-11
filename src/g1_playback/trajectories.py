from __future__ import annotations

import numpy as np
import mujoco

from .model import apply_named_joint_positions, keyframe_qpos
from .motion import MotionClip


def minimum_jerk(s: np.ndarray) -> np.ndarray:
    """Unit minimum-jerk position profile, s in [0, 1]."""
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def smooth_segment(
    p0: np.ndarray,
    p1: np.ndarray,
    n: int,
) -> np.ndarray:
    if n < 2:
        raise ValueError("n must be at least 2")
    s = np.linspace(0.0, 1.0, n)
    blend = minimum_jerk(s)[:, None]
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    return p0[None, :] + blend * (p1 - p0)[None, :]


def stand_clip(model: mujoco.MjModel, duration: float = 3.0, fps: float = 25.0) -> MotionClip:
    q0 = keyframe_qpos(model, "stand")
    n = max(2, round(duration * fps))
    qpos = np.repeat(q0[None, :], n, axis=0)
    target = np.repeat(np.array([[0.65, -0.25, 1.0]]), n, axis=0)
    return MotionClip(qpos=qpos, fps=fps, target_pos=target)


def wave_clip(model: mujoco.MjModel, duration: float = 4.0, fps: float = 25.0) -> MotionClip:
    q0 = keyframe_qpos(model, "stand")
    n = max(2, round(duration * fps))
    t = np.arange(n, dtype=np.float64) / fps

    qpos = np.repeat(q0[None, :], n, axis=0)

    # Explicit names: never rely on a presumed actuator/joint ordering.
    for k, tk in enumerate(t):
        frame = {
            "right_shoulder_pitch_joint": -0.75,
            "right_shoulder_roll_joint": -0.25,
            "right_elbow_joint": 0.75 + 0.20 * np.sin(2.0 * np.pi * 1.0 * tk),
            "right_wrist_yaw_joint": 0.30 * np.sin(2.0 * np.pi * 1.0 * tk),
        }
        qpos[k] = apply_named_joint_positions(model, q0, frame)

    target = np.repeat(np.array([[0.65, -0.25, 1.05]]), n, axis=0)
    return MotionClip(qpos=qpos, fps=fps, target_pos=target)


def reach_clip(model: mujoco.MjModel, duration: float = 4.0, fps: float = 25.0) -> MotionClip:
    """A simple planner-debugging clip.

    This is NOT IK and NOT a learned whole-body generator. It exists only to
    prove the playback/data plumbing before KiMoDo is connected.

    It:
      - moves the red human-hand target on a smooth path;
      - smoothly transitions a few right-arm joints toward a reach-like pose;
      - then slightly follows the target laterally.
    """
    q0 = keyframe_qpos(model, "stand")
    n = max(2, round(duration * fps))
    qpos = np.repeat(q0[None, :], n, axis=0)

    # Human hand moves to a handshake-like location and then drifts laterally.
    n_reach = max(2, int(0.55 * n))
    target_a = smooth_segment(
        np.array([0.78, -0.42, 1.08]),
        np.array([0.62, -0.26, 1.02]),
        n_reach,
    )
    n_hold = n - n_reach
    if n_hold > 0:
        phase = np.linspace(0.0, 2.0 * np.pi, n_hold, endpoint=False)
        target_b = np.column_stack(
            [
                0.62 + 0.015 * np.sin(phase),
                -0.26 + 0.045 * np.sin(0.5 * phase),
                1.02 + 0.010 * np.sin(phase),
            ]
        )
        target = np.vstack([target_a, target_b])
    else:
        target = target_a

    # Smooth arm movement. These values are intentionally conservative and
    # should later be replaced by the KiMoDo-generated full-body reference.
    blend = minimum_jerk(np.linspace(0.0, 1.0, n))
    for k, b in enumerate(blend):
        frame = {
            "right_shoulder_pitch_joint": (1.0 - b) * 0.2 + b * (-0.80),
            "right_shoulder_roll_joint": (1.0 - b) * (-0.2) + b * (-0.12),
            "right_shoulder_yaw_joint": b * 0.05,
            "right_elbow_joint": (1.0 - b) * 1.28 + b * 0.58,
            "right_wrist_pitch_joint": b * 0.05,
        }
        qpos[k] = apply_named_joint_positions(model, q0, frame)

    return MotionClip(qpos=qpos, fps=fps, target_pos=target)
