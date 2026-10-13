"""A synthetic kinematic plant for algorithm checks, not a robot simulator."""
from __future__ import annotations
import numpy as np
from .config import MPCConfig
from .geometry import rotation, yaw_quaternion
from .interfaces import RobotState, TaskSpacePlan


def neutral_state(config: MPCConfig | None = None, timestamp_s: float = 0.0,
                  pelvis_xz=(0.0, 0.0)) -> RobotState:
    c = config or MPCConfig()
    pelvis = np.array([pelvis_xz[0], c.pelvis_height_m, pelvis_xz[1]])
    body_q = yaw_quaternion(c.fixed_heading_rad)
    wrist = pelvis + rotation(body_q) @ np.asarray(c.neutral_wrist_local_m)
    return RobotState(timestamp_s, pelvis, body_q, wrist,
                      np.asarray(c.fixed_wrist_orientation_xyzw), frame_id=c.frame_id)


def advance_plant(state: RobotState, plan: TaskSpacePlan, dt_s: float,
                  acceleration_scale: float = 1.0) -> RobotState:
    """Apply first acceleration to measured state, optionally with plant mismatch.

    scale=1 is an ideal double integrator. scale=0.8 is a deliberately slower
    plant. No IK, balance, walking, contact or robot dynamics are modeled.
    """
    if not np.isfinite(acceleration_scale) or acceleration_scale <= 0:
        raise ValueError("acceleration_scale must be finite and positive")
    if dt_s <= 0 or dt_s > plan.timestamps_s[1] - plan.timestamps_s[0] + 1e-8:
        raise ValueError("Plant step must be inside the first MPC control interval")
    ab = acceleration_scale * plan.pelvis_accelerations_mps2[0]
    ah = acceleration_scale * plan.right_wrist_accelerations_mps2[0]
    return RobotState(
        state.timestamp_s + dt_s,
        state.pelvis_position_m + dt_s * state.pelvis_velocity_mps + 0.5 * dt_s**2 * ab,
        state.pelvis_orientation_xyzw,
        state.right_wrist_position_m + dt_s * state.right_wrist_velocity_mps + 0.5 * dt_s**2 * ah,
        state.right_wrist_orientation_xyzw,
        state.pelvis_velocity_mps + dt_s * ab,
        state.right_wrist_velocity_mps + dt_s * ah,
        state.frame_id)
