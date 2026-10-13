from __future__ import annotations
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import numpy as np
from .geometry import quaternion, vector, rotation


@dataclass
class RobotState:
    timestamp_s: float
    pelvis_position_m: np.ndarray
    pelvis_orientation_xyzw: np.ndarray
    right_wrist_position_m: np.ndarray
    right_wrist_orientation_xyzw: np.ndarray
    pelvis_velocity_mps: np.ndarray = field(default_factory=lambda: np.zeros(3))
    right_wrist_velocity_mps: np.ndarray = field(default_factory=lambda: np.zeros(3))
    frame_id: str = "planner_world"

    def __post_init__(self):
        if not math.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite")
        for name in ["pelvis_position_m", "right_wrist_position_m", "pelvis_velocity_mps",
                     "right_wrist_velocity_mps"]:
            setattr(self, name, vector(getattr(self, name), 3, name))
        for name in ["pelvis_orientation_xyzw", "right_wrist_orientation_xyzw"]:
            setattr(self, name, quaternion(getattr(self, name), name))

@dataclass
class HandshakeTarget:
    """Desired ROBOT hand-tip pose, not an unconverted observed human palm."""
    timestamp_s: float
    hand_tip_position_m: np.ndarray
    hand_tip_velocity_mps: np.ndarray = field(default_factory=lambda: np.zeros(3))
    wrist_orientation_xyzw: np.ndarray | None = None
    frame_id: str = "planner_world"

    def __post_init__(self):
        if not math.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite")
        self.hand_tip_position_m = vector(self.hand_tip_position_m, 3, "hand_tip_position_m")
        self.hand_tip_velocity_mps = vector(self.hand_tip_velocity_mps, 3, "hand_tip_velocity_mps")
        if self.wrist_orientation_xyzw is not None:
            self.wrist_orientation_xyzw = quaternion(self.wrist_orientation_xyzw)

    @classmethod
    def from_human_palm(cls, *, timestamp_s, human_palm_position_m,
                        human_palm_orientation_xyzw, palm_to_robot_tip_local_m,
                        human_palm_velocity_mps=None, **kwargs):
        """Explicit provisional contact geometry; no hidden wrist/palm/tip conflation.

        The offset is in the HUMAN palm frame. Angular-velocity compensation
        is omitted here; this mapping assumes that frame is not rotating.
        """
        position = vector(human_palm_position_m, 3, "human_palm_position_m")
        offset = vector(palm_to_robot_tip_local_m, 3, "palm_to_robot_tip_local_m")
        tip_target = position + rotation(human_palm_orientation_xyzw) @ offset
        velocity = np.zeros(3) if human_palm_velocity_mps is None else human_palm_velocity_mps
        return cls(timestamp_s, tip_target, velocity, **kwargs)


@dataclass(frozen=True)
class PlannerCommand:
    """Events: flags latch; neither flag means continue the previous command."""
    initiated: bool = False
    terminated: bool = False


@dataclass
class TaskSpacePlan:
    timestamps_s: np.ndarray
    pelvis_positions_m: np.ndarray
    pelvis_orientations_xyzw: np.ndarray
    torso_positions_m: np.ndarray
    torso_orientations_xyzw: np.ndarray
    right_wrist_positions_m: np.ndarray
    right_wrist_orientations_xyzw: np.ndarray
    right_hand_tip_positions_m: np.ndarray
    pelvis_velocities_mps: np.ndarray
    right_wrist_velocities_mps: np.ndarray
    pelvis_accelerations_mps2: np.ndarray
    right_wrist_accelerations_mps2: np.ndarray
    diagnostics: dict
    frame_id: str
    wrist_to_tip_local_m: np.ndarray
    pelvis_to_torso_local_m: np.ndarray

    @property
    def positions(self) -> np.ndarray:
        return np.column_stack([self.pelvis_positions_m[:, [0, 2]], self.right_wrist_positions_m])

    @property
    def velocities(self) -> np.ndarray:
        return np.column_stack([self.pelvis_velocities_mps[:, [0, 2]], self.right_wrist_velocities_mps])

    @property
    def accelerations(self) -> np.ndarray:
        return np.column_stack([self.pelvis_accelerations_mps2[:, [0, 2]],
                                self.right_wrist_accelerations_mps2])

    def sample(self, timestamp_s: float) -> dict:
        """Exact piecewise constant-acceleration interpolation, no extrapolation."""
        if not np.isfinite(timestamp_s) or timestamp_s < self.timestamps_s[0] - 1e-9 \
                or timestamp_s > self.timestamps_s[-1] + 1e-9:
            raise ValueError("Sample timestamp must lie inside the plan horizon")
        timestamp_s = float(np.clip(timestamp_s, self.timestamps_s[0], self.timestamps_s[-1]))
        k = min(np.searchsorted(self.timestamps_s, timestamp_s, side="right") - 1,
                len(self.timestamps_s) - 2)
        tau = timestamp_s - self.timestamps_s[k]
        pelvis = (self.pelvis_positions_m[k] + tau * self.pelvis_velocities_mps[k]
                  + 0.5 * tau**2 * self.pelvis_accelerations_mps2[k])
        wrist = (self.right_wrist_positions_m[k] + tau * self.right_wrist_velocities_mps[k]
                 + 0.5 * tau**2 * self.right_wrist_accelerations_mps2[k])
        body_q = self.pelvis_orientations_xyzw[k]
        wrist_q = self.right_wrist_orientations_xyzw[k]
        return {
            "timestamp_s": timestamp_s,
            "pelvis_position_m": pelvis,
            "pelvis_orientation_xyzw": body_q.copy(),
            "torso_position_m": pelvis + rotation(body_q) @ self.pelvis_to_torso_local_m,
            "torso_orientation_xyzw": body_q.copy(),
            "right_wrist_position_m": wrist,
            "right_wrist_orientation_xyzw": wrist_q.copy(),
            "right_hand_tip_position_m": wrist + rotation(wrist_q) @ self.wrist_to_tip_local_m,
            "pelvis_velocity_mps": self.pelvis_velocities_mps[k] + tau * self.pelvis_accelerations_mps2[k],
            "right_wrist_velocity_mps": self.right_wrist_velocities_mps[k] + tau * self.right_wrist_accelerations_mps2[k],
        }

    def to_payload(self, sample_rate_hz: float | None = None) -> dict:
        """Versioned task-space interchange. This is NOT KiMoDo native JSON."""
        if sample_rate_hz is None:
            times = self.timestamps_s
        else:
            if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
                raise ValueError("sample_rate_hz must be finite and positive")
            duration = self.timestamps_s[-1] - self.timestamps_s[0]
            times = self.timestamps_s[0] + np.arange(int(np.floor(duration * sample_rate_hz + 1e-8)) + 1) / sample_rate_hz
            times = times[times <= self.timestamps_s[-1] + 1e-9]
        samples = [self.sample(t) for t in times]
        names = ["pelvis_position_m", "pelvis_orientation_xyzw", "torso_position_m",
                 "torso_orientation_xyzw", "right_wrist_position_m", "right_wrist_orientation_xyzw",
                 "right_hand_tip_position_m", "pelvis_velocity_mps", "right_wrist_velocity_mps"]
        return {
            "schema_version": "handshake_task_space/v1",
            "coordinate_convention": {"frame_id": self.frame_id, "up_axis": "Y",
                "initial_forward_axis": "+Z", "length_unit": "m", "time_unit": "s",
                "quaternion_order": "xyzw", "rotation_semantics": "active_local_to_world"},
            "time_origin_s": float(times[0]),
            "timestamps_s": [float(t) for t in times],
            "relative_times_s": [float(t - times[0]) for t in times],
            "frame_indices": list(range(len(times))),
            "sample_rate_hz": sample_rate_hz,
            "geometry": {"wrist_to_tip_local_m": self.wrist_to_tip_local_m.tolist(),
                "pelvis_to_torso_local_m": self.pelvis_to_torso_local_m.tolist()},
            "trajectories": {name: [s[name].tolist() for s in samples] for name in names},
            "diagnostics": self.diagnostics,
            "adapter_required": True,
        }

    def save_json(self, path: str | Path, sample_rate_hz: float | None = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_payload(sample_rate_hz), indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
