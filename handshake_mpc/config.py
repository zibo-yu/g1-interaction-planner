from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import math
import numpy as np
from .geometry import quaternion, vector


@dataclass(frozen=True)
class MPCConfig:
    dt_s: float = 0.1
    horizon_steps: int = 30
    frame_id: str = "planner_world"
    pelvis_height_m: float = 0.95
    fixed_heading_rad: float = 0.0
    fixed_wrist_orientation_xyzw: tuple = (0.0, 0.0, 0.0, 1.0)
    # All offsets/regions are demo constants, not measured SOMA/robot geometry.
    wrist_to_tip_local_m: tuple = (0.0, 0.0, 0.12)
    pelvis_to_torso_local_m: tuple = (0.0, 0.25, 0.0)
    neutral_wrist_local_m: tuple = (0.22, -0.20, 0.05)
    # Forward interval [0, 0.58] includes a tucked wrist: under-extension
    # should not induce backwards walking merely to enter the preferred box.
    comfort_center_local_m: tuple = (0.22, 0.20, 0.29)
    comfort_half_widths_m: tuple = (0.14, 0.18, 0.29)
    max_pelvis_wrist_distance_m: float = 0.80
    pelvis_speed_max_mps: float = 0.45
    wrist_speed_max_mps: float = 0.65
    pelvis_acceleration_max_mps2: float = 0.65
    wrist_acceleration_max_mps2: float = 1.20
    hand_tracking_weight: float = 240.0
    terminal_hand_tracking_weight: float = 1800.0
    comfort_weight: float = 800.0
    pelvis_displacement_weight: float = 35.0
    pelvis_velocity_weight: float = 15.0
    pelvis_acceleration_weight: float = 1.0
    wrist_acceleration_weight: float = 0.10
    acceleration_change_weight: float = 0.25
    wrist_velocity_far_weight: float = 0.40
    wrist_velocity_near_weight: float = 35.0
    precontact_scale_m: float = 0.14
    max_target_age_s: float = 0.5
    target_prediction_cap_s: float = 0.6
    feasibility_tolerance: float = 2e-5
    solver_max_iterations: int = 150
    # V2 is opt-in: old configs and tests preserve their original behavior.
    gain_schedule_enabled: bool = False
    reach_gate_far_m: float = 2.0
    reach_gate_near_m: float = 1.2
    human_standoff_m: float = 0.85
    pelvis_goal_weight: float = 230.0
    terminal_pelvis_goal_weight: float = 600.0
    neutral_wrist_weight: float = 450.0

    def __post_init__(self):
        positive = ["dt_s", "pelvis_height_m", "max_pelvis_wrist_distance_m",
                    "pelvis_speed_max_mps", "wrist_speed_max_mps",
                    "pelvis_acceleration_max_mps2", "wrist_acceleration_max_mps2",
                    "precontact_scale_m", "max_target_age_s", "feasibility_tolerance",
                    "reach_gate_far_m", "reach_gate_near_m", "human_standoff_m"]
        nonnegative = [name for name in asdict(self) if name.endswith("weight")]
        nonnegative.append("target_prediction_cap_s")
        for name in positive + nonnegative:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name in positive and value == 0):
                raise ValueError(f"Invalid {name}: {value}")
        if self.reach_gate_near_m >= self.reach_gate_far_m:
            raise ValueError("reach_gate_near_m must be below reach_gate_far_m")
        if not isinstance(self.gain_schedule_enabled, bool):
            raise ValueError("gain_schedule_enabled must be boolean")
        if not math.isfinite(self.fixed_heading_rad):
            raise ValueError("fixed_heading_rad must be finite")
        if not isinstance(self.horizon_steps, int) or self.horizon_steps < 2:
            raise ValueError("horizon_steps must be an integer >= 2")
        if not isinstance(self.solver_max_iterations, int) or self.solver_max_iterations < 1:
            raise ValueError("solver_max_iterations must be a positive integer")
        if not self.frame_id:
            raise ValueError("frame_id must be nonempty")
        quaternion(self.fixed_wrist_orientation_xyzw)
        for name in ["wrist_to_tip_local_m", "pelvis_to_torso_local_m", "neutral_wrist_local_m",
                     "comfort_center_local_m", "comfort_half_widths_m"]:
            vector(getattr(self, name), 3, name)
        if np.any(np.asarray(self.comfort_half_widths_m) <= 0):
            raise ValueError("comfort_half_widths_m must be strictly positive")
        if np.linalg.norm(self.neutral_wrist_local_m) > self.max_pelvis_wrist_distance_m:
            raise ValueError("neutral wrist must satisfy the hard reach bound")

    @classmethod
    def load(cls, path: str | Path) -> "MPCConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
