"""Closed-loop kinematic MPC rollout -> dense task-space JSON for KiMoDo.

Run with the standalone Python 3.11+ MPC environment (CVXPY + Clarabel),
not KiMoDo's Python 3.10 environment. KiMoDo is only used downstream.

Human torso/hand locations here are synthetic fixed world-space measurements.
MPC updates from the actual synthetic plant at every 0.1s control step.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from handshake_mpc import HandshakeMPC, MPCConfig, HandshakeTarget, PlannerCommand
from handshake_mpc.simulation import neutral_state, advance_plant


def rollout(config_path, output_path, duration_s=10.0, fps=30.0,
            human_torso_xyz=(-0.02, 1.35, 3.20), human_hand_tip_xyz=(-0.178, 1.05, 2.75)):
    c = MPCConfig.load(config_path)
    if not c.gain_schedule_enabled:
        raise ValueError("Use configs/elf3_gain_scheduled.json with gain_schedule_enabled=true")
    if abs(round(c.dt_s * fps) - c.dt_s * fps) > 1e-9:
        raise ValueError("Output FPS must give an integer samples per MPC time step")
    nsteps = int(round(duration_s / c.dt_s))
    if abs(nsteps * c.dt_s - duration_s) > 1e-8:
        raise ValueError("Duration must be an integer number of MPC steps")
    body = np.asarray(human_torso_xyz, dtype=float)
    hand = np.asarray(human_hand_tip_xyz, dtype=float)
    planner = HandshakeMPC(c)
    state = neutral_state(c)
    samples, gains, distances, root_goals, solver_info = [], [], [], [], []
    num_per_step = round(c.dt_s * fps)
    for k in range(nsteps):
        plan = planner.update(
            state,
            HandshakeTarget(
                timestamp_s=state.timestamp_s,
                hand_tip_position_m=hand,
                human_torso_position_m=body,
                frame_id=c.frame_id,
            ),
            PlannerCommand(initiated=(k == 0)),
        )
        d = plan.diagnostics
        solver_info.append({"t_s": float(state.timestamp_s), "solve_wall_time_s": d["solve_wall_time_s"],
                            "max_constraint_violation": d["max_constraint_violation"]})
        for j in range(num_per_step):
            samples.append(plan.sample(state.timestamp_s + j / fps))
            gains.append(d["reach_gain"])
            distances.append(d["human_torso_distance_m"])
            root_goals.append(d["pelvis_goal_xz_m"])
        state = advance_plant(state, plan, c.dt_s)
    # The last sample is the measured plant endpoint, not an extrapolated goal.
    samples.append({
        **plan.sample(state.timestamp_s),
        "pelvis_position_m": state.pelvis_position_m,
        "torso_position_m": state.pelvis_position_m + planner.body_rotation @ np.asarray(c.pelvis_to_torso_local_m),
        "right_wrist_position_m": state.right_wrist_position_m,
        "right_hand_tip_position_m": state.right_wrist_position_m + planner.tip_offset_world,
        "pelvis_velocity_mps": state.pelvis_velocity_mps,
        "right_wrist_velocity_mps": state.right_wrist_velocity_mps,
    })
    gains.append(gains[-1]); distances.append(distances[-1]); root_goals.append(root_goals[-1])
    fields = ("pelvis_position_m", "pelvis_orientation_xyzw", "torso_position_m",
              "torso_orientation_xyzw", "right_wrist_position_m", "right_wrist_orientation_xyzw",
              "right_hand_tip_position_m", "pelvis_velocity_mps", "right_wrist_velocity_mps")
    times = np.arange(len(samples)) / fps
    result = {
        "schema_version": "handshake_task_space/v1",
        "coordinate_convention": {
            "frame_id": c.frame_id, "up_axis": "Y", "initial_forward_axis": "+Z",
            "length_unit": "m", "time_unit": "s", "quaternion_order": "xyzw",
            "rotation_semantics": "active_local_to_world",
        },
        "time_origin_s": 0.0,
        "timestamps_s": times.tolist(),
        "relative_times_s": times.tolist(),
        "frame_indices": list(range(len(times))),
        "sample_rate_hz": float(fps),
        "geometry": {"wrist_to_tip_local_m": list(c.wrist_to_tip_local_m),
                     "pelvis_to_torso_local_m": list(c.pelvis_to_torso_local_m)},
        "trajectories": {field: [np.asarray(s[field]).tolist() for s in samples] for field in fields},
        "scheduling": {
            "reach_gain": gains,
            "human_torso_distance_m": distances,
            "pelvis_goal_xz_m": root_goals,
            "human_torso_position_m": body.tolist(),
            "hand_tip_target_m": hand.tolist(),
            "gain_gate_far_m": c.reach_gate_far_m,
            "gain_gate_near_m": c.reach_gate_near_m,
        },
        "diagnostics": {
            "simulation_only": True,
            "mpc_steps": nsteps,
            "final_pelvis_m": state.pelvis_position_m.tolist(),
            "final_wrist_m": state.right_wrist_position_m.tolist(),
            "max_constraint_violation": max(x["max_constraint_violation"] for x in solver_info),
            "mean_solve_wall_time_s": float(np.mean([x["solve_wall_time_s"] for x in solver_info])),
        },
        "adapter_required": True,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    cross = next((times[i] for i, g in enumerate(gains) if g >= 0.6), None)
    print(f"Saved {len(times)} frames / {duration_s:.1f}s to {output}")
    print(f"Hand gain>=0.6 first occurs at {cross} s (None=no reach this clip)")
    print(f"Pelvis final {state.pelvis_position_m.round(3)}, wrist final {state.right_wrist_position_m.round(3)}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "configs" / "elf3_gain_scheduled.json"))
    p.add_argument("--out", default=str(ROOT / "outputs" / "gain_scheduled" / "executed_reference.json"))
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--fps", type=float, default=30.0)
    a=p.parse_args()
    rollout(a.config, a.out, a.duration, a.fps)

if __name__ == "__main__":
    main()
