"""CLI experiments with an ideal or mismatched synthetic feedback plant."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import numpy as np
import cvxpy
import clarabel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import MPCConfig
from .controller import HandshakeMPC
from .interfaces import HandshakeTarget, PlannerCommand
from .simulation import neutral_state, advance_plant

SCENARIOS = ("near", "far", "moving", "terminate", "moving_terminate", "mismatch", "unreachable")


def target_at(name, t):
    if name == "near":
        return np.array([0.22, 1.15, 0.60]), np.zeros(3)
    if name == "moving":
        return (np.array([0.22 + 0.10 * np.sin(0.7 * t), 1.15, 1.50 + 0.08 * np.sin(0.4 * t)]),
                np.array([0.07 * np.cos(0.7 * t), 0., 0.032 * np.cos(0.4 * t)]))
    if name == "unreachable":
        return np.array([0.22, 2.20, 0.60]), np.zeros(3)
    return np.array([0.22, 1.15, 2.20]), np.zeros(3)


def save_csv(path, samples):
    keys = ["pelvis_position_m", "torso_position_m", "right_wrist_position_m",
            "right_hand_tip_position_m", "pelvis_velocity_mps", "right_wrist_velocity_mps",
            "pelvis_orientation_xyzw", "torso_orientation_xyzw", "right_wrist_orientation_xyzw"]
    headings = ["timestamp_s"]
    for key in keys:
        headings += [key + "_" + axis for axis in ("xyzw" if "orientation" in key else "xyz")]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headings)
        for sample in samples:
            writer.writerow([sample["timestamp_s"]] + [float(v) for key in keys for v in sample[key]])


def plot_results(path, name, records, metrics, termination_time):
    t = np.array([r["timestamp_s"] for r in records])
    b = np.array([r["pelvis_position_m"] for r in records])
    h = np.array([r["right_hand_tip_position_m"] for r in records])
    target = np.array([r["evaluation_target_m"] for r in records])
    error = np.linalg.norm(h - target, axis=1)
    vb = np.array([np.linalg.norm(r["pelvis_velocity_mps"]) for r in records])
    vh = np.array([np.linalg.norm(r["right_wrist_velocity_mps"]) for r in records])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    ax = axes[0, 0]
    ax.plot(b[:, 2], b[:, 0], color="#245a94", label="Pelvis")
    ax.plot(h[:, 2], h[:, 0], color="#bf6331", label="Hand tip")
    ax.scatter(target[0, 2], target[0, 0], color="#3b8a68", marker="x", label="Initial tip target", s=70)
    ax.set(xlabel="Forward Z (m)", ylabel="Lateral X (m)", title="Measured synthetic paths (top view)")
    ax.legend()
    ax = axes[0, 1]
    ax.plot(t, b[:, 2], color="#245a94", label="Pelvis")
    ax.plot(t, h[:, 2], color="#bf6331", label="Hand tip")
    ax.plot(t, target[:, 2], "--", color="#3b8a68", label="Current objective")
    ax.set(xlabel="Time (s)", ylabel="Forward Z (m)", title="Body approach and arm extension")
    ax.legend()
    ax = axes[1, 0]
    ax.plot(t, error, color="#7b518f")
    ax.set(xlabel="Time (s)", ylabel="Tip/objective error (m)", title="Tracking / retraction error")
    ax = axes[1, 1]
    ax.plot(t, vb, color="#245a94", label="Pelvis speed")
    ax.plot(t, vh, color="#bf6331", label="Wrist speed")
    ax.set(xlabel="Time (s)", ylabel="Speed (m/s)", title="Measured synthetic velocities")
    ax.legend()
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        if termination_time is not None and ax != axes[0, 0]:
            ax.axvline(termination_time, color="#555555", linestyle=":", alpha=0.8)
    fig.suptitle(f"Stage 1 task-space MPC: {name} | final error {metrics['final_objective_error_m']:.4f} m\n"
                 "Synthetic kinematic feedback only; fixed body and wrist orientations", fontsize=14)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_scenario(name, config, output_root, sample_rate_hz=50., record_horizons=False):
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}")
    c = config
    if abs(c.dt_s * sample_rate_hz - round(c.dt_s * sample_rate_hz)) > 1e-8:
        raise ValueError("Demo export rate must contain an integer number of samples per MPC interval")
    total_s = 16. if name == "moving" else 12.
    termination_time = 8. if name == "terminate" else (1.5 if name == "moving_terminate" else None)
    count = int(round(total_s / c.dt_s))
    if termination_time is not None and abs(termination_time / c.dt_s - round(termination_time / c.dt_s)) > 1e-8:
        raise ValueError("Termination time must align with the MPC interval for this demo")
    out = Path(output_root) / name
    out.mkdir(parents=True, exist_ok=True)
    planner = HandshakeMPC(c)
    state = neutral_state(c)
    measured, executed, diagnostics = [], [], []
    initial_pelvis = state.pelvis_position_m.copy()
    termination_pelvis = None
    scale = 0.8 if name == "mismatch" else 1.0
    horizon_file = (out / "predicted_horizons.jsonl").open("w", encoding="utf-8") if record_horizons else None
    last_plan = None
    try:
        for k in range(count):
            tip_target, target_velocity = target_at(name, state.timestamp_s)
            event_terminate = termination_time is not None and k == round(termination_time / c.dt_s)
            if event_terminate:
                termination_pelvis = state.pelvis_position_m.copy()
            plan = planner.update(state, HandshakeTarget(state.timestamp_s, tip_target, target_velocity),
                                  PlannerCommand(initiated=k == 0, terminated=event_terminate))
            last_plan = plan
            sample = plan.sample(state.timestamp_s)
            if plan.diagnostics["command"] == "terminated":
                evaluation_target = state.pelvis_position_m + planner.body_rotation @ np.asarray(c.neutral_wrist_local_m) + planner.tip_offset_world
            else:
                evaluation_target = tip_target
            measured.append({**sample, "evaluation_target_m": evaluation_target})
            diagnostics.append(plan.diagnostics)
            if horizon_file is not None:
                horizon_file.write(json.dumps(plan.to_payload(), allow_nan=False) + "\n")
            for j in range(round(c.dt_s * sample_rate_hz)):
                executed.append(plan.sample(state.timestamp_s + j / sample_rate_hz))
            state = advance_plant(state, plan, c.dt_s, acceleration_scale=scale)
    finally:
        if horizon_file is not None:
            horizon_file.close()
    # Endpoint is from the actual synthetic plant, not the old desired horizon.
    final = last_plan.sample(state.timestamp_s)
    final.update({"pelvis_position_m": state.pelvis_position_m,
                  "right_wrist_position_m": state.right_wrist_position_m,
                  "right_hand_tip_position_m": state.right_wrist_position_m + planner.tip_offset_world,
                  "torso_position_m": state.pelvis_position_m + planner.body_rotation @ np.asarray(c.pelvis_to_torso_local_m),
                  "pelvis_velocity_mps": state.pelvis_velocity_mps,
                  "right_wrist_velocity_mps": state.right_wrist_velocity_mps})
    final_target = target_at(name, state.timestamp_s)[0] if termination_time is None else (
        state.pelvis_position_m + planner.body_rotation @ np.asarray(c.neutral_wrist_local_m) + planner.tip_offset_world)
    measured.append({**final, "evaluation_target_m": final_target})
    # Export the reference endpoint separately from the measured endpoint.
    executed.append(last_plan.sample(state.timestamp_s))
    pelvis = np.array([s["pelvis_position_m"] for s in measured])
    wrist = np.array([s["right_wrist_position_m"] for s in measured])
    times = np.array([s["timestamp_s"] for s in measured])
    solve_times = np.array([d["solve_wall_time_s"] for d in diagnostics])
    reach = np.linalg.norm(wrist - pelvis, axis=1)
    metrics = {
        "scenario": name, "duration_s": state.timestamp_s, "mpc_updates": count,
        "plant_acceleration_scale": scale, "terminated_at_s": termination_time,
        "final_objective_error_m": float(np.linalg.norm(final["right_hand_tip_position_m"] - final_target)),
        "final_pelvis_displacement_m": float(np.linalg.norm(pelvis[-1] - initial_pelvis)),
        "max_measured_pelvis_wrist_distance_m": float(np.max(reach)),
        "max_predicted_constraint_violation": max(d["max_constraint_violation"] for d in diagnostics),
        "max_pelvis_speed_mps": max(float(np.linalg.norm(s["pelvis_velocity_mps"])) for s in measured),
        "max_wrist_speed_mps": max(float(np.linalg.norm(s["right_wrist_velocity_mps"])) for s in measured),
        "solve_wall_ms_first": float(solve_times[0] * 1000),
        "solve_wall_ms_median_after_first": float(np.median(solve_times[1:]) * 1000),
        "solve_wall_ms_p95_after_first": float(np.percentile(solve_times[1:], 95) * 1000),
        "solver_status_counts": {s: sum(d["solver_status"] == s for d in diagnostics)
                                 for s in sorted({d["solver_status"] for d in diagnostics})},
    }
    if termination_time is not None:
        post = pelvis[times >= termination_time - 1e-7]
        metrics["pelvis_movement_after_termination_m"] = float(np.max(np.linalg.norm(post - termination_pelvis, axis=1)))
        metrics["final_pelvis_speed_mps"] = float(np.linalg.norm(state.pelvis_velocity_mps))
    # Automated behavioral checks distinguish an intentionally unreachable target.
    checks = {
        "hard_constraints": metrics["max_predicted_constraint_violation"] <= c.feasibility_tolerance,
        "measured_reach": metrics["max_measured_pelvis_wrist_distance_m"] <= c.max_pelvis_wrist_distance_m + 1e-4,
        "objective_behavior": metrics["final_objective_error_m"] < (0.06 if name == "moving" else 0.02)
                              if name != "unreachable" else metrics["final_objective_error_m"] > 0.3,
    }
    if name == "near":
        checks["little_pelvis_motion"] = metrics["final_pelvis_displacement_m"] < 0.06
    if name in ("far", "terminate", "mismatch"):
        checks["body_approaches"] = float(pelvis[:, 2].max()) > 1.0
    if name == "terminate":
        checks["stationary_root_stays_approximately_fixed"] = metrics["pelvis_movement_after_termination_m"] < 0.005
    if name == "moving_terminate":
        checks["moving_root_brakes"] = metrics["final_pelvis_speed_mps"] < 1e-4
    metrics["behavior_checks"] = checks
    metrics["all_checks_pass"] = all(checks.values())
    payload = last_plan.to_payload()
    payload.update({"time_origin_s": float(executed[0]["timestamp_s"]),
                    "timestamps_s": [s["timestamp_s"] for s in executed],
                    "relative_times_s": [s["timestamp_s"] - executed[0]["timestamp_s"] for s in executed],
                    "frame_indices": list(range(len(executed))),
                    "sample_rate_hz": sample_rate_hz,
                    "trajectories": {key: [s[key].tolist() for s in executed]
                                     for key in payload["trajectories"]},
                    "diagnostics": {"scenario": name, "kind": "executed_first_interval_references", "metrics": metrics}})
    (out / "executed_reference.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save_csv(out / "executed_reference.csv", executed)
    save_csv(out / "measured_state.csv", measured)
    last_plan.save_json(out / "last_predicted_horizon.json", sample_rate_hz)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    plot_results(out / "trajectory.png", name, measured, metrics, termination_time)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--sample-rate", type=float, default=50.)
    parser.add_argument("--record-horizons", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.sample_rate) or args.sample_rate <= 0:
        parser.error("--sample-rate must be finite and positive")
    config = MPCConfig.load(args.config) if args.config else MPCConfig()
    args.output.mkdir(parents=True, exist_ok=True)
    config.save(args.output / "config_used.json")
    names = SCENARIOS if args.scenario == "all" else (args.scenario,)
    results = []
    for name in names:
        result = run_scenario(name, config, args.output, args.sample_rate, args.record_horizons)
        results.append(result)
        print(f"{name:18s} {'PASS' if result['all_checks_pass'] else 'FAIL'} | "
              f"error={result['final_objective_error_m']:.4f} m | "
              f"pelvis travel={result['final_pelvis_displacement_m']:.3f} m | "
              f"median solve={result['solve_wall_ms_median_after_first']:.1f} ms", flush=True)
    report = {"environment": {"python": sys.version, "platform": platform.platform(),
                              "numpy": np.__version__, "cvxpy": cvxpy.__version__, "clarabel": clarabel.__version__},
              "all_checks_pass": all(r["all_checks_pass"] for r in results), "scenarios": results}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Outputs: {args.output.resolve()}")
    return 0 if report["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
