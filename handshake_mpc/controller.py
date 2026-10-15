"""Convex SOCP MPC: five translational coordinates; fixed orientations."""
from __future__ import annotations
import time
import numpy as np
import cvxpy as cp
from .config import MPCConfig
from .geometry import rotation, same_orientation, yaw_quaternion
from .interfaces import RobotState, HandshakeTarget, PlannerCommand, TaskSpacePlan


class MPCSolveError(RuntimeError):
    """No valid plan was produced. Caller must not execute a failed horizon."""


def smooth_reach_gain(distance_m: float, near_m: float, far_m: float) -> float:
    """C1-continuous 0 (far) -> 1 (near) distance schedule."""
    s = float(np.clip((far_m - distance_m) / (far_m - near_m), 0., 1.))
    return s * s * (3. - 2. * s)


def stopping_root_goal(pelvis_xz: np.ndarray, human_xz: np.ndarray, standoff_m: float) -> np.ndarray:
    """Project a stop point along the current robot-to-human approach line."""
    delta = human_xz - pelvis_xz
    norm = float(np.linalg.norm(delta))
    if norm < 1e-8:
        return pelvis_xz.copy()  # No well-defined approach bearing.
    return human_xz - standoff_m * delta / norm


def braking_path(position, velocity, acceleration_limit, dt, steps):
    """Straight-line bounded deceleration, then exact position hold."""
    p = np.empty((len(position), steps + 1))
    v = np.empty_like(p)
    p[:, 0], v[:, 0] = position, velocity
    for k in range(steps):
        speed = np.linalg.norm(v[:, k])
        a = -v[:, k] / dt if speed <= acceleration_limit * dt else -acceleration_limit * v[:, k] / speed
        p[:, k + 1] = p[:, k] + dt * v[:, k] + 0.5 * dt**2 * a
        v[:, k + 1] = v[:, k] + dt * a
    return p, v


class HandshakeMPC:
    def __init__(self, config: MPCConfig | None = None):
        self.config = config or MPCConfig()
        self.body_q = yaw_quaternion(self.config.fixed_heading_rad)
        self.body_rotation = rotation(self.body_q)
        self.wrist_q = np.asarray(self.config.fixed_wrist_orientation_xyzw, dtype=float)
        self.tip_offset_world = rotation(self.wrist_q) @ np.asarray(self.config.wrist_to_tip_local_m)
        self._command = "inactive"
        self._previous_acceleration = np.zeros(5)
        self._last_timestamp = None
        self._build_problem()

    def reset(self):
        """Clear command latch and acceleration history before a new episode."""
        self._command = "inactive"
        self._previous_acceleration[:] = 0
        self._last_timestamp = None

    def _build_problem(self):
        c, n = self.config, self.config.horizon_steps
        dt = c.dt_s
        self.P = cp.Variable((5, n + 1), name="position")
        self.V = cp.Variable((5, n + 1), name="velocity")
        self.A = cp.Variable((5, n), name="acceleration")
        self.slack = cp.Variable((3, n + 1), nonneg=True, name="comfort_violation_m")
        self.p0 = cp.Parameter(5)
        self.v0 = cp.Parameter(5)
        self.target_wrist = cp.Parameter((3, n + 1))
        self.body_anchor = cp.Parameter(2)
        self.position_lower = cp.Parameter((5, n + 1))
        self.position_upper = cp.Parameter((5, n + 1))
        self.previous_acceleration = cp.Parameter(5)
        # Weight-scaled residuals keep the model DPP, avoiding recompilation.
        self.sqrt_wrist_velocity_weight = cp.Parameter(nonneg=True)
        self.weighted_wrist_velocity_target = cp.Parameter((3, n + 1))
        self.sqrt_comfort_weight = cp.Parameter(nonneg=True)
        self.sqrt_hand_tracking_weight = cp.Parameter(nonneg=True)
        self.sqrt_hand_terminal_weight = cp.Parameter(nonneg=True)
        self.sqrt_neutral_wrist_weight = cp.Parameter(nonneg=True)
        self.sqrt_pelvis_goal_weight = cp.Parameter(nonneg=True)
        self.sqrt_pelvis_terminal_weight = cp.Parameter(nonneg=True)
        # Pre-multiplied targets prevent parameter*parameter products (DPP).
        self.weighted_hand_target = cp.Parameter((3, n + 1))
        self.weighted_hand_terminal_target = cp.Parameter(3)
        self.weighted_pelvis_goal = cp.Parameter(2)
        self.weighted_pelvis_terminal_goal = cp.Parameter(2)
        ground_lift = np.array([[1., 0.], [0., 0.], [0., 1.]])
        relative = self.P[2:, :] - ground_lift @ self.P[:2, :] - np.array([[0.], [c.pelvis_height_m], [0.]])
        local_relative = self.body_rotation.T @ relative
        delta_comfort = local_relative - np.asarray(c.comfort_center_local_m).reshape(3, 1)
        half = np.asarray(c.comfort_half_widths_m).reshape(3, 1)
        self.relative = relative
        relative_velocity = self.V[2:, :] - ground_lift @ self.V[:2, :]
        # P0, P1, P2 are quadratic Bezier control points for each interval.
        # Putting them all inside a convex reach ball bounds the entire arc.
        bezier_middle = relative[:, :-1] + 0.5 * dt * relative_velocity[:, :-1]
        self.bezier_middle = bezier_middle
        self.constraints = [
            self.P[:, 0] == self.p0,
            self.V[:, 0] == self.v0,
            self.P[:, 1:] == self.P[:, :-1] + dt * self.V[:, :-1] + 0.5 * dt**2 * self.A,
            self.V[:, 1:] == self.V[:, :-1] + dt * self.A,
            self.P >= self.position_lower, self.P <= self.position_upper,
            cp.norm(relative, 2, axis=0) <= c.max_pelvis_wrist_distance_m,
            cp.norm(bezier_middle, 2, axis=0) <= c.max_pelvis_wrist_distance_m,
            cp.norm(self.V[:2, :], 2, axis=0) <= c.pelvis_speed_max_mps,
            cp.norm(self.V[2:, :], 2, axis=0) <= c.wrist_speed_max_mps,
            cp.norm(self.A[:2, :], 2, axis=0) <= c.pelvis_acceleration_max_mps2,
            cp.norm(self.A[2:, :], 2, axis=0) <= c.wrist_acceleration_max_mps2,
            delta_comfort <= half + self.slack,
            -delta_comfort <= half + self.slack,
        ]
        neutral_offset_world = self.body_rotation @ np.asarray(c.neutral_wrist_local_m)
        neutral_error = relative - neutral_offset_world.reshape(3, 1)
        acceleration_change = cp.hstack([
            self.A[:, 0:1] - cp.reshape(self.previous_acceleration, (5, 1), order="F"),
            self.A[:, 1:] - self.A[:, :-1]])
        objective = (
            dt * cp.sum_squares(self.sqrt_hand_tracking_weight * self.P[2:, 1:]
                                 - self.weighted_hand_target[:, 1:])
            + cp.sum_squares(self.sqrt_hand_terminal_weight * self.P[2:, -1]
                             - self.weighted_hand_terminal_target)
            + dt * cp.sum_squares(self.sqrt_neutral_wrist_weight * neutral_error[:, 1:])
            + dt * cp.sum_squares(self.sqrt_pelvis_goal_weight * self.P[:2, 1:]
                                 - cp.reshape(self.weighted_pelvis_goal, (2, 1), order="F"))
            + cp.sum_squares(self.sqrt_pelvis_terminal_weight * self.P[:2, -1]
                             - self.weighted_pelvis_terminal_goal)
            + dt * cp.sum_squares(self.sqrt_comfort_weight * self.slack[:, 1:])
            + dt * c.pelvis_displacement_weight * cp.sum_squares(
                self.P[:2, 1:] - cp.reshape(self.body_anchor, (2, 1), order="F"))
            + dt * c.pelvis_velocity_weight * cp.sum_squares(self.V[:2, 1:])
            + dt * c.pelvis_acceleration_weight * cp.sum_squares(self.A[:2, :])
            + dt * c.wrist_acceleration_weight * cp.sum_squares(self.A[2:, :])
            + c.acceleration_change_weight * cp.sum_squares(acceleration_change)
            + dt * cp.sum_squares(self.sqrt_wrist_velocity_weight * self.V[2:, 1:]
                                 - self.weighted_wrist_velocity_target[:, 1:])
        )
        self.problem = cp.Problem(cp.Minimize(objective), self.constraints)
        if not self.problem.is_dcp() or not self.problem.is_dpp():
            raise RuntimeError("Internal error: MPC must be convex and parameter-cacheable")

    def _validate_state(self, state):
        c = self.config
        if state.frame_id != c.frame_id:
            raise ValueError("RobotState frame_id does not match planner frame")
        if self._last_timestamp is not None and state.timestamp_s < self._last_timestamp - 1e-9:
            raise ValueError("RobotState timestamp moved backwards; reset for a new episode")
        if abs(state.pelvis_position_m[1] - c.pelvis_height_m) > c.feasibility_tolerance:
            raise ValueError("Stage 1 requires the configured fixed pelvis height")
        if abs(state.pelvis_velocity_mps[1]) > c.feasibility_tolerance:
            raise ValueError("Stage 1 does not control pelvis vertical motion")
        if not same_orientation(state.pelvis_orientation_xyzw, self.body_q):
            raise ValueError("Stage 1 requires configured fixed upright pelvis heading")
        if not same_orientation(state.right_wrist_orientation_xyzw, self.wrist_q):
            raise ValueError("Stage 1 holds configured wrist orientation; angular control is not implemented")
        if np.linalg.norm(state.right_wrist_position_m - state.pelvis_position_m) > c.max_pelvis_wrist_distance_m + c.feasibility_tolerance:
            raise ValueError("Measured wrist already exceeds the hard reach bound")
        for velocity, limit in [(state.pelvis_velocity_mps, c.pelvis_speed_max_mps),
                                (state.right_wrist_velocity_mps, c.wrist_speed_max_mps)]:
            if np.linalg.norm(velocity) > limit + c.feasibility_tolerance:
                raise ValueError("Measured velocity already exceeds configured speed bound")

    def update(self, robot_state: RobotState, target: HandshakeTarget | None = None,
               command: PlannerCommand | None = None) -> TaskSpacePlan:
        """Replan from measured state. Apply only the first dt before replanning.

        Neither flag: continue the latched command. Termination wins ties.
        Before any event the controller brakes/holds. No internal task phases.
        """
        state, c = robot_state, self.config
        self._validate_state(state)
        command = command or PlannerCommand()
        if command.terminated:
            self._command = "terminated"
        elif command.initiated:
            self._command = "active"
        n, dt = c.horizon_steps, c.dt_s
        p0 = np.r_[state.pelvis_position_m[[0, 2]], state.right_wrist_position_m]
        v0 = np.r_[state.pelvis_velocity_mps[[0, 2]], state.right_wrist_velocity_mps]
        self.p0.value, self.v0.value = p0, v0
        self.body_anchor.value = p0[:2]
        self.previous_acceleration.value = self._previous_acceleration
        # Broad position bounds are replaced by equalities for hold/braking.
        travel = np.array([c.pelvis_speed_max_mps] * 2 + [c.wrist_speed_max_mps] * 3) * n * dt + 0.01
        lower = np.tile((p0 - travel)[:, None], (1, n + 1))
        upper = np.tile((p0 + travel)[:, None], (1, n + 1))
        target_velocity = np.zeros((3, n + 1))
        gate = 0.0
        reach_gain = 1.0
        human_distance = None
        root_goal = p0[:2].copy()
        neutral_weight = 0.0
        root_goal_weight = 0.0
        root_terminal_weight = 0.0
        hand_weight = c.hand_tracking_weight
        hand_terminal_weight = c.terminal_hand_tracking_weight
        if self._command == "active":
            if target is None:
                raise ValueError("An active handshake requires a fresh HandshakeTarget")
            if target.frame_id != c.frame_id:
                raise ValueError("HandshakeTarget frame_id does not match planner frame")
            age = state.timestamp_s - target.timestamp_s
            if age < -1e-6 or age > c.max_target_age_s:
                raise ValueError("Target timestamp is in the future or stale; no valid plan produced")
            if target.wrist_orientation_xyzw is not None and not same_orientation(target.wrist_orientation_xyzw, self.wrist_q):
                raise ValueError("Stage 1 cannot change wrist orientation; choose matching fixed configuration")
            if c.gain_schedule_enabled:
                if target.human_torso_position_m is None:
                    raise ValueError("Gain-scheduled MPC requires observed human_torso_position_m")
                human_xz = target.human_torso_position_m[[0, 2]]
                human_distance = float(np.linalg.norm(human_xz - p0[:2]))
                reach_gain = smooth_reach_gain(human_distance, c.reach_gate_near_m, c.reach_gate_far_m)
                root_goal = stopping_root_goal(p0[:2], human_xz, c.human_standoff_m)
                hand_weight *= reach_gain
                hand_terminal_weight *= reach_gain
                neutral_weight = (1. - reach_gain) * c.neutral_wrist_weight
                root_goal_weight = c.pelvis_goal_weight
                root_terminal_weight = c.terminal_pelvis_goal_weight
            if target.hand_tip_position_m is None:
                if not c.gain_schedule_enabled or reach_gain > 0:
                    raise ValueError("Hand-tip observation is required once the reach gain is positive")
                # The hand is unobserved during root-only walking. The zero-weight
                # target is a placeholder; the neutral-relative wrist cost applies.
                wrist_target = np.repeat(state.right_wrist_position_m[:, None], n+1, axis=1)
                distance = np.inf
                gate = 0.0
            else:
                lead = np.minimum(np.maximum(age, 0) + np.arange(n + 1) * dt, c.target_prediction_cap_s)
                target_tip = target.hand_tip_position_m[:, None] + target.hand_tip_velocity_mps[:, None] * lead
                wrist_target = target_tip - self.tip_offset_world[:, None]
                moving = (np.maximum(age, 0) + np.arange(n + 1) * dt) < c.target_prediction_cap_s
                target_velocity[:, moving] = target.hand_tip_velocity_mps[:, None]
                distance = np.linalg.norm(state.right_wrist_position_m + self.tip_offset_world - target_tip[:, 0])
                gate = float(np.exp(-0.5 * (distance / c.precontact_scale_m)**2))
            velocity_weight = c.wrist_velocity_far_weight + (c.wrist_velocity_near_weight - c.wrist_velocity_far_weight) * gate
            comfort_weight = c.comfort_weight
            if c.gain_schedule_enabled:
                comfort_weight *= reach_gain
                velocity_weight = c.wrist_velocity_far_weight + reach_gain * (
                    velocity_weight - c.wrist_velocity_far_weight)
                target_velocity *= reach_gain
        else:
            root_path, _ = braking_path(p0[:2], v0[:2], c.pelvis_acceleration_max_mps2, dt, n)
            lower[:2, :], upper[:2, :] = root_path, root_path
            pelvis_path = np.vstack([root_path[0], np.full(n + 1, c.pelvis_height_m), root_path[1]])
            if self._command == "terminated":
                wrist_target = pelvis_path + (self.body_rotation @ np.asarray(c.neutral_wrist_local_m))[:, None]
            else:
                wrist_path, _ = braking_path(p0[2:], v0[2:], c.wrist_acceleration_max_mps2, dt, n)
                lower[2:, :], upper[2:, :] = wrist_path, wrist_path
                wrist_target = wrist_path
            velocity_weight = c.wrist_velocity_near_weight
            comfort_weight = 0.0  # Neutral need not belong to the reaching region.
        self.position_lower.value, self.position_upper.value = lower, upper
        self.target_wrist.value = wrist_target
        self.sqrt_wrist_velocity_weight.value = np.sqrt(velocity_weight)
        self.weighted_wrist_velocity_target.value = np.sqrt(velocity_weight) * target_velocity
        self.sqrt_comfort_weight.value = np.sqrt(comfort_weight)
        self.sqrt_hand_tracking_weight.value = np.sqrt(hand_weight)
        self.sqrt_hand_terminal_weight.value = np.sqrt(hand_terminal_weight)
        self.sqrt_neutral_wrist_weight.value = np.sqrt(neutral_weight)
        self.sqrt_pelvis_goal_weight.value = np.sqrt(root_goal_weight)
        self.sqrt_pelvis_terminal_weight.value = np.sqrt(root_terminal_weight)
        self.weighted_hand_target.value = np.sqrt(hand_weight) * wrist_target
        self.weighted_hand_terminal_target.value = np.sqrt(hand_terminal_weight) * wrist_target[:, -1]
        self.weighted_pelvis_goal.value = np.sqrt(root_goal_weight) * root_goal
        self.weighted_pelvis_terminal_goal.value = np.sqrt(root_terminal_weight) * root_goal
        started = time.perf_counter()
        try:
            self.problem.solve(solver="CLARABEL", warm_start=True, max_iter=c.solver_max_iterations,
                               tol_gap_abs=1e-7, tol_gap_rel=1e-7, tol_feas=1e-8)
        except cp.error.SolverError as error:
            raise MPCSolveError(f"Clarabel failed: {error}") from error
        elapsed = time.perf_counter() - started
        if self.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise MPCSolveError(f"No feasible MPC plan: {self.problem.status}")
        max_violation = max(float(np.max(np.asarray(con.violation()))) for con in self.constraints)
        if not np.isfinite(max_violation) or max_violation > c.feasibility_tolerance:
            raise MPCSolveError(f"Rejected solver result: constraint residual {max_violation:g}")
        p, v, a = self.P.value.T.copy(), self.V.value.T.copy(), self.A.value.T.copy()
        if not all(np.all(np.isfinite(array)) for array in (p, v, a)):
            raise MPCSolveError("Rejected nonfinite solution")
        self._previous_acceleration = a[0].copy()
        self._last_timestamp = state.timestamp_s
        pelvis = np.column_stack([p[:, 0], np.full(n + 1, c.pelvis_height_m), p[:, 1]])
        zeros = np.zeros(n + 1)
        pelvis_v = np.column_stack([v[:, 0], zeros, v[:, 1]])
        pelvis_a = np.column_stack([a[:, 0], np.zeros(n), a[:, 1]])
        torso = pelvis + self.body_rotation @ np.asarray(c.pelvis_to_torso_local_m)
        diagnostics = {
            "command": self._command, "solver_status": self.problem.status,
            "solve_wall_time_s": float(elapsed), "solver_time_s": self.problem.solver_stats.solve_time,
            "iterations": self.problem.solver_stats.num_iters,
            "objective": float(self.problem.value), "max_constraint_violation": max_violation,
            "precontact_gate": gate, "wrist_velocity_weight": float(velocity_weight),
            "gain_schedule_enabled": c.gain_schedule_enabled,
            "reach_gain": float(reach_gain if self._command == "active" else 0.0),
            "human_torso_distance_m": human_distance,
            "pelvis_goal_xz_m": root_goal.tolist(),
            "max_predicted_pelvis_wrist_distance_m": float(np.max(np.linalg.norm(self.relative.value, axis=0))),
            "max_reach_bezier_control_distance_m": float(np.max(np.linalg.norm(self.bezier_middle.value, axis=0))),
            "max_comfort_violation_m": float(np.max(np.maximum(
                np.abs(self.body_rotation.T @ self.relative.value - np.asarray(c.comfort_center_local_m)[:, None])
                - np.asarray(c.comfort_half_widths_m)[:, None], 0))),
        }
        return TaskSpacePlan(
            state.timestamp_s + np.arange(n + 1) * dt,
            pelvis, np.tile(self.body_q, (n + 1, 1)), torso, np.tile(self.body_q, (n + 1, 1)),
            p[:, 2:], np.tile(self.wrist_q, (n + 1, 1)), p[:, 2:] + self.tip_offset_world,
            pelvis_v, v[:, 2:], pelvis_a, a[:, 2:], diagnostics, c.frame_id,
            np.asarray(c.wrist_to_tip_local_m), np.asarray(c.pelvis_to_torso_local_m))
