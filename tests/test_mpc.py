import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
from handshake_mpc import (MPCConfig, RobotState, HandshakeTarget, PlannerCommand, HandshakeMPC, MPCSolveError)
from handshake_mpc.geometry import tip_from_wrist, rotation, yaw_quaternion
from handshake_mpc.simulation import neutral_state, advance_plant


class GeometryAndContractTests(unittest.TestCase):
    def test_tip_offset_rotates_with_wrist(self):
        q = yaw_quaternion(np.pi / 2)
        np.testing.assert_allclose(tip_from_wrist([1, 2, 3], q, [0, 0, .12]), [1.12, 2, 3], atol=1e-12)

    def test_palm_mapping_has_explicit_geometry(self):
        t = HandshakeTarget.from_human_palm(timestamp_s=0, human_palm_position_m=[1, 2, 3],
            human_palm_orientation_xyzw=yaw_quaternion(np.pi / 2), palm_to_robot_tip_local_m=[0, 0, -.02])
        np.testing.assert_allclose(t.hand_tip_position_m, [.98, 2, 3], atol=1e-12)

    def test_invalid_quaternion_is_rejected(self):
        with self.assertRaises(ValueError):
            RobotState(0, [0, .95, 0], [0, 0, 0, 0], [0, 1, 0], [0, 0, 0, 1])

    def test_nan_target_is_rejected(self):
        with self.assertRaises(ValueError):
            HandshakeTarget(0, [np.nan, 1, 2])

    def test_configuration_round_trip(self):
        config = replace(MPCConfig(), wrist_to_tip_local_m=(.01, .02, .13))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            config.save(path)
            loaded = MPCConfig.load(path)
        self.assertEqual(loaded.dt_s, config.dt_s)
        np.testing.assert_allclose(loaded.wrist_to_tip_local_m, config.wrist_to_tip_local_m)

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            replace(MPCConfig(), comfort_half_widths_m=(0, .2, .3))
        with self.assertRaises(ValueError):
            replace(MPCConfig(), neutral_wrist_local_m=(0, 0, 2))


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.config = MPCConfig()
        self.controller = HandshakeMPC(self.config)
        self.state = neutral_state(self.config)

    def target(self, position=(.22, 1.15, .60), timestamp=0):
        return HandshakeTarget(timestamp, np.array(position))

    def active_plan(self):
        return self.controller.update(self.state, self.target(), PlannerCommand(initiated=True))

    def test_problem_is_convex_and_parameter_cacheable(self):
        self.assertTrue(self.controller.problem.is_dcp())
        self.assertTrue(self.controller.problem.is_dpp())

    def test_initial_state_is_measured_state(self):
        self.state.right_wrist_position_m += [.01, .02, .01]
        self.state.right_wrist_velocity_mps = np.array([.02, 0, .03])
        plan = self.active_plan()
        np.testing.assert_allclose(plan.right_wrist_positions_m[0], self.state.right_wrist_position_m, atol=1e-7)
        np.testing.assert_allclose(plan.right_wrist_velocities_mps[0], self.state.right_wrist_velocity_mps, atol=1e-7)

    def test_hard_bounds_hold_on_entire_interpolated_horizon(self):
        plan = self.controller.update(self.state, self.target((.22, 1.15, 2.2)), PlannerCommand(initiated=True))
        c = self.config
        for t in np.linspace(0, c.dt_s * c.horizon_steps, 501):
            ref = plan.sample(t)
            self.assertLessEqual(np.linalg.norm(ref["right_wrist_position_m"] - ref["pelvis_position_m"]),
                                 c.max_pelvis_wrist_distance_m + 1e-6)
            self.assertLessEqual(np.linalg.norm(ref["pelvis_velocity_mps"]), c.pelvis_speed_max_mps + 1e-6)
            self.assertLessEqual(np.linalg.norm(ref["right_wrist_velocity_mps"]), c.wrist_speed_max_mps + 1e-6)
        self.assertLessEqual(np.max(np.linalg.norm(plan.pelvis_accelerations_mps2, axis=1)), c.pelvis_acceleration_max_mps2 + 1e-6)
        self.assertLessEqual(np.max(np.linalg.norm(plan.right_wrist_accelerations_mps2, axis=1)), c.wrist_acceleration_max_mps2 + 1e-6)

    def test_sampling_matches_double_integrator_and_endpoint(self):
        plan = self.active_plan()
        tau = .04
        sample = plan.sample(tau)
        expected = plan.right_wrist_positions_m[0] + tau * plan.right_wrist_velocities_mps[0] + .5 * tau**2 * plan.right_wrist_accelerations_mps2[0]
        np.testing.assert_allclose(sample["right_wrist_position_m"], expected, atol=1e-10)
        endpoint = plan.sample(plan.timestamps_s[-1])
        np.testing.assert_allclose(endpoint["right_wrist_position_m"], plan.right_wrist_positions_m[-1], atol=1e-7)
        with self.assertRaises(ValueError):
            plan.sample(-1)
        with self.assertRaises(ValueError):
            plan.sample(10)

    def test_all_requested_pose_channels_and_geometry_are_exported(self):
        c = replace(self.config, fixed_heading_rad=.3, fixed_wrist_orientation_xyzw=tuple(yaw_quaternion(.6)),
                    pelvis_to_torso_local_m=(.03, .25, .02))
        m = HandshakeMPC(c)
        plan = m.update(neutral_state(c), self.target(), PlannerCommand(initiated=True))
        payload = plan.to_payload(sample_rate_hz=50)
        self.assertEqual(payload["schema_version"], "handshake_task_space/v1")
        self.assertTrue(payload["adapter_required"])
        for name in ("right_hand_tip_position_m", "right_wrist_position_m", "right_wrist_orientation_xyzw",
                     "torso_position_m", "torso_orientation_xyzw"):
            self.assertEqual(len(payload["trajectories"][name]), 151)
        wrist = np.array(payload["trajectories"]["right_wrist_position_m"])
        tip = np.array(payload["trajectories"]["right_hand_tip_position_m"])
        pelvis = np.array(payload["trajectories"]["pelvis_position_m"])
        torso = np.array(payload["trajectories"]["torso_position_m"])
        np.testing.assert_allclose(tip - wrist, np.tile(rotation(c.fixed_wrist_orientation_xyzw) @ c.wrist_to_tip_local_m, (151, 1)), atol=1e-8)
        np.testing.assert_allclose(torso - pelvis, np.tile(m.body_rotation @ c.pelvis_to_torso_local_m, (151, 1)), atol=1e-8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plan.json"
            plan.save_json(path, 50)
            reloaded = json.loads(path.read_text())
        self.assertEqual(reloaded["coordinate_convention"]["quaternion_order"], "xyzw")

    def test_inactive_holds_stationary_robot(self):
        plan = self.controller.update(self.state)
        np.testing.assert_allclose(plan.pelvis_positions_m, np.tile(self.state.pelvis_position_m, (31, 1)), atol=1e-7)
        np.testing.assert_allclose(plan.right_wrist_positions_m, np.tile(self.state.right_wrist_position_m, (31, 1)), atol=1e-7)

    def test_event_latches_and_termination_has_priority(self):
        first = self.active_plan()
        state = advance_plant(self.state, first, .1)
        second = self.controller.update(state, self.target(timestamp=.1))
        self.assertEqual(second.diagnostics["command"], "active")
        stop = self.controller.update(state, command=PlannerCommand(initiated=True, terminated=True))
        self.assertEqual(stop.diagnostics["command"], "terminated")
        self.assertEqual(self.controller.update(state).diagnostics["command"], "terminated")
        self.controller.reset()
        self.assertEqual(self.controller.update(self.state).diagnostics["command"], "inactive")

    def test_stationary_termination_locks_pelvis_and_retracts(self):
        self.state.right_wrist_position_m = np.array([.22, 1.15, .55])
        plan = self.controller.update(self.state, command=PlannerCommand(terminated=True))
        np.testing.assert_allclose(plan.pelvis_positions_m, np.tile(self.state.pelvis_position_m, (31, 1)), atol=1e-7)
        neutral = neutral_state(self.config).right_wrist_position_m
        self.assertLess(np.linalg.norm(plan.right_wrist_positions_m[-1] - neutral), .01)

    def test_moving_termination_brakes_without_instantaneous_freeze(self):
        self.state.pelvis_velocity_mps = np.array([0., 0., .3])
        plan = self.controller.update(self.state, command=PlannerCommand(terminated=True))
        self.assertGreater(plan.pelvis_positions_m[-1, 2], 0)
        self.assertLess(plan.pelvis_positions_m[-1, 2], .09)
        self.assertLess(np.linalg.norm(plan.pelvis_velocities_mps[-1]), 1e-6)
        np.testing.assert_allclose(plan.pelvis_positions_m[10:], np.tile(plan.pelvis_positions_m[-1], (21, 1)), atol=1e-7)

    def test_near_target_increases_velocity_penalty_continuously(self):
        tip = self.state.right_wrist_position_m + self.controller.tip_offset_world
        near = self.controller.update(self.state, self.target(tip + [0, 0, .03]), PlannerCommand(initiated=True))
        far = self.controller.update(self.state, self.target(tip + [0, 0, 1]))
        self.assertGreater(near.diagnostics["wrist_velocity_weight"], far.diagnostics["wrist_velocity_weight"] * 10)
        self.assertGreater(near.diagnostics["precontact_gate"], .9)

    def test_stale_future_and_wrong_frame_inputs_are_rejected(self):
        for target in [self.target(timestamp=-1), self.target(timestamp=1),
                       HandshakeTarget(0, [0, 1, 1], frame_id="camera")]:
            with self.assertRaises(ValueError):
                self.controller.update(self.state, target, PlannerCommand(initiated=True))
        state = neutral_state(self.config)
        state.frame_id = "camera"
        with self.assertRaises(ValueError):
            self.controller.update(state)

    def test_active_requires_target_and_rejects_angular_motion(self):
        with self.assertRaises(ValueError):
            self.controller.update(self.state, command=PlannerCommand(initiated=True))
        target = self.target()
        target.wrist_orientation_xyzw = yaw_quaternion(.8)
        with self.assertRaises(ValueError):
            self.controller.update(self.state, target)

    def test_outside_initial_reach_and_overspeed_are_rejected(self):
        self.state.right_wrist_position_m = self.state.pelvis_position_m + [0, 0, 2]
        with self.assertRaises(ValueError):
            self.active_plan()
        self.state = neutral_state(self.config)
        self.state.right_wrist_velocity_mps = np.array([0., 0., 2.])
        with self.assertRaises(ValueError):
            self.active_plan()

    def test_nonviable_outward_motion_fails_instead_of_exporting_bad_plan(self):
        self.state.right_wrist_position_m = self.state.pelvis_position_m + np.array([0., 0., .8])
        self.state.right_wrist_velocity_mps = np.array([0., 0., .2])
        with self.assertRaises(MPCSolveError):
            self.active_plan()


class ClosedLoopTests(unittest.TestCase):
    def rollout(self, target_position, scale=1.0, seconds=10):
        c = MPCConfig()
        m = HandshakeMPC(c)
        state = neutral_state(c)
        for step in range(round(seconds / c.dt_s)):
            plan = m.update(state, HandshakeTarget(state.timestamp_s, np.array(target_position)),
                            PlannerCommand(initiated=step == 0))
            state = advance_plant(state, plan, c.dt_s, scale)
        return state, m

    def test_near_target_reaches_without_substantial_pelvis_motion(self):
        state, m = self.rollout([.22, 1.15, .6], seconds=8)
        self.assertLess(np.linalg.norm(state.right_wrist_position_m + m.tip_offset_world - [.22, 1.15, .6]), .005)
        self.assertLess(np.linalg.norm(state.pelvis_position_m[[0, 2]]), .06)

    def test_distant_target_drives_body_and_hand(self):
        state, m = self.rollout([.22, 1.15, 2.2])
        self.assertGreater(state.pelvis_position_m[2], 1.0)
        self.assertGreater((state.right_wrist_position_m - state.pelvis_position_m)[2], .4)
        self.assertLess(np.linalg.norm(state.right_wrist_position_m + m.tip_offset_world - [.22, 1.15, 2.2]), .005)

    def test_replanning_corrects_slower_plant(self):
        state, m = self.rollout([.22, 1.15, 2.2], scale=.8, seconds=12)
        self.assertLess(np.linalg.norm(state.right_wrist_position_m + m.tip_offset_world - [.22, 1.15, 2.2]), .02)

    def test_infeasible_target_reports_remaining_error_without_reach_violation(self):
        state, m = self.rollout([.22, 2.2, .6], seconds=8)
        self.assertGreater(np.linalg.norm(state.right_wrist_position_m + m.tip_offset_world - [.22, 2.2, .6]), .3)
        self.assertLessEqual(np.linalg.norm(state.right_wrist_position_m - state.pelvis_position_m), .80001)


if __name__ == "__main__":
    unittest.main()
