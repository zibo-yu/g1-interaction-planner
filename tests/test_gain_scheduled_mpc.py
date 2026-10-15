"""Convex controller tests. Run in Python 3.11+ with CVXPY/Clarabel installed."""
import unittest
from pathlib import Path
import numpy as np
from handshake_mpc import HandshakeMPC, MPCConfig, HandshakeTarget, PlannerCommand
from handshake_mpc.simulation import neutral_state, advance_plant

CONFIG=Path(__file__).resolve().parents[1]/'configs'/'elf3_gain_scheduled.json'


class GainScheduledMPCTests(unittest.TestCase):
    def setUp(self):
        self.c=MPCConfig.load(CONFIG)
        self.m=HandshakeMPC(self.c)

    def target(self, timestamp=0., torso_z=3.2):
        return HandshakeTarget(timestamp, [-.178,1.05,torso_z-.45],
                               human_torso_position_m=[0.,1.35,torso_z])

    def test_parameter_cacheable_and_walking_objective(self):
        self.assertTrue(self.m.problem.is_dcp())
        self.assertTrue(self.m.problem.is_dpp())
        state=neutral_state(self.c)
        plan=self.m.update(state,self.target(),PlannerCommand(initiated=True))
        self.assertAlmostEqual(plan.diagnostics['reach_gain'],0.,places=8)
        self.assertAlmostEqual(plan.diagnostics['pelvis_goal_xz_m'][1],2.35,places=6)
        self.assertGreater(plan.pelvis_positions_m[-1,2],.1)
        # The walking controller holds wrist offset relative to moving pelvis.
        rel=plan.right_wrist_positions_m[-1]-plan.pelvis_positions_m[-1]
        np.testing.assert_allclose(rel,self.c.neutral_wrist_local_m,atol=.025)
        self.assertLessEqual(plan.diagnostics['max_constraint_violation'],self.c.feasibility_tolerance)

    def test_smooth_gain_and_close_reaching(self):
        from handshake_mpc.controller import smooth_reach_gain
        self.assertAlmostEqual(smooth_reach_gain(2.,1.2,2.),0.)
        self.assertAlmostEqual(smooth_reach_gain(1.6,1.2,2.),.5)
        self.assertAlmostEqual(smooth_reach_gain(1.2,1.2,2.),1.)
        state=neutral_state(self.c,pelvis_xz=(0.,2.15))
        plan=self.m.update(state,self.target(),PlannerCommand(initiated=True))
        self.assertAlmostEqual(plan.diagnostics['reach_gain'],1.,places=7)
        self.assertLess(np.linalg.norm(plan.right_hand_tip_positions_m[-1]-[-.178,1.05,2.75]),.12)

    def test_far_walking_needs_only_torso_observation(self):
        state=neutral_state(self.c)
        obs=HandshakeTarget(0.,None,human_torso_position_m=[0.,1.35,3.2])
        plan=self.m.update(state,obs,PlannerCommand(initiated=True))
        self.assertEqual(plan.diagnostics['reach_gain'],0.)
        self.assertGreater(plan.pelvis_positions_m[-1,2],.1)

    def test_near_reaching_needs_hand_observation(self):
        state=neutral_state(self.c,pelvis_xz=(0,2.15))
        obs=HandshakeTarget(0.,None,human_torso_position_m=[0.,1.35,3.2])
        with self.assertRaisesRegex(ValueError,'Hand-tip observation'):
            self.m.update(state,obs,PlannerCommand(initiated=True))

    def test_requires_torso_observation(self):
        state=neutral_state(self.c)
        with self.assertRaisesRegex(ValueError,'human_torso_position_m'):
            self.m.update(state,HandshakeTarget(0.,[-.178,1.05,2.75]),PlannerCommand(initiated=True))

    def test_gain_updates_from_measured_position(self):
        state=neutral_state(self.c)
        gains=[]
        for k in range(6):
            plan=self.m.update(state,self.target(state.timestamp_s),PlannerCommand(initiated=k==0))
            gains.append(plan.diagnostics['reach_gain'])
            state=advance_plant(state,plan,self.c.dt_s)
        self.assertEqual(gains[0],0.)
        self.assertTrue(all(0<=x<=1 for x in gains))
        self.assertGreater(state.pelvis_position_m[2],0.)


if __name__=='__main__':unittest.main()
