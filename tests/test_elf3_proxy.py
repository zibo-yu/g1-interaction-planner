"""Kinematic file comparisons and proxy calibration tests; no MPC solver needed."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from elf3_proxy_calibration import load_urdf, kinematic_signature, fk, derive, ELF_TO_PLANNER
import json


class TestELF3Proxy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a=load_urdf(ROOT/'models/elf3.urdf')
        cls.b=load_urdf(ROOT/'models/elf3_hand_mesh.urdf')
        cls.c,cls.r=derive(cls.a,json.loads((ROOT/'configs/default.json').read_text()))

    def test_kinematics_identical(self):
        self.assertEqual(kinematic_signature(self.a),kinematic_signature(self.b))
        self.assertEqual(len(self.a['joints']),35)
        self.assertEqual(sum(j['type']=='revolute' for j in self.a['joints'].values()),29)

    def test_proper_coordinate_rotation(self):
        np.testing.assert_allclose(ELF_TO_PLANNER@ELF_TO_PLANNER.T,np.eye(3))
        self.assertAlmostEqual(np.linalg.det(ELF_TO_PLANNER),1.0)
        np.testing.assert_allclose(ELF_TO_PLANNER@np.array([1,0,0]),[0,0,1])
        np.testing.assert_allclose(ELF_TO_PLANNER@np.array([0,0,1]),[0,1,0])

    def test_virtual_hip_root_and_torso_offset(self):
        np.testing.assert_allclose(self.r['urdf_derived']['virtual_pelvis_in_torso_frame_elf3_m'],[0,0,-.3825])
        np.testing.assert_allclose(self.c['pelvis_to_torso_local_m'],[0,.3825,0])

    def test_zero_wrist_fk(self):
        np.testing.assert_allclose(self.c['neutral_wrist_local_m'],[-.178,.2135,.256],atol=1e-7)
        np.testing.assert_allclose(self.r['urdf_derived']['right_shoulder_relative_to_pelvis_planner_m'],[-.178,.4695,0],atol=1e-7)

    def test_arm_lengths_and_floor_proxy(self):
        self.assertAlmostEqual(self.r['urdf_derived']['upper_arm_segment_m'],.256)
        self.assertAlmostEqual(self.r['urdf_derived']['forearm_segment_m'],.256)
        self.assertAlmostEqual(self.c['pelvis_height_m'],.64)

    def test_reach_seed_preserves_global_wrist_orientation(self):
        ts=fk(self.a,{'r_shoulder_y_joint':-0.55,'r_wrist_y_joint':0.55})
        np.testing.assert_allclose(ts['r_wrist_z_link'][:3,:3],np.eye(3),atol=1e-9)
        self.assertLess(np.linalg.norm(self.c['comfort_center_local_m']),self.c['max_pelvis_wrist_distance_m'])
        self.assertLess(self.c['comfort_center_local_m'][0],0) # anatomical robot-right is negative X in planner

    def test_reject_out_of_limits(self):
        with self.assertRaises(ValueError):
            fk(self.a,{'r_wrist_y_joint':2.0})

    def test_tip_is_not_urdf_calibrated(self):
        self.assertIn('NO wrist-to-tip',self.r['assumptions_not_derived_from_urdf']['wrist_to_tip_local_m'])
        self.assertEqual(self.c['wrist_to_tip_local_m'],[0,0,.12])

if __name__=='__main__': unittest.main()
