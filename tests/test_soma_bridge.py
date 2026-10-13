"""Tests for stage-2 interchange, frames, and MuJoCo skeleton geometry (no model weights)."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]

from integration.soma_bridge.targets import load_plan,prepare
from integration.soma_bridge.generate import _single_clip,create_constraints,_validate_target_file
from integration.soma_bridge.play_mujoco import (
    SOMA_TO_MUJOCO,_orientation_quat_wxyz,make_xml,load_motion,validate_skeleton
)


class TestMPCToSOMA(unittest.TestCase):
    def setUp(self):
        self.plan=ROOT/'validation'/'far'/'executed_reference.json'

    def test_real_v1_payload_and_resampling(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'targets.json'
            a=prepare(self.plan,out,fps=30,start_s=0,duration_s=3,hand_stride=12)
            self.assertEqual(a['num_frames'],90)
            self.assertEqual(a['right_hand_frame_indices'][0],0)
            self.assertEqual(a['right_hand_frame_indices'][-1],89)
            self.assertEqual(len(a['trajectories']['right_wrist_position_m']),90)
            self.assertAlmostEqual(a['time_s'][-1],89/30)
            self.assertEqual(_validate_target_file(out)['num_frames'],90)

    def test_reject_outside_window(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                prepare(self.plan,Path(td)/'t.json',start_s=11,duration_s=3)

    def test_reject_bad_coordinates(self):
        with tempfile.TemporaryDirectory() as td:
            x=json.loads(self.plan.read_text())
            x['coordinate_convention']['up_axis']='Z'
            f=Path(td)/'bad.json';f.write_text(json.dumps(x))
            with self.assertRaisesRegex(ValueError,'coordinate convention'):
                load_plan(f)

    def test_reference_tensor_shapes(self):
        a=np.zeros((10,30,3,3))
        self.assertEqual(_single_clip(a,'local_rot_mats',(3,3)).shape,(10,30,3,3))
        self.assertEqual(_single_clip(a[None],'local_rot_mats',(3,3)).shape,(10,30,3,3))
        b=np.zeros((10,3))
        self.assertEqual(_single_clip(b,'root_positions',(3,)).shape,(10,3))
        self.assertEqual(_single_clip(b[None],'root_positions',(3,)).shape,(10,3))
        with self.assertRaises(ValueError):
            _single_clip(np.zeros((2,10,30,3,3)),'local_rot_mats',(3,3))

    def test_constraint_inmemory_no_overlap_height_and_tip(self):
        import torch
        class DummySet:
            def __init__(self,*args):self.args=args
        stub=types.ModuleType('kimodo.constraints')
        stub.RightHandConstraintSet=DummySet
        stub.Root2DConstraintSet=DummySet
        fake=types.ModuleType('kimodo')
        fake.constraints=stub
        class Skel:
            nbjoints=3
            bone_index={'Hips':0,'RightHand':1,'RightHandMiddleEnd':2}
            root_idx=0
            neutral_joints=torch.zeros((3,3))
        sk=Skel()
        ref_pos=np.array([[0,1,0],[.2,1.2,.3],[.2,1.2,.4]],dtype=float)
        ref_rot=np.tile(np.eye(3),(3,1,1))
        n=30
        obj={
            'right_hand_frame_indices':[0,15,29],
            'num_frames':n,
            'trajectories':{
                'pelvis_position_m':[[0,.64,i*.005] for i in range(n)],
                'pelvis_orientation_xyzw':[[0,0,0,1]]*n,
                'right_wrist_position_m':[[.2,.84,.3+i*.005] for i in range(n)],
                'right_hand_tip_position_m':[[.2,.84,.42+i*.005] for i in range(n)],
                'right_wrist_orientation_xyzw':[[0,0,0,1]]*n,
            }
        }
        with patch.dict(sys.modules,{'kimodo':fake,'kimodo.constraints':stub}):
            constraints,diag=create_constraints(obj,sk,ref_pos,ref_rot,ref_pos[0])
        self.assertEqual(len(constraints),2)
        hand=constraints[-1]
        self.assertEqual(hand.args[2].shape,(3,3,3))
        self.assertAlmostEqual(diag['soma_root_height_shift_m'],.36)
        self.assertAlmostEqual(diag['max_tip_discrepancy_m'],.02,places=6)
        self.assertFalse(set(diag['frame_indices']) & set(diag['root_dense_indices_without_hand_overlap']))
        self.assertAlmostEqual(hand.args[2][0,1,1].item(),1.2,places=5)

    def test_soma_to_mujoco_coordinate_rotation(self):
        self.assertAlmostEqual(np.linalg.det(SOMA_TO_MUJOCO),1)
        np.testing.assert_allclose(SOMA_TO_MUJOCO@[0,1,0],[0,0,1])
        np.testing.assert_allclose(SOMA_TO_MUJOCO@[0,0,1],[1,0,0])

    def test_xml_and_quaternion(self):
        q=np.asarray(_orientation_quat_wxyz([1.,0.,0.]))
        from scipy.spatial.transform import Rotation
        np.testing.assert_allclose(Rotation.from_quat(q[[1,2,3,0]]).apply([0,0,1]),[1,0,0],atol=1e-6)
        xml=make_xml(np.array([[[0,0,1],[0,0,1.3]]]),[-1,0])
        self.assertIn('mocap="true"',xml)
        validate_skeleton(['root','tip'],[-1,0],2)
        with self.assertRaises(ValueError):validate_skeleton(['root','tip'],[-1,1],2)

    def test_npz_load(self):
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'sample.npz'
            np.savez(f,posed_joints=np.zeros((10,30,3)))
            self.assertEqual(load_motion(f).shape,(10,30,3))


if __name__=='__main__':unittest.main()
