"""Offline light-dependency scheduling contract tests; no diffusion weights."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from integration.soma_bridge.targets import prepare
from integration.soma_bridge.generate import _validate_target_file,create_constraints,_run_generation
ROOT=Path(__file__).resolve().parents[1]


class SchedulingBridgeTests(unittest.TestCase):
    def setUp(self):
        self.src=ROOT/'validation/far/executed_reference.json'

    def test_gain_gate_excludes_walking_hand_constraints(self):
        with tempfile.TemporaryDirectory() as td:
            obj=json.loads(self.src.read_text())
            times=np.asarray(obj['relative_times_s'])
            obj['scheduling']={'reach_gain':np.clip((times - 1.0)/0.6,0,1).tolist()}
            src=Path(td)/'plan.json';src.write_text(json.dumps(obj))
            dest=Path(td)/'targets.json'
            out=prepare(src,dest,duration_s=4.0,hand_stride=12,root_stride=15)
            self.assertGreater(out['right_hand_frame_indices'][0],30)
            self.assertEqual(out['phase_split_frame'],out['right_hand_frame_indices'][0])
            self.assertTrue(all(i>=out['phase_split_frame'] for i in out['right_hand_frame_indices']))
            self.assertEqual(out['root_frame_indices'][0],0)
            self.assertEqual(out['root_frame_indices'][-1],119)
            self.assertEqual(_validate_target_file(dest)['num_frames'],120)

    def test_no_hand_is_supported(self):
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)/'targets.json'
            out=prepare(self.src,dest,duration_s=3.0,no_hand=True)
            self.assertEqual(out['right_hand_frame_indices'],[])
            self.assertIsNone(out['phase_split_frame'])
            _validate_target_file(dest)

    def test_root_only_constraint_set_and_no_max_error(self):
        import torch
        class DummySet:
            def __init__(self,*args): self.args=args
        stub=types.ModuleType('kimodo.constraints')
        stub.RightHandConstraintSet=DummySet
        stub.Root2DConstraintSet=DummySet
        fake=types.ModuleType('kimodo');fake.constraints=stub
        class Skel:
            nbjoints=3
            bone_index={'Hips':0,'RightHand':1,'RightHandMiddleEnd':2}
            root_idx=0
            neutral_joints=torch.zeros((3,3))
        obj={
            'right_hand_frame_indices':[], 'root_frame_indices':[0,4,9], 'num_frames':10,
            'trajectories':{
                'pelvis_position_m':[[0,.64,i*.005] for i in range(10)],
                'pelvis_orientation_xyzw':[[0,0,0,1]]*10,
                'right_wrist_position_m':[[.2,.84,.3+i*.005] for i in range(10)],
                'right_hand_tip_position_m':[[.2,.84,.42+i*.005] for i in range(10)],
                'right_wrist_orientation_xyzw':[[0,0,0,1]]*10,
            },
        }
        ref=np.array([[0,1,0],[.2,1.2,.3],[.2,1.2,.4]],float)
        with patch.dict(sys.modules,{'kimodo':fake,'kimodo.constraints':stub}):
            constraints,diag=create_constraints(obj,Skel(),ref,np.tile(np.eye(3),(3,1,1)),ref[0])
        self.assertEqual(len(constraints),1)
        self.assertEqual(constraints[0].args[1].tolist(),[0,4,9])
        self.assertEqual(diag['frame_indices'],[])
        self.assertIsNone(diag['max_tip_discrepancy_m'])

    def test_multi_prompt_durations_match_clip(self):
        observed={}
        class FakeModel:
            def __call__(self,prompts,lengths,**kwargs):
                observed.update(prompts=prompts,lengths=lengths,kwargs=kwargs)
                return {'ok':True}
        mock=types.ModuleType('kimodo.tools')
        mock.seed_everything=lambda seed:None
        fake=types.ModuleType('kimodo');fake.tools=mock
        obj={'num_frames':210,'right_hand_frame_indices':[135,150,209],'phase_split_frame':135}
        with patch.dict(sys.modules,{'kimodo':fake,'kimodo.tools':mock}):
            out=_run_generation(FakeModel(),[],obj,'fallback',diffusion_steps=10)
        self.assertEqual(observed['lengths'],[135,75])
        self.assertEqual(observed['kwargs']['num_denoising_steps'],10)
        self.assertEqual(observed['kwargs']['num_transition_frames'],8)
        self.assertTrue(observed['kwargs']['multi_prompt'])
        self.assertEqual(out,{'ok':True})

    def test_legacy_targets_remain_all_clip(self):
        with tempfile.TemporaryDirectory() as td:
            out=prepare(self.src,Path(td)/'out.json',duration_s=3.0)
            self.assertEqual(out['right_hand_frame_indices'][0],0)
            self.assertEqual(out['right_hand_frame_indices'][-1],89)
            self.assertIsNone(out['phase_split_frame'])

    def test_config_contains_controller_schedule(self):
        c=json.loads((ROOT/'configs/elf3_gain_scheduled.json').read_text())
        self.assertTrue(c['gain_schedule_enabled'])
        self.assertEqual(c['reach_gate_far_m'],2.0)
        self.assertEqual(c['reach_gate_near_m'],1.2)
        self.assertLess(c['human_standoff_m'],c['reach_gate_near_m'])


if __name__=='__main__':unittest.main()
