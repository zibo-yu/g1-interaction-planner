"""Closed-loop Stage-1 task-space reach using the calibrated ELF3-proxy geometry.

Run from the project root on Windows after .venv setup:
    .\.venv\Scripts\python.exe examples\elf3_proxy_demo.py

Uses a synthetic double-integrator plant; does NOT execute ELF3 joints or KiMoDo.
"""
from pathlib import Path
import csv
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from handshake_mpc import HandshakeMPC,MPCConfig,HandshakeTarget,PlannerCommand
from handshake_mpc.simulation import neutral_state,advance_plant
from handshake_mpc.geometry import rotation

c=MPCConfig.load(ROOT/'configs/elf3_proxy.json')
ctrl=HandshakeMPC(c)
state=neutral_state(c)
# Set the goal from the pelvis-local *seed reaching pose*, in negative-X for robot right.
# Hand-tip offset remains provisional because its geometry is absent from the URDF.
comfortable_wrist_target=(state.pelvis_position_m +
                          rotation(state.pelvis_orientation_xyzw) @ np.asarray(c.comfort_center_local_m))
goal_tip=comfortable_wrist_target + rotation(state.right_wrist_orientation_xyzw) @ np.asarray(c.wrist_to_tip_local_m)
records=[]
plan=None
for k in range(30):
    plan=ctrl.update(state,HandshakeTarget(state.timestamp_s,goal_tip),
                     PlannerCommand(initiated=(k==0)))
    records.append((float(state.timestamp_s),*state.pelvis_position_m,*state.right_wrist_position_m))
    state=advance_plant(state,plan,c.dt_s)
records.append((float(state.timestamp_s),*state.pelvis_position_m,*state.right_wrist_position_m))
out=ROOT/'outputs'/'elf3_proxy_reach'
out.mkdir(parents=True,exist_ok=True)
plan.save_json(out/'last_predicted_horizon.json',sample_rate_hz=30)
with (out/'executed_states.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f)
    w.writerow(['time_s','pelvis_x','pelvis_y','pelvis_z','wrist_x','wrist_y','wrist_z'])
    w.writerows(records)
actual_tip=state.right_wrist_position_m + rotation(state.right_wrist_orientation_xyzw) @ np.asarray(c.wrist_to_tip_local_m)
print('Target tip:',goal_tip)
print('Final tip: ',actual_tip)
print('Error [m]: ',float(np.linalg.norm(actual_tip-goal_tip)))
print('Files: ',out)
print('Synthetic planner check only. No SOMA constraints / robot IK or contact modelling.')
