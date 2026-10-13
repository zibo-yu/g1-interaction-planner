"""Offline MuJoCo kinematic playback of a SOMA KiMoDo NPZ.

Uses mocap-controlled capsules and spheres for a SOMA skeleton. No robot physics,
ELF3 retargeting, or whole-body tracking is performed. No mesh assets required.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

# Proper right-handed transform: SOMA (x left,y up,z forward) -> MuJoCo (x forward,y left,z up)
SOMA_TO_MUJOCO = np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])


def load_motion(npz_path):
    with np.load(npz_path, allow_pickle=False) as data:
        if 'posed_joints' not in data.files:
            raise ValueError('KiMoDo motion NPZ must contain posed_joints')
        positions = np.asarray(data['posed_joints'], dtype=float)
    if positions.ndim == 4:
        if positions.shape[0] != 1:
            raise ValueError('Only a single motion clip is supported')
        positions = positions[0]
    if positions.ndim != 3 or positions.shape[2] != 3 or not np.isfinite(positions).all():
        raise ValueError('posed_joints should be [T,J,3] in meters')
    if positions.shape[0] < 2:
        raise ValueError('At least two frames required')
    return positions


def load_hierarchy(motion_path, hierarchy_file=None):
    f = Path(hierarchy_file) if hierarchy_file else Path(str(Path(motion_path).with_suffix(''))+'.skeleton.json')
    if f.exists():
        v=json.loads(f.read_text(encoding='utf8'))
        names=v['joint_names'];parents=v['joint_parents'];fps=float(v['fps'])
        return names,parents,fps
    # For externally generated KiMoDo NPZ, infer skeleton from the installed KiMoDo package.
    try:
        from kimodo.skeleton import SOMASkeleton30,SOMASkeleton77
        poses=load_motion(motion_path)
        skeleton = {30:SOMASkeleton30,77:SOMASkeleton77}[poses.shape[1]]()
        return list(skeleton.bone_order_names),[int(p) for p in skeleton.joint_parents.tolist()],30.0
    except Exception as e:
        raise FileNotFoundError(f'Missing {f}. Generate with soma_bridge.generate or supply --hierarchy. '
                                'For an external NPZ you may install KiMoDo to infer the hierarchy.') from e


def validate_skeleton(names,parents,joint_count):
    if len(names)!=joint_count or len(parents)!=joint_count:
        raise ValueError(f'Hierarchy has {len(names)} joints and {len(parents)} parents, motion has {joint_count} joints')
    if len(set(names))!=len(names):
        raise ValueError('Duplicate skeleton joint names')
    for i,par in enumerate(parents):
        if not isinstance(par,int) or par>=i or par< -1:
            raise ValueError(f'Invalid hierarchy at joint {i}: parent={par}')
    if parents.count(-1)!=1:
        raise ValueError('Exactly one root required')


def _orientation_quat_wxyz(segment):
    if np.linalg.norm(segment)<1e-8:
        return [1.,0.,0.,0.]
    u=np.asarray(segment,dtype=float)/np.linalg.norm(segment)
    # Minimum-angle rotation from local Z axis to this segment.
    a=np.cross([0.,0.,1.],u)
    dot=np.clip(u[2],-1.0,1.0)
    if dot < -1+1e-8:
        quat=np.array([1.,0.,0.,0.])  # 180 degrees about X, xyzw
    elif np.linalg.norm(a)<1e-8:
        quat=np.array([0.,0.,0.,1.])
    else:
        a/=np.linalg.norm(a)
        angle=np.arccos(dot)
        quat=np.r_[a*np.sin(angle/2),np.cos(angle/2)]
    return quat[[3,0,1,2]].tolist()


def make_xml(positions_mj, parents):
    """Generate independent mocap-controlled bodies, not an actuated humanoid."""
    root=ET.Element('mujoco',{'model':'SOMA skeleton offline player'})
    ET.SubElement(root,'option',{'gravity':'0 0 0','timestep':'0.0033333'})
    visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'global',{'offwidth':'1280','offheight':'900'})
    world=ET.SubElement(root,'worldbody')
    ET.SubElement(world,'light',{'pos':'0 1 4','dir':'0 0 -1'})
    ET.SubElement(world,'geom',{'name':'floor','type':'plane','pos':'0 0 0','size':'8 8 0.1','rgba':'0.75 0.77 0.8 1'})
    for i,par in enumerate(parents):
        b=ET.SubElement(world,'body',{'name':f'joint_{i}', 'mocap':'true','pos':'0 0 1'})
        color='0.16 0.56 0.85 1' if i in (0,1,2) else '0.25 0.29 0.36 1'
        ET.SubElement(b,'geom',{'type':'sphere','size':'.026','rgba':color,'contype':'0','conaffinity':'0'})
        if par >=0:
            length=float(np.linalg.norm(positions_mj[0,i]-positions_mj[0,par]))
            b=ET.SubElement(world,'body',{'name':f'bone_{i}','mocap':'true','pos':'0 0 1'})
            ET.SubElement(b,'geom',{'type':'capsule','size':f'.015 {max(.001,length/2):.6g}',
                                    'rgba':'0.42 0.48 0.56 1','contype':'0','conaffinity':'0'})
    return ET.tostring(root,encoding='unicode')


def set_frame(model,data,positions_mj,parents,frame):
    pts=positions_mj[int(frame)]
    for i,par in enumerate(parents):
        # Simpler robust indexing across mujoco releases:
        j_id=int(model.body_mocapid[model.body(f'joint_{i}').id])
        data.mocap_pos[j_id] = pts[i]
        data.mocap_quat[j_id] = [1.,0.,0.,0.]
        if par >=0:
            bid=int(model.body_mocapid[model.body(f'bone_{i}').id])
            segment=pts[i]-pts[par]
            data.mocap_pos[bid] = 0.5*(pts[i]+pts[par])
            data.mocap_quat[bid] = _orientation_quat_wxyz(segment)


def play(path,hierarchy=None,headless=False,speed=1.0,loops=1):
    import mujoco
    path=Path(path)
    positions=load_motion(path)
    names,parents,fps=load_hierarchy(path,hierarchy)
    validate_skeleton(names,parents,positions.shape[1])
    pmj=positions @ SOMA_TO_MUJOCO.T
    model=mujoco.MjModel.from_xml_string(make_xml(pmj,parents))
    data=mujoco.MjData(model)
    set_frame(model,data,pmj,parents,0)
    mujoco.mj_forward(model,data)
    print(f'MuJoCo offline SOMA playback: {len(pmj)} frames, {len(parents)} joints, {fps:.1f} fps')
    if headless:
        for i in range(len(pmj)):
            set_frame(model,data,pmj,parents,i)
            mujoco.mj_forward(model,data)
        print('PASS: headless mocap skeleton stepped successfully (no ELF3 physics).')
        return
    if speed<=0 or loops<1:raise ValueError('speed>0 and loops>=1 required')
    import mujoco.viewer
    with mujoco.viewer.launch_passive(model,data) as viewer:
        viewer.cam.lookat[:]=pmj.reshape(-1,3).mean(axis=0)
        viewer.cam.distance=max(2.,float(np.linalg.norm(np.ptp(pmj.reshape(-1,3),axis=0)))*1.5)
        viewer.cam.azimuth=125
        viewer.cam.elevation=-20
        for _ in range(loops):
            t0=time.perf_counter()
            for i in range(len(pmj)):
                if not viewer.is_running():return
                set_frame(model,data,pmj,parents,i)
                mujoco.mj_forward(model,data)
                viewer.sync()
                delay=(i+1)/(fps*speed)-(time.perf_counter()-t0)
                if delay>0:time.sleep(delay)
        print('Playback complete; close the MuJoCo window to exit.')
        while viewer.is_running():
            time.sleep(0.08)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',required=True)
    p.add_argument('--hierarchy')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--speed',type=float,default=1.0)
    p.add_argument('--loops',type=int,default=1)
    args=p.parse_args()
    play(args.motion,args.hierarchy,args.headless,args.speed,args.loops)

if __name__=='__main__':main()
