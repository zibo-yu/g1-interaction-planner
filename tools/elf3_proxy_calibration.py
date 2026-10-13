"""Derive an *ELF3 proxy* task-space calibration for Stage 1 (not SOMA).

Use: python tools/elf3_proxy_calibration.py
Requires only numpy, ElementTree and the bundled URDFs. No meshes, MuJoCo or KiMoDo.
The main outputs separate URDF-derived quantities from engineering assumptions.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

# +X URDF forward => +Z planner forward; +Y URDF left => +X planner left;
# +Z URDF up => +Y planner up. This permutation is a proper rotation (det=+1).
ELF_TO_PLANNER = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])


def xyz(elem, attr='xyz', default=(0., 0., 0.)):
    return np.array([float(x) for x in (elem.get(attr) if elem is not None and elem.get(attr) is not None
                                        else ' '.join(str(a) for a in default)).split()])


def axis_angle(axis, theta):
    n = float(np.linalg.norm(axis))
    if n == 0:
        raise ValueError('Revolute axis must be nonzero')
    x,y,z = axis/n
    c,s = np.cos(theta),np.sin(theta)
    C=1-c
    return np.array([[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
                     [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
                     [z*x*C-y*s, z*y*C+x*s, c+z*z*C]])


def rpy_matrix(rpy):
    r,p,yaw = rpy
    return axis_angle(np.array([0.,0.,1.]),yaw) @ axis_angle(np.array([0.,1.,0.]),p) @ axis_angle(np.array([1.,0.,0.]),r)


def trans(R, p):
    T=np.eye(4)
    T[:3,:3]=R
    T[:3,3]=p
    return T


def load_urdf(path: Path):
    root=ET.parse(path).getroot()
    if root.tag!='robot':
        raise ValueError('Expected URDF robot element')
    links={e.get('name') for e in root.findall('link')}
    joints={}
    for j in root.findall('joint'):
        name=j.get('name')
        if name in joints: raise ValueError('Duplicate joint: '+name)
        o=j.find('origin'); ax=j.find('axis'); lim=j.find('limit')
        entry={'type':j.get('type'), 'parent':j.find('parent').get('link'),
               'child':j.find('child').get('link'),
               'origin_xyz':xyz(o), 'origin_rpy':xyz(o,'rpy'),
               'axis':xyz(ax,default=(1,0,0)),
               'limit':None if lim is None else {key:float(val) for key,val in lim.attrib.items()}}
        if entry['parent'] not in links or entry['child'] not in links:
            raise ValueError('Link missing from URDF: '+name)
        joints[name]=entry
    child_links={j['child'] for j in joints.values()}
    roots=links-child_links
    if len(roots)!=1:raise ValueError(f'Expected one root link, got {roots}')
    return {'name':root.get('name'),'links':links,'joints':joints,'root':roots.pop()}


def kinematic_signature(model):
    def serial(v):
        if isinstance(v,np.ndarray): return [float(x) for x in v]
        if isinstance(v,dict):return {k:serial(val) for k,val in sorted(v.items())}
        return v
    return json.dumps({name:serial(j) for name,j in sorted(model['joints'].items())},sort_keys=True)


def fk(model, angles=None):
    """Map all link frames into URDF root-link coordinates; URDF joint rotation is origin * R_axis(q)."""
    angles=angles or {}
    transforms={model['root']:np.eye(4)}
    pending=list(model['joints'].items())
    while pending:
        left=[]
        for name,j in pending:
            if j['parent'] not in transforms:
                left.append((name,j));continue
            q=float(angles.get(name,0.0))
            if j['type']=='revolute' or j['type']=='continuous':
                lim=j['limit']
                if lim and not (lim.get('lower',-np.inf)-1e-9 <= q <= lim.get('upper',np.inf)+1e-9):
                    raise ValueError(f'Joint {name} value {q} outside URDF limits')
                Jrot=axis_angle(j['axis'],q)
            elif j['type']=='fixed':
                if abs(q)>1e-12: raise ValueError('Nonzero fixed joint: '+name)
                Jrot=np.eye(3)
            else:raise ValueError('Unsupported URDF joint type: '+str(j['type']))
            transforms[j['child']] = transforms[j['parent']] @ trans(rpy_matrix(j['origin_rpy']),j['origin_xyz']) @ trans(Jrot,[0,0,0])
        if len(left)==len(pending):
            raise ValueError('Disconnected or cyclic link tree')
        pending=left
    return transforms


def planner(vec):return ELF_TO_PLANNER @ vec


def joint_summary(model,names):
    return {name:{'axis_elf3':j['axis'].tolist(), 'limit_rad':None if j['limit'] is None else
            [j['limit'].get('lower'),j['limit'].get('upper')],
            'origin_xyz_elf3_m':j['origin_xyz'].tolist()} for name in names for j in [model['joints'][name]]}


def derive(model, original_config):
    T=fk(model)
    hip_l=T['l_hip_y_link'][:3,3]
    hip_r=T['r_hip_y_link'][:3,3]
    pelvis_elf=(hip_l+hip_r)/2  # VIRTUAL pelvis origin, not a link
    torso_elf=T['torso_link'][:3,3]
    shoulder_elf=T['r_shoulder_y_link'][:3,3]
    wrist_elf=T['r_wrist_z_link'][:3,3]
    ankle_l=T['l_ankle_x_link'][:3,3]
    ankle_r=T['r_ankle_x_link'][:3,3]
    nominal_reach_joints={'r_shoulder_y_joint':-0.55,'r_wrist_y_joint':0.55}
    T_reach=fk(model,nominal_reach_joints)
    reach_wrist_elf=T_reach['r_wrist_z_link'][:3,3]
    wrist_orientation_reach=T_reach['r_wrist_z_link'][:3,:3]
    # At this reference pose R_y(-0.55)R_y(+0.55)=I, preserving a fixed world wrist orientation.
    if np.max(np.abs(wrist_orientation_reach-np.eye(3)))>1e-8:
        raise AssertionError('Chosen reference wrist is not globally aligned')
    upper=np.linalg.norm(model['joints']['r_elbow_y_joint']['origin_xyz'])
    fore=np.linalg.norm(model['joints']['r_wrist_x_joint']['origin_xyz'])
    shoulder_relative=shoulder_elf-pelvis_elf
    upper_bound=float(np.linalg.norm(shoulder_relative)+upper+fore)
    straight_ankle_height=float(pelvis_elf[2]-0.5*(ankle_l[2]+ankle_r[2]))
    c=dict(original_config)
    c['pelvis_height_m']=round(straight_ankle_height,7)
    c['pelvis_to_torso_local_m']=[round(float(v),7) for v in planner(torso_elf-pelvis_elf)]
    c['neutral_wrist_local_m']=[round(float(v),7) for v in planner(wrist_elf-pelvis_elf)]
    c['comfort_center_local_m']=[round(float(v),7) for v in planner(reach_wrist_elf-pelvis_elf)]
    # Deliberately retain original Stage-1 comfort widths, tip length, wrist orientation and hard radius.
    # They are assumptions or heuristics, NOT measured from the URDF.
    original_fields=['comfort_half_widths_m','wrist_to_tip_local_m','fixed_wrist_orientation_xyzw',
                     'max_pelvis_wrist_distance_m','pelvis_speed_max_mps','wrist_speed_max_mps',
                     'pelvis_acceleration_max_mps2','wrist_acceleration_max_mps2']
    assert np.linalg.norm(c['neutral_wrist_local_m']) < c['max_pelvis_wrist_distance_m']
    assert np.linalg.norm(c['comfort_center_local_m']) < c['max_pelvis_wrist_distance_m']
    report={
        'purpose':'ELF3 kinematic proxy for Stage-1 planner; NOT native KiMoDo-SOMA calibration',
        'source_model':'elf3',
        'urdf_root_link':model['root'],
        'coordinate_transform':{
            'matrix_elf3_to_planner':ELF_TO_PLANNER.tolist(),
            'elf3_frame':'Z up, +X forwards, +Y left (inferred from symmetric right/left shoulder frames)',
            'planner_frame':'Y up, +Z forwards, +X left (right arm is at negative X)',
            'quaternion_convention':'The planner and robot coordinate axes are rotated bases; orientations need R_planner=M R_elf M^T',
        },
        'references':{
            'virtual_pelvis_definition':'midpoint of l_hip_y_link and r_hip_y_link origins at zero joint angles',
            'torso_link':'torso_link', 'right_wrist_link':'r_wrist_z_link',
            'right_shoulder_link':'r_shoulder_y_link',
            'neutral_configuration':'all URDF joint coordinates equal zero; NOT verified to be natural standing pose',
            'comfort_center_seed_joint_angles_rad':nominal_reach_joints,
        },
        'urdf_derived':{
            'virtual_pelvis_in_torso_frame_elf3_m':pelvis_elf.tolist(),
            'right_shoulder_relative_to_pelvis_planner_m':planner(shoulder_relative).tolist(),
            'pelvis_to_torso_local_planner_m':c['pelvis_to_torso_local_m'],
            'zero_joint_wrist_relative_to_pelvis_planner_m':c['neutral_wrist_local_m'],
            'sample_reach_wrist_relative_to_pelvis_planner_m':c['comfort_center_local_m'],
            'upper_arm_segment_m':float(upper),
            'forearm_segment_m':float(fore),
            'upper_bound_shoulder_to_wrist_m':float(upper+fore),
            'triangle_inequality_pelvis_to_wrist_bound_m':upper_bound,
            'straight_zero_joint_hip_to_ankle_vertical_m':straight_ankle_height,
            'torso_to_hip_midpoint_vertical_m':float(torso_elf[2]-pelvis_elf[2]),
            'right_arm_joint_limits':joint_summary(model,[n for n in model['joints'] if n.startswith('r_shoulder_') or n.startswith('r_elbow_') or n.startswith('r_wrist_')]),
        },
        'assumptions_not_derived_from_urdf':{
            'pelvis_height_m':'Set equal to hip-to-ankle separation at URDF q=0, with ankle center at z=0. NOT sole-ground height or measured standing height.',
            'neutral_wrist_local_m':'Zero-joint FK result; not an ergonomically validated neutral configuration.',
            'comfort_center_local_m':'One joint-limit-valid FK seed with shoulder_y=-0.55, wrist_y=+0.55 radians, preserving global wrist orientation; not an optimized comfortable pose.',
            'comfort_half_widths_m':'Retained from original MPC settings. This box is NOT an inverse-kinematic feasible-set certificate.',
            'wrist_to_tip_local_m':'Retained provisional original value: the two URDFs contain NO wrist-to-tip joint/frame; mesh geometries were NOT supplied.',
            'fixed_wrist_orientation_xyzw':'Identity from the zero-joint / compensated-reach reference; true handshake orientation remains to be calibrated.',
            'max_pelvis_wrist_distance_m':'Retained original hard radius; smaller than the nominal fixed-torso geometric upper bound, but is NOT sufficient for reachability.',
            'motion_limits':'Retained MPC speed/acceleration limits; URDF revolute effort/velocity limits cannot determine task-space locomotion/acceleration bounds.',
        },
        'limits_of_use':[
            'ELF3 and SOMA skeletal geometries are not interchangeable; use this as approximate planning geometry only.',
            'The URDF files use meshes/hand models that are not included; no contact surface, finger tip or foot sole can be measured.',
            'Only nominal rigid fixed-waist link geometry is used; true whole-body IK, contact feasibility and motion generation remain unverified.',
            'Stage 1 holds heading and wrist orientation fixed; retargeted motion may need different constraints.',
            'The torso is defined as torso_link, whereas the pelvis is a virtual hip midpoint. Do not substitute the torso for KiMoDo Hips.',
        ],
    }
    return c,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf',type=Path,default=Path('models/elf3.urdf'))
    parser.add_argument('--other-urdf',type=Path,default=Path('models/elf3_hand_mesh.urdf'))
    parser.add_argument('--base-config',type=Path,default=Path('configs/default.json'))
    parser.add_argument('--config-out',type=Path,default=Path('configs/elf3_proxy.json'))
    parser.add_argument('--report-out',type=Path,default=Path('calibration/elf3_geometry_report.json'))
    args=parser.parse_args()
    first=load_urdf(args.urdf)
    other=load_urdf(args.other_urdf)
    if kinematic_signature(first)!=kinematic_signature(other):
        raise ValueError('ELF3 URDFs differ in kinematic chain/joint limits; select which model to use explicitly')
    config,report=derive(first,json.loads(args.base_config.read_text(encoding='utf-8')))
    report['urdf_comparison']='Identical joints, limits, axes and link transformations; differ only in visual/collision wrist mesh paths.'
    report['urdf_files']=[str(args.urdf),str(args.other_urdf)]
    args.config_out.parent.mkdir(parents=True,exist_ok=True)
    args.report_out.parent.mkdir(parents=True,exist_ok=True)
    args.config_out.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    args.report_out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('ELF3 proxy written:',args.config_out)
    print('Geometry report written:',args.report_out)
    for k in ('pelvis_height_m','pelvis_to_torso_local_m','neutral_wrist_local_m','comfort_center_local_m','max_pelvis_wrist_distance_m'):
        print(k,config[k])
    print('NOTE: proxy calibration only; not SOMA or certified joint feasibility')


if __name__=='__main__':main()
