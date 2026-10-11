from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import mujoco
import mujoco_menagerie as mm


TARGET_BODY_NAME = "human_hand_target"
TARGET_GEOM_NAME = "human_hand_target_geom"


def build_g1_model(add_hand_target: bool = True) -> mujoco.MjModel:
    """Load the official Menagerie G1 scene and optionally add a mocap target.

    We compile an editable MjSpec so the target marker becomes part of the same
    MuJoCo model. The target is a mocap body: changing data.mocap_pos moves it
    without adding dynamics to the robot.
    """
    robot = mm.get("unitree_g1")
    spec = robot.spec()  # default scene: G1 + floor/lights

    if add_hand_target:
        target = spec.worldbody.add_body(
            name=TARGET_BODY_NAME,
            pos=[0.65, -0.25, 1.0],
            mocap=True,
        )
        target.add_geom(
            name=TARGET_GEOM_NAME,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.04, 0.0, 0.0],
            rgba=[0.9, 0.1, 0.1, 1.0],
            contype=0,
            conaffinity=0,
            density=0,
        )

    return spec.compile()


def name2id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise KeyError(f"{name!r} not found for MuJoCo object type {obj_type}.")
    return idx


def joint_qpos_address(model: mujoco.MjModel, joint_name: str) -> int:
    """Return the first qpos address for a named joint.

    G1 actuated joints are 1-DoF hinge joints, so one qpos entry per named joint
    is expected. The root free joint is handled separately.
    """
    jid = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_qposadr[jid])


def keyframe_qpos(model: mujoco.MjModel, key_name: str = "stand") -> np.ndarray:
    kid = name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    return np.array(model.key_qpos[kid], dtype=np.float64, copy=True)


def set_mocap_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    position: np.ndarray,
    body_name: str = TARGET_BODY_NAME,
) -> None:
    bid = name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mocap_id = int(model.body_mocapid[bid])
    if mocap_id < 0:
        raise ValueError(f"Body {body_name!r} is not a mocap body.")
    data.mocap_pos[mocap_id] = np.asarray(position, dtype=np.float64)


def apply_named_joint_positions(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    joint_positions: dict[str, float],
) -> np.ndarray:
    """Apply a named joint dictionary to a complete MuJoCo qpos vector."""
    out = np.asarray(qpos, dtype=np.float64).copy()

    for joint_name, value in joint_positions.items():
        jid = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(
                f"{joint_name!r} is not a 1-DoF hinge joint; "
                "use a representation-specific adapter."
            )
        adr = int(model.jnt_qposadr[jid])
        out[adr] = float(value)

    return out


def joint_names(model: mujoco.MjModel) -> list[str]:
    names: list[str] = []
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name is not None:
            names.append(name)
    return names


def body_names(model: mujoco.MjModel) -> list[str]:
    names: list[str] = []
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if name is not None:
            names.append(name)
    return names
