from __future__ import annotations

import mujoco

from .model import build_g1_model


def main() -> None:
    model = build_g1_model(add_hand_target=True)

    print(f"model: {model.names!r}" if hasattr(model, "names") else "Unitree G1")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}, njnt={model.njnt}, nbody={model.nbody}")
    print()

    print("Keyframes")
    print("---------")
    for kid in range(model.nkey):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, kid)
        print(f"{kid:2d}  {name}")
    print()

    print("Joints")
    print("------")
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        qadr = int(model.jnt_qposadr[jid])
        dofadr = int(model.jnt_dofadr[jid])
        jtype = int(model.jnt_type[jid])
        print(
            f"{jid:2d}  qpos[{qadr:2d}]  dof[{dofadr:2d}]  "
            f"type={jtype}  {name}"
        )
    print()

    print("Selected bodies")
    print("---------------")
    for body_name in [
        "pelvis",
        "torso_link",
        "right_shoulder_pitch_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
        "human_hand_target",
    ]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        print(f"{body_name:32s} id={bid}")


if __name__ == "__main__":
    main()
