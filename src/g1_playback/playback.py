from __future__ import annotations

import time
import numpy as np
import mujoco
import mujoco.viewer

from .model import set_mocap_target
from .motion import MotionClip


def play_clip(
    model: mujoco.MjModel,
    clip: MotionClip,
    *,
    speed: float = 1.0,
    loop: bool = False,
) -> None:
    """Kinematically play a MotionClip in the interactive MuJoCo viewer."""
    if clip.qpos.shape[1] != model.nq:
        raise ValueError(
            f"Clip nq={clip.qpos.shape[1]} does not match model.nq={model.nq}."
        )
    if speed <= 0:
        raise ValueError("speed must be positive")

    data = mujoco.MjData(model)
    frame_period = 1.0 / (clip.fps * speed)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for k in range(clip.nframes):
                if not viewer.is_running():
                    return

                tick = time.perf_counter()

                # Reference playback: set qpos exactly, then recompute kinematics.
                data.qpos[:] = clip.qpos[k]
                data.qvel[:] = 0.0

                if clip.target_pos is not None:
                    set_mocap_target(model, data, clip.target_pos[k])

                mujoco.mj_forward(model, data)
                viewer.sync()

                elapsed = time.perf_counter() - tick
                if elapsed < frame_period:
                    time.sleep(frame_period - elapsed)

            if not loop:
                # Hold the last frame while allowing camera inspection.
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.02)
                return
