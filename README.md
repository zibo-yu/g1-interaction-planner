# G1 MuJoCo Playback Base

A deliberately small MuJoCo harness for the **planner** side of the project.

Its purpose is to let us debug:

1. loading a Unitree G1 model;
2. named-joint ↔ MuJoCo `qpos` mapping;
3. kinematic playback of a whole-body reference motion;
4. a moving "human hand" target;
5. the file/API boundary that KiMoDo/ARDY will eventually feed.

This repo does **not** implement the RL tracker. During playback we set `data.qpos`
directly and call `mujoco.mj_forward(...)`. Therefore the displayed robot is the
planner's **reference motion**, not a physically tracked robot.

## Architecture

```text
mock vision / scripted hand target
                |
                v
        planner (future)
                |
                v
       MotionClip / .npz
                |
                v
     MuJoCo kinematic playback
```

Later:

```text
TaskSpacePlan -> KiMoDo/ARDY adapter -> MotionClip -> this viewer
```

The tracker team can consume the same reference motion separately.

---

## 1. Installation

Recommended: use a clean Python 3.10+ environment.

```bash
cd g1_mujoco_playback_base

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e .
```

If you want MP4 export later:

```bash
pip install -e ".[video]"
```

The project uses the official `mujoco-menagerie` Python package. The G1 model is
downloaded into its local cache on first use.

If running under Windows 11 + WSL2, the interactive MuJoCo viewer normally uses
WSLg. Test GUI support first with:

```bash
python -c "import mujoco; print(mujoco.__version__)"
g1-demo --mode stand
```

---

## 2. Inspect the G1 model

```bash
g1-inspect
```

This prints:

- `nq`, `nv`, `nu`;
- keyframes;
- every joint name;
- MuJoCo qpos address for every joint;
- body names useful for later FK/debugging.

This is the first command to run when connecting another motion representation.

---

## 3. Run the base demos

Static standing pose:

```bash
g1-demo --mode stand
```

Simple right-arm motion:

```bash
g1-demo --mode wave
```

Handshake-like reaching motion plus a moving red human-hand target:

```bash
g1-demo --mode reach
```

Change playback speed:

```bash
g1-demo --mode reach --speed 0.5
g1-demo --mode reach --speed 2.0
```

Loop:

```bash
g1-demo --mode reach --loop
```

Save the generated reference motion:

```bash
g1-demo --mode reach --save-npz outputs/reach.npz
```

Then replay it:

```bash
g1-play outputs/reach.npz
```

---

## 4. Motion file format

The canonical interchange file is an `.npz` containing:

```text
qpos        [T, nq]   complete MuJoCo generalized position vector
fps         scalar
target_pos  [T, 3]    optional human-hand target for visualization
```

The `MotionClip` class is in:

```text
src/g1_playback/motion.py
```

For generated motions that contain **named robot joints** rather than complete
MuJoCo `qpos`, use `apply_named_joint_positions(...)`. It maps by MuJoCo joint
name and does not assume a hard-coded joint order.

Example:

```python
joint_frame = {
    "right_shoulder_pitch_joint": -0.8,
    "right_shoulder_roll_joint": -0.15,
    "right_elbow_joint": 0.7,
}

qpos = stand_qpos.copy()
apply_named_joint_positions(model, qpos, joint_frame)
```

That is the pattern the future KiMoDo adapter should use.

---

## 5. Coordinate convention

This harness uses the coordinates supplied by the G1 MuJoCo model. Do not
silently assume that KiMoDo/vision uses the same convention.

Before integration, explicitly define transforms:

```text
vision frame -> planner world frame -> generator frame -> MuJoCo world frame
```

and explicitly map joint names.

---

## 6. Next integration step

The next useful file to add is:

```text
src/g1_playback/backends/kimodo_adapter.py
```

with one responsibility:

```text
KiMoDo output
    -> named G1 joints + root pose
    -> MuJoCo qpos[T, nq]
    -> MotionClip
```

Do **not** put FSM, trajectory planning, or KiMoDo-specific code inside the
MuJoCo viewer.

---

## Repo layout

```text
g1_mujoco_playback_base/
├── pyproject.toml
├── README.md
├── .gitignore
├── src/
│   └── g1_playback/
│       ├── __init__.py
│       ├── model.py
│       ├── motion.py
│       ├── playback.py
│       ├── trajectories.py
│       ├── inspect_model.py
│       ├── demo.py
│       └── play_npz.py
└── tests/
    └── test_trajectories.py
```

## Important design choice

`mujoco.mj_forward(...)` is intentionally used instead of `mujoco.mj_step(...)`
for the main playback path.

That means:

- the robot cannot fall;
- contacts are not controlling anything;
- no actuator torques are being solved;
- the displayed pose is exactly the desired planner reference.

Once the tracker subgroup is ready, dynamic tracking is a separate integration
path and should not be mixed into this debugging harness.
