# Stage 2: Windows MPC → KiMoDo-SOMA → MuJoCo offline skeleton playback

**Implementation status:** the task-space preprocessing, in-memory constraint construction,
KiMoDo inference API call, SOMA NPZ/BVH export, and skeletal MuJoCo viewer are implemented.
The preprocessing and shape/geometry tests are exercised without model weights. **An actual
KiMoDo generation has not been executed here**; run the commands below in the existing KiMoDo
WSL2 environment. Inference still needs working model downloads and a text encoder.

## What is reused (and what is not)

- **Windows MPC remains unchanged**; it writes `handshake_task_space/v1` JSON.
- The ELF3 URDF geometry is a **planner proxy only**. It is not substituted for SOMA bones.
- The adapter uses the **actual SOMA skeleton** and one KiMoDo reference pose to construct
  direct `RightHandConstraintSet` plus `Root2DConstraintSet` objects in Python.
- In this first bridge, global wrist and tip positions may not be perfectly IK-consistent.
  The generator sees soft conditioning, not a guaranteed feasible joint trajectory.
- MuJoCo uses independent mocap bodies to animate SOMA joints and bones from `posed_joints`.
  **It is not ELF3 articulation, tracking, contact simulation, or retargeting.** No meshes needed.

## 0 — Generate a trajectory on native Windows (optional)

PowerShell from repo root (after `setup_windows.bat`):

```powershell
.\.venv\Scripts\python.exe -m handshake_mpc.demo --scenario near --output outputs\mpc_scenarios
```

This generates `outputs/mpc_scenarios/near/executed_reference.json` (12 seconds).
A bundled validated original-run fixture (the FAR scenario) also exists at
`validation/far/executed_reference.json` so you can start without CVXPY.

**Note:** the included fixtures use the original/default humanlike config. The
ELF3-proxy run can be prepared separately but does not constitute SOMA geometry.

## 1 — Prepare sparse hand targets and dense root path (Windows or WSL)

Run in the repo root with Python including `numpy` and `scipy`:

```bash
python -m integration.soma_bridge.targets \
  --plan validation/far/executed_reference.json \
  --start 0 --duration 4 --fps 30 --hand-stride 12 \
  --out outputs/bridge/targets.json
```

- Input is `handshake_task_space/v1` (Y-up, +Z forward, XYZW quaternion).
- Output `soma_task_targets/v1` is **NOT a native KiMoDo `constraints.json`**.
- 4 seconds × 30 fps = 120 generated frames (indices 0..119).
- Root targets are sampled at all 120 frames; hand keyframes are sampled about every 0.4 s.
- You can replace `validation/near/...` with the executed first-interval reference from
  your own moving/far/terminated synthetic episode. One static MPC **predicted horizon**
  is not the same as the accumulated executed reference across replan updates.
- Keep the first clip ≤10 seconds and ensure the time window is inside the input.

To run this on Windows with the existing venv, replace `python` with
`.\.venv\Scripts\python.exe` and `/` with `\` as desired. This step does NOT import KiMoDo.

## 2 — KiMoDo environment in WSL2

Start from the environment where `~/kimodo` is already installed (`pip install -e '.[all]'`).
If the bridge folder is in a Windows drive, open it as `/mnt/c/...` from WSL2.

```bash
cd /mnt/c/Users/<USER>/Documents/GitHub/handshake_mpc_kimodo_bridge
python -m pip install numpy scipy mujoco
hf auth whoami
```

**Model/text encoder prerequisites:** KiMoDo has previously encountered a 403 for the
gated `meta-llama/Meta-Llama-3-8B-Instruct` model. Authorize the Hugging Face model before
expecting inference to work. In a separate WSL terminal, if the installed version uses
its text encoder service:

```bash
TEXT_ENCODER_DEVICE=cpu kimodo_textencoder
```

Keep it running while invoking the generation step. Do not reinstall CUDA/PyTorch merely
because Hugging Face model authorization is missing.

## 3 — Use an actual SOMA reference, smoke-check the adapter

Use an existing **KiMoDo SOMA motion NPZ**, not ELF3 URDF rotations. The official checkout
provides a useful example (check that this file exists and is not a Git LFS pointer):

```bash
REF="$HOME/kimodo/kimodo/assets/demo/examples/kimodo-soma-rp/04_ee_constraint/motion.npz"
ls -lh "$REF"
```

Alternatively, generate a standing/right-hand reach with KiMoDo once and supply its NPZ.
The NPZ must contain `local_rot_mats` and `root_positions`. Choose a plausible reference
frame with `--reference-frame`; frame 0 is only the default, not necessarily a good pose.

```bash
python -m integration.soma_bridge.generate \
  --targets outputs/bridge/targets.json \
  --reference "$REF" \
  --reference-frame 28 \
  --out outputs/bridge/handshake \
  --check-only
```

This builds the actual `Root2DConstraintSet` and `RightHandConstraintSet` using the
SOMA skeleton and writes `handshake.constraint_diagnostics.json`. It does **not** download
model weights or invoke diffusion. Check the vertical root shift and discrepancy between
the planned tip and the SOMA-calibrated tip.

By default, planned wrist positions are followed, while the SOMA hand endpoint is
reconstructed from the reference wrist-to-`RightHandMiddleEnd` offset. The planner's
0.12 m ELF3-proxy tip offset is **not blindly enforced on SOMA**. The guide target is
shifted vertically by `reference_root_height - initial_plan_pelvis_height` to avoid
putting a humanlike SOMA at an ELF3-proxy ground height. The shift is recorded.

The reference is assumed to face +Z. Use `--reference-yaw <radians>` when this is
not true. A reference-generated wrist orientation is used as the baseline, and
relative changes in the plan's wrist quaternion are applied on top.

## 4 — Generate a constrained SOMA clip

```bash
python -m integration.soma_bridge.generate \
  --targets outputs/bridge/targets.json \
  --reference "$REF" \
  --reference-frame 28 \
  --model Kimodo-SOMA-RP-v1.1 \
  --out outputs/bridge/handshake \
  --diffusion-steps 100
```

Expected files **if inference succeeds**:

```text
outputs/bridge/
  targets.json
  handshake.constraint_diagnostics.json
  handshake.npz
  handshake.bvh
  handshake.skeleton.json
  handshake.metrics.json
```

`handshake.metrics.json` records full-clip and constrained-keyframe wrist/root errors and
max generated wrist and hand-tip speeds. KiMoDo constraint conditioning is *soft*,
so compare the resulting trajectory against the planner rather than assuming exact tracking.

If the model's FPS is not 30, the generation command will stop and print the actual FPS;
rerun step 1 with the model's frame rate. Avoid claiming success based solely on a
well-formed target file.

## 5 — View generated motion in MuJoCo

First verify the motion can be loaded and animated without a display:

```bash
python -m integration.soma_bridge.play_mujoco \
  --motion outputs/bridge/handshake.npz --headless
```

Then, on desktop Windows with WSLg display enabled:

```bash
python -m integration.soma_bridge.play_mujoco \
  --motion outputs/bridge/handshake.npz --loops 2 --speed 1.0
```

The viewer reads `handshake.skeleton.json` written in step 4 and animates the
SOMA joint positions with MuJoCo mocap capsules/spheres, converting Y-up to MuJoCo
Z-up. `--headless` validates the mocap stepping but does not produce a rendered movie.
If WSLg/OpenGL fails, try installing `mujoco` in a local Python environment with
compatible graphical drivers, then copy the NPZ and hierarchy JSON to that machine.

**This is kinematic visualization, not dynamic MuJoCo simulation.** It deliberately
uses no ELF3 URDF or meshes, since SOMA→ELF3 retargeting is an independent stage.

## Troubleshooting gates

1. `targets.py` fails: verify the MPC JSON contains the required full pose channels,
   correct frame convention, finite monotonic times and enough duration.
2. `--check-only` fails: verify KiMoDo is importable, the reference NPZ exists and
   contains valid SOMA local rotations/root positions, and the SOMA skeleton assets exist.
3. Text encoder / 403: resolve access to the gated Meta Llama model; this is not a GPU failure.
4. Model generation fails or hand misses targets: inspect the printed model FPS, reference
   frame, SOMA tip mismatch and planned reference-vs-root offset. Reduce hand movement,
   use more suitable reference poses, or add explicit SOMA arm IK.
5. MuJoCo says joints mismatch: provide the `handshake.skeleton.json` from generation.
6. Skeletal playback succeeds but ELF3 motion looks wrong: this bridge is not yet
   retargeting. Use the partner's SOMA→ELF3 pipeline before actual articulated ELF3 playback.

## Run bridge unit tests (no KiMoDo weights required)

```bash
python -m unittest discover -s tests -p 'test_soma_bridge.py' -v
```
