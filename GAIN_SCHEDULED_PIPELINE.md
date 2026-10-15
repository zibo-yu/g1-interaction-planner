# Gain-scheduled handshake: MPC → KiMoDo-SOMA → offline MuJoCo

This version retains the original V1 controller and bridge; **new behavior is opt-in** via `configs/elf3_gain_scheduled.json`. No ELF3 joint-space retargeting or dynamic control is implied.

## Architecture and parameter meanings

* **MPC, closed-loop synthetic-plant demonstration:** RobotState contains measured pelvis and wrist poses/velocities. HandshakeTarget contains a `human_torso_position_m` and an optional desired robot hand-tip; torso-only observations are accepted while the gain is zero. At each 0.1 s update, horizontal pelvis-to-human distance determines a smoothstep gain `alpha`: 0 above 2 m, 1 below 1.2 m.
* **Root target:** human torso XZ minus a provisional 0.85 m standing separation along the approach bearing. This prevents the previous problem where lowering the hand weight removed all motivation to walk.
* **Observation contract:** In far walking (gain zero), `HandshakeTarget(timestamp_s=t, hand_tip_position_m=None, human_torso_position_m=torso)` is valid. Once gain becomes positive, a hand-tip observation is mandatory. Wrist *state* is initialized at its nominal rest offset in the demonstration; in future online use it must come from the measured robot state.
* **Objective:** independent root destination tracking, `(1-alpha)` neutral-relative wrist tracking, `alpha` hand tracking/comfort, plus unchanged acceleration/speed/reach constraints. Nominal wrist offset is based on the supplied ELF3 proxy configuration. Actual SOMA wrist pose remains model-dependent.
* **Bridge scheduling:** original uninterrupted export is augmented with `scheduling.reach_gain` per frame. In `targets.py`, the hand keyframes begin at the **first** frame with gain ≥ 0.6; no right-hand constraints are provided before that point. The root path can be constrained densely or at sparse waypoints.
* **KiMoDo:** the model uses two sequential prompts when the gate splits a clip into nonempty walking and reaching segments. Uses the built-in `multi_prompt=True` transition support; all constraint frame indices are global.
* **MuJoCo:** offline playback of the generated **SOMA skeleton** using the pre-existing mocap-stick-figure viewer. It is *not* ELF3 and is *not* a simulated physical handshake.

`configs/elf3_gain_scheduled.json` is a provisional benchmark configuration. It uses ELF3 geometry and synthetic observations, not actual balance, COM, or contact-force constraints. The model remains fixed-heading and fixed-height. Only the right wrist is controlled, not two arms.

## Environments

The full MPC optimizer requires Python 3.11–3.13 with the package dependencies. KiMoDo is installed separately in your working Conda `kimodo` Python 3.10 environment. Do not install the MPC's pinned `requirements.txt` into the KiMoDo environment. The bridge and its tests need NumPy/SciPy but not CVXPY.

### Part A: run the MPC rollout (Python 3.11–3.13)

From project root, create a separate Conda environment for the MPC:

```bash
conda create -n mpc-v2 python=3.12 -y
conda activate mpc-v2
python -m pip install -e .
python -m unittest discover -s tests -p 'test_mpc.py' -v
python -m unittest discover -s tests -p 'test_gain_scheduled_mpc.py' -v
python examples/gain_scheduled_pipeline.py \
  --config configs/elf3_gain_scheduled.json \
  --duration 10 --fps 30 \
  --out outputs/gain_scheduled/executed_reference.json
```

The demonstration simulates a person at torso `[−0.02, 1.35, 3.20]` m and contact tip `[−0.178, 1.05, 2.75]` m, and explicitly re-plans from the simulated plant state every 0.1 s. Outputs include `scheduling.reach_gain` and `scheduling.human_torso_distance_m`. No KiMoDo or MuJoCo is needed yet.

### Part B: bridge targets (Conda `kimodo`, Python 3.10)

```bash
conda activate kimodo
python -m unittest discover -s tests -p 'test_soma_bridge.py' -v
python -m unittest discover -s tests -p 'test_gain_scheduling.py' -v
python -m integration.soma_bridge.targets \
  --plan outputs/gain_scheduled/executed_reference.json \
  --start 0 --duration 10 --fps 30 \
  --root-stride 15 --hand-stride 12 --hand-gate-threshold 0.6 \
  --out outputs/gain_scheduled/targets.json
```

The target output reports the automatically chosen `phase_split_frame`, root waypoint count, and hand keyframe count. Hand keyframes before this frame are absent. `--root-stride 1` gives dense root conditioning; `--root-stride 15` gives waypoints every 0.5 s and should be compared for gait quality. In the legacy mode without scheduling metadata, hand constraints still start at frame zero unless `--hand-start-s` or `--no-hand` is used.

### Part C: SOMA constraints and KiMoDo generation

Use your **already converted** SOMA reference NPZ from the previous working run:

```bash
REF=outputs/bridge/soma_reference.npz
python -m integration.soma_bridge.generate \
  --targets outputs/gain_scheduled/targets.json \
  --reference "$REF" --reference-frame 0 \
  --out outputs/gain_scheduled/handshake --check-only
```

If this passes, start the KiMoDo text encoder service in a separate terminal:

```bash
conda activate kimodo
TEXT_ENCODER_DEVICE=cpu kimodo_textencoder
```

Then, from project root:

```bash
export TEXT_ENCODER_MODE=api
export LOCAL_CACHE=true
python -m integration.soma_bridge.generate \
  --targets outputs/gain_scheduled/targets.json \
  --reference "$REF" --reference-frame 0 \
  --model Kimodo-SOMA-RP-v1.1 --diffusion-steps 20 \
  --out outputs/gain_scheduled/handshake
```

This automatically uses two prompts if the gate lies within the 10 s clip: walking first, then slowing/stopping/reaching, with a default 8-frame transition. `--num-transition-frames 5` can be used if a segment would otherwise be too short. `--single-prompt` disables segmentation for a comparison baseline. The output includes NPZ, BVH, hierarchy, and JSON metrics. For better generation fidelity, increase diffusion steps after the smoke test.

### Part D: MuJoCo playback

```bash
python -m pip install mujoco
python -m integration.soma_bridge.play_mujoco \
  --motion outputs/gain_scheduled/handshake.npz --headless
python -m integration.soma_bridge.play_mujoco \
  --motion outputs/gain_scheduled/handshake.npz --loops 3 --speed 1.0
```

### Ablations and debugging

**Root-only walking**, with no hand constraints, for the first five seconds:

```bash
python -m integration.soma_bridge.targets \
  --plan outputs/gain_scheduled/executed_reference.json \
  --duration 5 --fps 30 --root-stride 15 --no-hand \
  --out outputs/gain_scheduled/walk_only_targets.json
python -m integration.soma_bridge.generate \
  --targets outputs/gain_scheduled/walk_only_targets.json \
  --reference "$REF" --reference-frame 0 --diffusion-steps 20 \
  --out outputs/gain_scheduled/walk_only
python -m integration.soma_bridge.play_mujoco \
  --motion outputs/gain_scheduled/walk_only.npz --loops 2
```

Compare gait with `--root-stride 1`, then test the two-prompt run. To inspect systematic 70 cm wrist errors, read `handshake.metrics.json`, especially `wrist_keyframe_mean_m`, `pelvis_relative_wrist_keyframe_mean_m`, and `root_mean_m`. During root-only walking the keyframe wrist errors are `null` by design; the full-frame wrist error is labeled as including unconstrained frames and is **not** an adherence metric.

**Known limitation:** The bridge still writes direct Cartesian right-hand positions into in-memory KiMoDo constraints. Those positions are not IK-consistent by construction. Gain scheduling may improve movement timing and gait, but there is no guarantee it resolves the previously measured 0.70 m hand-target error. A future SOMA shoulder/elbow IK calibration or alternative constraint pathway may be required.

## GitHub integration

This ZIP is a modified source snapshot, not a GitHub push. Copy the changed files to the GitHub checkout on the `MPC` branch, then run `git diff`, commit, push, and `git pull` from Ubuntu. Alternatively use the accompanying unified `.patch` with `git apply --check ...` and `git apply ...` from the repository root. **Do not overwrite your own post-snapshot edits without reviewing differences.**
