# Handshake MPC — Stage 1 + optional KiMoDo/MuJoCo bridge

**New integration:** See [KIMODO_MUJOCO_README.md](KIMODO_MUJOCO_README.md) for the
Windows MPC → task-space JSON → KiMoDo-SOMA inference → MuJoCo offline skeleton
playback commands. The original Stage-1 Windows MPC remains unchanged.

## Stage-1 reference documentation


A standalone, CPU-only Python repository for testing coupled body approach and
right-hand reaching. It has no KiMoDo, MuJoCo, PyTorch, CUDA, ROS, or WSL dependency.
The controller produces task-space reference trajectories, not joint commands.

The optimizer controls **pelvis XZ and wrist XYZ**. Pelvis height, body heading,
torso orientation, and wrist orientation are fixed configured values in this
stage. Every output includes the five requested motion-generation channels:
**hand-tip position, wrist position, wrist orientation, torso position, torso
orientation**. It also includes the pelvis/root pose and velocities.

## Windows quick start

Install **64-bit Python 3.12**. The Python launcher (`py`) is optional: setup
tries `py -3.12`, then checks `python` and `python3` on `PATH` for a supported
64-bit Python 3.11–3.13. Extract the ZIP, open PowerShell in this folder, and run:

```powershell
.\setup_windows.bat
.\run_tests.bat
.\run_demo.bat
```

Setup creates a local virtual environment and uses binary dependency wheels.
Activation and PowerShell execution-policy changes are unnecessary. All demos
produce plots and JSON/CSV files under `outputs`. The default command runs all
seven experiments; a failed behavioral check gives a nonzero exit status.

Manual installation, or if you prefer your existing environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m handshake_mpc.demo --scenario all
```

Python 3.11–3.13 is allowed; dependency resolution and binary-wheel availability
were checked for **Python 3.12 / Windows x64**. Actual numerical tests were run
on Linux, not Windows. CVXPY and Clarabel provide the optimization backend.
If your machine cannot resolve a binary dependency, use 64-bit Python 3.12
and a current pip. The repo does not require a compiler-based solver build.

## What the demos test

```powershell
.\run_demo.bat --scenario far
.\run_demo.bat --scenario moving
.\run_demo.bat --scenario terminate
.\run_demo.bat --scenario all --sample-rate 30 --record-horizons
.\run_demo.bat --config configs\default.json --output outputs\my_run
```

| Scenario | Experiment |
|---|---|
| `near` | Reach a nearby tip target, with little pelvis movement |
| `far` | Approach a distant target while extending the arm |
| `moving` | Replan against a smoothly moving tip target |
| `terminate` | Terminate after approaching; retract toward neutral and hold/brake the pelvis |
| `moving_terminate` | Terminate during approach; bounded body braking and concurrent retraction |
| `mismatch` | Far target with a synthetic plant applying 80% of requested accelerations |
| `unreachable` | Target too high for the fixed-height reach ball; report residual instead of pretending success |

The plant is a synthetic double integrator. It applies the first MPC control
interval, reports its resulting position/velocity, and the controller solves
again from that state. The mismatch test uses actual synthetic feedback rather
than copying the predicted next state. It is one empirical perturbation test,
not a proof of robustness to arbitrary tracker errors.

Each scenario writes:

| File | Contents |
|---|---|
| `trajectory.png` | Measured synthetic paths, error, and speeds |
| `executed_reference.json` | Concatenated first-interval references at 50 Hz by default; all pose channels |
| `executed_reference.csv` | Same references as flat CSV |
| `measured_state.csv` | Synthetic plant feedback at 10 Hz |
| `last_predicted_horizon.json` | Last complete 3-second predicted horizon |
| `metrics.json` | Behavioral checks, errors, constraints, and solve times |
| `predicted_horizons.jsonl` | Every full horizon, when `--record-horizons` is enabled |

`outputs/summary.json` contains aggregate checks and dependency versions.
`validation/` contains results from the delivered revision.

## MPC formulation

The state is position and velocity for five task-space coordinates:

```text
position = [pelvis_x, pelvis_z, wrist_x, wrist_y, wrist_z]
velocity = [pelvis_vx, pelvis_vz, wrist_vx, wrist_vy, wrist_vz]
control  = [pelvis_ax, pelvis_az, wrist_ax, wrist_ay, wrist_az]
```

At each interval, acceleration is constant:

\[
p_{k+1}=p_k+\Delta t v_k+\tfrac12\Delta t^2a_k,
\qquad v_{k+1}=v_k+\Delta t a_k.
\]

The quadratic objective combines wrist tracking, terminal wrist error, a soft
comfortable-region penalty, pelvis movement, velocities, acceleration, and
changes in acceleration. The hard constraints bound pelvis/wrist speed and
acceleration, plus the wrist–pelvis distance:

\[
\|p_{w,k}-p_{b,k}\|_2\le L_{\max}.
\]

The comfortable set is a box in the pelvis-local frame:

\[
|R_b^T(p_w-p_b)-r_c|\le d+s,\quad s\ge0,
\]

with a quadratic penalty on the per-axis slack `s` in meters. The fixed heading
makes this affine; the reach ball and norm limits produce a convex SOCP.
Clarabel solves it, and CVXPY caches the parameterized model between updates.

The pelvis displacement reference is **the current measured pelvis position at
each solve**, not the original episode position. This discourages walking in
each horizon without permanently trading target error against total distance
traveled. Nearby targets can still cause a small body adjustment: body motion
is penalized, not locked during an active handshake.

Near the target, the wrist velocity penalty increases smoothly:

\[
g=\exp[-d^2/(2\sigma^2)],\qquad
w_v=w_{v,\mathrm{far}}+(w_{v,\mathrm{near}}-w_{v,\mathrm{far}})g.
\]

`d` is the current measured tip-to-predicted-target distance. The gate is
computed before solving and held constant across that horizon, preserving
convexity. It updates at the next feedback sample. This gives continuous
precontact-like slowing without an approach/reach/precontact state machine.
It is a **velocity cost**, not a certified near-contact speed limit or a
contact detector. Reaching the objective is not confirmation of physical contact.

Reach is checked beyond the time knots. Each interval has a quadratic Bezier
representation; its endpoints and middle control point are constrained inside
the reach ball. The convex-hull property then bounds the whole interpolated
interval. Speed bounds also hold throughout because velocity is affine in time.
Acceleration can jump between intervals; the change penalty reduces jumps but
does not impose a hard jerk bound or guarantee continuous acceleration.

## Controller interface

```python
import numpy as np
from handshake_mpc import (
    HandshakeMPC, MPCConfig, RobotState, HandshakeTarget, PlannerCommand,
)

config = MPCConfig.load("configs/default.json")
controller = HandshakeMPC(config)

# Obtain these from your estimator; demo values shown here.
state = RobotState(
    timestamp_s=0.0,
    pelvis_position_m=np.array([0.0, 0.95, 0.0]),
    pelvis_orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    right_wrist_position_m=np.array([0.22, 0.75, 0.05]),
    right_wrist_orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    pelvis_velocity_mps=np.zeros(3),
    right_wrist_velocity_mps=np.zeros(3),
)
target = HandshakeTarget(
    timestamp_s=0.0,
    hand_tip_position_m=np.array([0.22, 1.15, 2.20]),
)
plan = controller.update(state, target, PlannerCommand(initiated=True))
reference = plan.sample(0.1)
plan.save_json("outputs/plan.json", sample_rate_hz=30)

# Execute only the first 0.1 seconds, obtain fresh measured state and target,
# then call update again. Neither flag means continue the previous command.
# controller.update(new_state, new_target)

# Termination does not require a human target:
# controller.update(new_state, command=PlannerCommand(terminated=True))
```

`PlannerCommand` has **event semantics**. Initiation remains active until
termination; termination remains latched until another initiation or `reset()`.
Termination wins when both flags are true. Before any initiation, the controller
brakes/holds the body and wrist. The only command selection is active,
terminated, or inactive; there are no internal interaction phases.

On termination, the controller fixes a stationary pelvis, or imposes a bounded
straight-line braking trajectory if it has nonzero velocity. The wrist target
is the configured neutral offset relative to that pelvis trajectory. Retracting
does not cause extra body repositioning. An instantaneous freeze of a moving
body is deliberately not attempted.

The controller assumes updates happen after the first control interval has
been applied. Its acceleration-change cost remembers the previously issued
first acceleration. If your actuator applies something different, that memory
is only a smoothness reference; measured position and velocity still initialize
the new horizon.

All timestamps share a monotonic clock. Positions are meters in `planner_world`,
Y is up, and the initial forward axis is +Z. Quaternions are `[x,y,z,w]`, active
local-to-world rotations. An optional constant-velocity target prediction is
capped at 0.6 seconds by default, including observation age. Stale/future target
timestamps and mismatched frames are rejected. Velocities are inputs; an
estimator or finite-difference filter must provide them when integrating vision.

## The five KiMoDo-adapter channels

`reference = plan.sample(t)` contains:

| Key | Shape | Meaning |
|---|---|---|
| `right_hand_tip_position_m` | `(3,)` | Fixed reference point on the right hand |
| `right_wrist_position_m` | `(3,)` | Wrist origin |
| `right_wrist_orientation_xyzw` | `(4,)` | Wrist-local to world orientation |
| `torso_position_m` | `(3,)` | Torso origin derived from pelvis and configured offset |
| `torso_orientation_xyzw` | `(4,)` | Fixed torso orientation |
| `pelvis_position_m` | `(3,)` | Actual modeled root origin; retained separately from torso |
| `pelvis_orientation_xyzw` | `(4,)` | Fixed upright body heading |

The tip and torso transforms are:

\[
p_{\mathrm{tip}}=p_{\mathrm{wrist}}+R_{\mathrm{wrist}}r_{\mathrm{wrist\to tip}},
\qquad
p_{\mathrm{torso}}=p_{\mathrm{pelvis}}+R_b r_{\mathrm{pelvis\to torso}}.
\]

The **tip offset is a vector in the wrist-local frame**, not a world-frame
translation or just a scalar length. It is a constant only while finger
configuration is fixed. Choose the actual SOMA point/joint you mean by "tip"
and calibrate this vector later. Likewise, torso and pelvis origins must not
be conflated. Stage 1 assumes a rigid upright torso with a calibrated offset.

All these constants, neutral pose, comfortable box, reach radius, and limits
live in `configs/default.json`. They are deliberately **illustrative**, not
measured skeleton geometry or verified robot limits. Wrist orientation is
emitted but held fixed; requesting another orientation raises a clear error.
Full wrist/torso angular planning belongs to the next stage.

The target input is a **desired robot tip position**, not a raw observed human
palm. `HandshakeTarget.from_human_palm(...)` provides an explicit optional
mapping with a human-palm-local offset. Its contact geometry must be calibrated;
the helper assumes no palm rotation when propagating velocity. No particular
human-palm/fingertip coincidence is silently assumed.

## KiMoDo integration boundary

The versioned JSON export `handshake_task_space/v1` contains the synchronized
channels above, times, frame indices, frame convention, geometry, and solver
diagnostics. `docs/task_space_plan.schema.json` describes it. Resampling is
exact within each constant-acceleration interval; no spline can overshoot the
optimized reference. Exported angular channels remain constant in Stage 1.

**This JSON is our task-space adapter input, not native KiMoDo constraint JSON.**
KiMoDo's documented `root2d` constraints take XZ positions and optional heading
direction pairs. Its documented `right-hand`/`end-effector` JSON also takes
`local_joints_rot` and `root_positions`, from which skeletal FK produces the
end-effector constraints. Those local joint rotations cannot be inferred from
these five task-space channels alone. A reference-pose + IK/skeleton adapter is
still required for that route. A direct end-effector tensor API is another
possible integration route once the installed version and skeleton are known.

The future adapter should:

1. Map planner world coordinates and quaternion bases to the KiMoDo convention.
2. Select the SOMA pelvis, torso, wrist, and tip joint/reference-point identities.
3. Resample/export at the generator's actual frame rate.
4. Use a nominal reference pose plus IK, or the installed direct constraint API.
5. Convert the generated SOMA clip back into these task-space channels and report
   root/wrist/tip errors separately from this MPC's predicted errors.

The upstream schema and documentation checked when preparing this repo:

- https://research.nvidia.com/labs/sil/projects/kimodo/docs/user_guide/constraints.html
- https://github.com/nv-tlabs/kimodo/blob/main/kimodo/constraints.py
- https://www.cvxpy.org/install/index.html
- https://www.cvxpy.org/tutorial/solvers/index.html

## Practical boundaries

The pelvis–wrist ball couples body and arm movement but does **not** encode joint
limits, self-collision, legged balance, foot placement, dynamic walking feasibility,
or human keep-out distance. A point inside the ball need not have an IK solution.
The comfort box is a preference, not a certified capability map. This repo does
not regulate contact forces or generate handshake oscillations.

Hard constraints are enforced on the **predicted nominal trajectory**. They are
not invariant guarantees for an arbitrary downstream tracker. Even an initially
in-bound state can be nonviable with outward momentum near the reach boundary.
There is no recursive-feasibility or stability proof, terminal invariant set,
or robust tube in this Stage 1 prototype. The default comfort weighting leaves
reserve in the tested approach scenario, but that reserve is empirical.

Failed or infeasible solves raise `MPCSolveError`; invalid/stale inputs raise
`ValueError`. A failed horizon is never exported as a successful plan. When
integrating a tracker, the caller must handle failure with its own available
stop/hold behavior. Target error is soft: an unreachable target can yield a
valid constrained plan with a persistent residual, as the `unreachable` demo
shows. No real robot should treat this software-only test as a safety controller.

## Source map

| File | Responsibility |
|---|---|
| `handshake_mpc/controller.py` | Convex model, feedback update, command latch, braking |
| `handshake_mpc/config.py` | Geometry, limits, weights, configuration validation |
| `handshake_mpc/interfaces.py` | Input dataclasses, pose channels, interpolation, JSON export |
| `handshake_mpc/geometry.py` | Quaternion/frame and rigid-point operations |
| `handshake_mpc/simulation.py` | Synthetic feedback plant |
| `handshake_mpc/demo.py` | Scenarios, CSV/JSON export, plots, behavioral checks |
| `tests/test_mpc.py` | Interface, geometry, constraints, retraction, failure, convergence |
| `examples/minimal_loop.py` | Small reusable control-loop example |

Tune `comfort_weight` and `pelvis_displacement_weight` to change the balance
between arm extension and body approach. Change one at a time and inspect both
path and speed plots. Reduce `hand_tracking_weight` or increase velocity and
acceleration costs for a gentler approach. The default near-contact cost does
not prescribe an exact speed profile. `horizon_steps * dt_s` is the prediction
horizon; the execution/replanning interval is `dt_s`, independently of export fps.
