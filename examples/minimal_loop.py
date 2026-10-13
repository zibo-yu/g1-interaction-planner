"""Run from the repository root: python examples/minimal_loop.py (after pip install -e .)."""
import numpy as np
from handshake_mpc import HandshakeMPC, MPCConfig, HandshakeTarget, PlannerCommand
from handshake_mpc.simulation import neutral_state, advance_plant

config = MPCConfig()
controller = HandshakeMPC(config)
state = neutral_state(config)
for step in range(100):
    target = HandshakeTarget(state.timestamp_s, np.array([0.22, 1.15, 2.2]))
    plan = controller.update(state, target, PlannerCommand(initiated=step == 0))
    # A single sample contains all five requested KiMoDo-adapter inputs.
    reference = plan.sample(state.timestamp_s + config.dt_s)
    state = advance_plant(state, plan, config.dt_s)
plan.save_json("outputs/example_predicted_horizon.json", sample_rate_hz=50)
print("Torso:", reference["torso_position_m"])
print("Wrist:", reference["right_wrist_position_m"])
print("Tip:", reference["right_hand_tip_position_m"])
