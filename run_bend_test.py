import os
import sys
import time
import jax
import jax.numpy as jp
import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from para_env.grasp_env import ParaHandGrasp

# actuator index mapping (for reference):
# [0:3] thumb joints
# [4:7] swing joints
# [8:11] finger joint_0 (closest to palm)
# [12:15] tendon motors
# [16:21] palm (slide/rotate)

motor_targets_bend = jp.array([
    0.28063294,  0.16764747,  0.767294,   -0.03304268,   # thumb
   -0.7112028,  -0.56188554,  0.6638142,  -0.10870725,   # swings
    0.8, 0.8, 0.8, 0.8,                     # proximal finger joints -> encourage bending
   -1.0, -1.0, -1.0, -1.0,                 # tendon motors -> pull/tighten
   -0.30171746,  0.779827,   -0.0172186,  -0.49334955, -0.5959107,  -0.02602314  # palm
])

def main():
    env = ParaHandGrasp()
    rng = jax.random.PRNGKey(1)
    state = env.reset(rng)

    default_ctrl = env._default_ctrl
    action_scale = env._config.action_scale
    action = (motor_targets_bend - default_ctrl) / action_scale

    print("Applying modified motor_targets to encourage bending (proximal joints=0.8, tendons=-1)")
    for i in range(30):
        # ramp tendon targets gradually in first 5 steps to avoid instability
        if i < 5:
            factor = (i+1)/5.0
            ramp_targets = motor_targets_bend.at[12:16].set((1-factor)*0.0 + factor * motor_targets_bend[12:16])
            action = (ramp_targets - default_ctrl) / action_scale
        state = env.step(state, action)
        qpos_hand = np.array(state.data.qpos[np.array(env._hand_qids)])
        tips = np.array(env.get_tips_positions(state.data))
        print(f"step={i}")
        print(" hand qpos[:12]:", qpos_hand[:12])
        for j, name in enumerate(['thumb','index','middle','ring','little']):
            print(f"  {name} tip:", tips[j])
        time.sleep(0.05)

if __name__ == '__main__':
    main()
