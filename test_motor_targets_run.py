import os
import sys
import time
import jax
import jax.numpy as jp
import numpy as np

# Ensure repo root on path
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from para_env.grasp_env import ParaHandGrasp


def main():
    env = ParaHandGrasp()
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)

    motor_targets = jp.array([
        0.28063294,  0.16764747,  0.767294,   -0.03304268,
       -0.7112028,  -0.56188554,  0.6638142,  -0.10870725,
       -0.03764224,  0.0901849,  -0.00373352,  0.8705306,
        0.41919628, -0.7833171,  -0.7029478,   0.11265983,
       -0.30171746,  0.779827,   -0.0172186,  -0.49334955,
       -0.5959107,  -0.02602314
    ])

    default_ctrl = env._default_ctrl
    action_scale = env._config.action_scale
    action = (motor_targets - default_ctrl) / action_scale

    print("actuator count:", env.action_size)
    try:
        ctrlrange = env.mjx_model.actuator_ctrlrange
        print("ctrlrange[:action_size]:", np.array(ctrlrange[: env.action_size]))
    except Exception as e:
        print("could not read ctrlrange:", e)

    for i in range(20):
        state = env.step(state, action)
        qpos_hand = np.array(state.data.qpos[np.array(env._hand_qids)])
        tips = np.array(env.get_tips_positions(state.data))
        print(f"step {i}")
        print(" hand qpos (first 12):", qpos_hand[:12])
        for j, name in enumerate(['thumb','index','middle','ring','little']):
            print(f"  {name} tip:", tips[j])
        time.sleep(0.05)


if __name__ == '__main__':
    main()
