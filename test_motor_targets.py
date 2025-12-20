import jax
import jax.numpy as jp
import numpy as np
from para_env.grasp_env import ParaHandGrasp
from para_env import para_hand_constants as consts


def main():
    env = ParaHandGrasp()
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)

    # 用户提供的 motor targets
    motor_targets = jp.array([
        0.28063294,  0.16764747,  0.767294,   -0.03304268,
       -0.7112028,  -0.56188554,  0.6638142,  -0.10870725,
       -0.03764224,  0.0901849,  -0.00373352,  0.8705306,
        0.41919628, -0.7833171,  -0.7029478,   0.11265983,
       -0.30171746,  0.779827,   -0.0172186,  -0.49334955,
       -0.5959107,  -0.02602314
    ])

    # Convert motor_targets to action expected by env: motor_targets = default_ctrl + action*action_scale
    default_ctrl = env._default_ctrl
    action_scale = env._config.action_scale
    action = (motor_targets - default_ctrl) / action_scale

    print("Applying motor_targets as constant action for 50 steps")
    print("actuator count:", env.action_size)

    # Print ctrlrange
    try:
        ctrlrange = env.mjx_model.actuator_ctrlrange
        print("actuator ctrlrange (first 22 rows):")
        print(np.array(ctrlrange[: env.action_size]))
    except Exception as e:
        print("Could not read ctrlrange:", e)

    for i in range(50):
        state = env.step(state, action)
        qpos_hand = state.data.qpos[jp.array(env._hand_qids)]
        tips = env.get_tips_positions(state.data)
        palm = env.get_palm_position(state.data)
        cube = env.get_cube_position(state.data)
        # Convert to numpy for readability
        qpos_hand_np = np.array(qpos_hand)
        tips_np = np.array(tips)
        palm_np = np.array(palm)
        cube_np = np.array(cube)
        print(f"step={i}")
        print(" hand qpos (len={}):".format(len(qpos_hand_np)), qpos_hand_np)
        print(" fingertip positions:")
        for j, name in enumerate(['thumb','index','middle','ring','little']):
            print(f"  {name}: {tips_np[j]}")
        print(" palm:", palm_np, " cube:", cube_np)


if __name__ == '__main__':
    main()
