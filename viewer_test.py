import time
import mujoco
import mujoco.viewer
from para_env.para_hand_env import TestTask
import jax
import jax.numpy as jp

def main():
    # 创建环境实例
    env = TestTask()
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)

    # 创建 MuJoCo 模型和数据
    m = env.mj_model
    d = mujoco.MjData(m)

    # 初始化控制信号S
    ctrl = jp.zeros(m.nu)

    # 启动 MuJoCo Viewer 的被动模式
    # with mujoco.viewer.launch_passive(m, d) as viewer:
    #     # 设置 Viewer 自动关闭时间（30 秒）
    #     start = time.time()
    #     while viewer.is_running() and time.time() - start < 30:
    #         step_start = time.time()

    #         # 应用控制信号并执行一步仿真
    #         state = env.step(state, ctrl)
    #         mujoco.mj_step(m, d)

    #         # 将仿真状态同步到 Viewer
    #         mjx.get_data_into(d, m, state.data)

    #         # 示例：每两秒切换一次接触点的可视化
    #         with viewer.lock():
    #             viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    #         # 同步 Viewer 的状态
    #         viewer.sync()

    #         # 控制仿真速度
    #         time_until_next_step = m.opt.timestep - (time.time() - step_start)
    #         if time_until_next_step > 0:
    #             time.sleep(time_until_next_step)

    while 1:
        state = env.step(state, ctrl)
        print(state.data.qpos)
        time.sleep(0.1)
        # mjx.get_data_into(d, m, state.data)



if __name__ == "__main__":
    main()