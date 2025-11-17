import os
# os.environ['XLA_FLAGS'] = '--xla_gpu_graph_min_graph_size=1' 

import time
import numpy as np
import jax
import jax.numpy as jp
import mujoco as mj
from mujoco import mjx
from mujoco.glfw import glfw

from para_env.para_hand_env import TestTask
from sample.glfw_viewer_control import setup_mouse_callbacks

def main():
    # 初始化 GLFW
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    # 创建 GLFW 窗口
    width, height = 640, 480  # 调整为较低分辨率
    window = glfw.create_window(width, height, "MuJoCo Simulation", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    glfw.make_context_current(window)  # 激活 OpenGL 上下文

    # 创建环境实例
    env = TestTask()
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)

    data = mj.MjData(env.mj_model)  # 创建 MjData 用于渲染

    # 创建 MuJoCo 渲染上下文
    scene = mj.MjvScene(env.mj_model, maxgeom=1000)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, 0.1]
    cam.distance = 1.0
    cam.elevation = -20
    cam.azimuth = 90
    context = mj.MjrContext(env.mj_model, mj.mjtFontScale.mjFONTSCALE_150)  # 必须在 OpenGL 上下文激活后创建

    # 创建 mjvOption 和 mjvPerturb
    opt = mj.MjvOption()
    pert = mj.MjvPerturb()

    # 初始化鼠标交互
    mouse_state = setup_mouse_callbacks(window, env.mj_model, scene, cam, width, height)

    # 主仿真循环
    while not glfw.window_should_close(window):
        # 检查 state 是否有效
        jax.debug.print("state before step is valid: {}", jp.all(jp.isfinite(state.data.qpos)))

        # 控制逻辑：让手模型的某个关节来回移动
        # ctrl = ctrl.at[0].set(-1.0)  # 设置控制信号

        # 执行一步仿真
        ctrl = state.data.ctrl
        # jax.debug.print("ctrl before step: {}", ctrl)
        state = env.step(state, ctrl)

        # 检查 state 是否有效
        jax.debug.print("state after step is valid: {}", jp.all(jp.isfinite(state.data.qpos)))
        jax.debug.print("qpos: {}", state.data.qpos)
        jax.debug.print("qvel: {}", state.data.qvel)
        jax.debug.print("ctrl: {}", state.data.ctrl)

        # 将 mjx 的新状态同步到 MjData 以供渲染
        mjx.get_data_into(data, env.mj_model, state.data)

        # 渲染到窗口
        viewport = mj.MjrRect(0, 0, width, height)
        mj.mjv_updateScene(env.mj_model, data, opt, pert, cam, mj.mjtCatBit.mjCAT_ALL, scene)
        mj.mjr_render(viewport, scene, context)

        # 交换缓冲区并处理事件
        glfw.swap_buffers(window)
        glfw.poll_events()

        # 控制渲染速度
        time.sleep(0.01)  # 控制仿真速度

    # 清理资源
    glfw.destroy_window(window)
    glfw.terminate()
    print("仿真结束")

if __name__ == "__main__":
    main()