import os, sys, time
os.environ['XLA_FLAGS'] = '--xla_gpu_graph_min_graph_size=1' 

import jax
import jax.numpy as jp
import mujoco as mj
from mujoco import mjx
import imageio

from para_env.para_hand_env import TestTask

def main():
    # --- 初始化 ---
    env = TestTask()
    mj_model = env.mj_model
    mj_data = mj.MjData(mj_model)

    # 设置场景选项以显示关节
    scene_option = mj.MjvOption()
    scene_option.flags[mj.mjtVisFlag.mjVIS_JOINT] = True

    # 初始化渲染器和帧列表
    renderer = mj.Renderer(mj_model, height=480, width=640)
    frames = []

    # 重置环境到初始状态
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    
    # --- 仿真循环 ---
    duration = 1  # 仿真总时长（秒）
    framerate = 30 # 视频帧率 (fps)
    print("开始仿真...")
    
    start_time = time.time()
    
    ctrl = state.data.ctrl
    # ctrl = ctrl.at[0].set(-1.0)

    while state.data.time < duration:
        # 执行一步仿真
        state = env.step(state, ctrl)
        
        if len(frames) < state.data.time * framerate:
            # 将 JAX 数据同步到 MuJoCo 数据
            mjx.get_data_into(mj_data, mj_model, state.data)
            # 更新场景并渲染
            renderer.update_scene(mj_data, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)
        # jax.debug.print("仿真时间: {:.1f} 秒", state.data.time)

    # --- 清理和保存 ---
    renderer.close() # 释放渲染器资源
    
    end_time = time.time()
    print(f"仿真耗时: {end_time - start_time:.2f} 秒")

    output_path = 'simulation_output.mp4'
    print(f"仿真完成，正在将视频保存到 {output_path}...")
    with imageio.get_writer(output_path, fps=framerate) as writer:
        for frame in frames:
            writer.append_data(frame)
    print("视频保存成功！")
    # ----------------

if __name__ == "__main__":
    main()