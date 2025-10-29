"""
爆显存测试，通过长时间运行环境的 step 函数来观察显存使用情况
使用了多个不同的xml文件观察显存变化情况
"""

import os
import jax
import jax.numpy as jp
import mujoco as mj
from mujoco import mjx
import mediapy as media
import pickle

from para_env.para_hand_env import TestTask
from mujoco_playground import registry

# 确保运行在 GPU 上
os.environ["JAX_PLATFORM_NAME"] = "gpu"

env = TestTask()
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

state = jit_reset(jax.random.PRNGKey(0))

# 初始化控制信号
ctrl = jp.zeros(18)
ctrl = ctrl.at[0].set(-18)
ctrl = ctrl.at[5].set(-18)
ctrl = ctrl.at[7].set(-18)
ctrl = ctrl.at[8].set(-18)
ctrl = ctrl.at[10].set(-18)

# 分批仿真和渲染
rollout = [state]
total_steps = 1000

# 尝试不同 steps 数量的 rollout
for step in range(total_steps):
    state = jit_step(state, ctrl)
    rollout.append(state)
print(f"{total_steps} Test Completed.\n")
print(env.xml_path)

# # 设置摄像机参数
# camera = mj.MjvCamera()
# camera.type = mj.mjtCamera.mjCAMERA_FREE
# camera.lookat[:] = [0, 0, 0.2]  # 对准手部模型的中心位置
# camera.distance = 0.3           # 缩短摄像机距离
# camera.azimuth = 90             # 从侧面观察
# camera.elevation = -20          # 从略微向下的角度观察

# frames = env.render(rollout, height=480, width=640, camera=camera)
# media.write_video(f"./video/oomTest_obj_tac_{total_steps}.mp4", frames, fps=30)

# 保存 rollout 到本地文件
with open("rollout.pkl", "wb") as f:
    pickle.dump(rollout, f)
print("Rollout has been saved to 'rollout.pkl'.")