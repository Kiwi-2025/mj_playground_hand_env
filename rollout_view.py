import os, sys, time
import gc
gc.collect()
import jax
import jax.numpy as jp
import mujoco as mj
from mujoco import mjx
import mediapy as media

from para_env.para_hand_env import TestTask
from mujoco_playground import registry

# 强制使用 GPU 进行推理
os.environ["JAX_PLATFORM_NAME"] = "gpu"

env = TestTask()
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# 设置摄像机参数
camera = mj.MjvCamera()
camera.type = mj.mjtCamera.mjCAMERA_FREE
camera.lookat[:] = [0, 0, 0.2]  # 对准手部模型的中心位置
camera.distance = 0.3           # 缩短摄像机距离
camera.azimuth = 90             # 从侧面观察
camera.elevation = -20          # 从略微向下的角度观察

state = jit_reset(jax.random.PRNGKey(0))

# 初始化控制信号
ctrl = jp.zeros(18)
ctrl = ctrl.at[0].set(-18)
ctrl = ctrl.at[5].set(-18)
ctrl = ctrl.at[7].set(-18)
ctrl = ctrl.at[8].set(-18)
ctrl = ctrl.at[10].set(-18)

# 分批仿真和渲染
frames = []
rollout = [state]
batch_size = 50  # 每批次仿真步数
total_steps = 800  # 总仿真步数

# 观察是否会将显存/内存快速消耗殆尽
# for i in range(total_steps):
#     state = jit_step(state, ctrl)
#     rollout.append(state) # 不转移到CPU

# demo渲染：先握紧，再伸出2指头
for _ in range(400):
    ctrl = ctrl.at[0].set(-18)
    ctrl = ctrl.at[5].set(-18)
    state = jit_step(state, ctrl)
    rollout.append(state)

for _ in range(400):
    ctrl = ctrl.at[5].set(0)
    ctrl = ctrl.at[7].set(0)
    state = jit_step(state, ctrl)
    rollout.append(state)

frames = env.render(rollout, height=480, width=640, camera=camera)

# 保存视频到本地
output_path = f"./video/obj_hand_rollout_{total_steps}.mp4"
media.write_video(output_path, frames, fps=30)
print(f"视频已保存到 {output_path}")
