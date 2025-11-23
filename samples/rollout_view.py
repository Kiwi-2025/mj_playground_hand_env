import os, sys, time
import gc

from para_env.para_hand_env import TestTask
from mujoco_playground import registry

# 禁止 JAX 预分配全部显存，限制显存占比，避免 OOM
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
# 强制使用 GPU 进行推理
os.environ["JAX_PLATFORM_NAME"] = "gpu"

gc.collect()
import jax
import jax.numpy as jp
import mujoco as mj
from mujoco import mjx
import mediapy as media


env = TestTask()
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# 设置摄像机参数
camera = mj.MjvCamera()
camera.type = mj.mjtCamera.mjCAMERA_FREE
camera.lookat[:] = [0, 0, 0.2]  # 对准手部模型的中心位置
camera.distance = 0.3           # 缩短摄像机距离
camera.azimuth = 270             # 从侧面观察
camera.elevation = -20          # 从略微向下的角度观察

state = jit_reset(jax.random.PRNGKey(0))

# 初始化控制信号
ctrl = jp.zeros(18)
# ctrl = ctrl.at[0].set(-18)


# 分批仿真和渲染
frames = []
rollout = [state]
index_quats = []
index_pos = []
index_tendon_lengths = []
total_steps = 400  # 总仿真步数

ctrl = ctrl.at[5].set(-15) 
for i in range(total_steps):
     state = jit_step(state, ctrl)
     state_host = jax.device_get(state)
     # middle_quats.append(env.get_finger_quat(state.data, "middle"))
     index_pos.append(env.get_finger_pos(state_host.data, "index"))
     index_quats.append(env.get_finger_quat(state_host.data, "index"))
     index_tendon_lengths.append(env.get_tendon_length(state_host.data, "index_tendon"))
     
     frame = env.render([state_host], height=480, width=640, camera=camera)[0]
     frames.append(frame)

     if (i % 200) == 0:
          gc.collect()

# ctrl = ctrl.at[5].set(0) # 放松index tendon
# for _ in range(total_steps):
#      state = jit_step(state, ctrl)
#      index_pos.append(env.get_finger_pos(state.data, "index"))
#      index_quats.append(env.get_finger_quat(state.data, "index"))
#      index_tendon_lengths.append(env.get_tendon_length(state.data, "index_tendon"))
#      rollout.append(state)
ctrl = ctrl.at[5].set(0) # 放松index tendon
for _ in range(total_steps):
     state = jit_step(state, ctrl)
     state_host = jax.device_get(state)
     index_pos.append(env.get_finger_pos(state_host.data, "index"))
     index_quats.append(env.get_finger_quat(state_host.data, "index"))
     index_tendon_lengths.append(env.get_tendon_length(state_host.data, "index_tendon"))
     
     frame = env.render([state_host], height=480, width=640, camera=camera)[0]
     frames.append(frame)

     if (i % 200) == 0:
          gc.collect()

# 保存四元数数据到本地文件
# jp.save("./middle_finger_quats.npy", jp.stack(middle_quats))
jp.save("./data/index_finger_pos.npy", jp.stack(index_pos))
jp.save("./data/index_finger_quats.npy", jp.stack(index_quats))
jp.save("./data/index_tendon_lengths.npy", jp.stack(index_tendon_lengths))

# frames = env.render(rollout, height=480, width=640, camera=camera)
output_path = f"./video/sim2real.mp4"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
media.write_video(output_path, frames, fps=30)
print(f"视频已保存到 {output_path}")
