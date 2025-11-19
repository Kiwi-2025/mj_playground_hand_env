import os, sys, time
import gc

if "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "egl")
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


from para_env.para_hand_env import TestTask
from para_env.reorient_env import ParaHandReorient
from mujoco_playground import registry

env = ParaHandReorient()
# env = TestTask()
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
total_steps = 20  # 总仿真步数

# ctrl = ctrl.at[5].set(-15) 
for i in range(total_steps):
     state = jit_step(state, ctrl)
     rollout.append(state)

# 保存四元数数据到本地文件
# jp.save("./middle_finger_quats.npy", jp.stack(middle_quats))
frames = env.render(rollout, height=480, width=640, camera=camera)

output_path = f"./video/reorient.mp4"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
media.write_video(output_path, frames, fps=30)
print(f"视频已保存到 {output_path}")
