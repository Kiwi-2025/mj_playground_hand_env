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
from para_env.grasp_env import ParaHandGrasp
from para_env.rotateZ_env import ParaHandRotateZ
from mujoco_playground import registry

# env = ParaHandReorient()
env = ParaHandGrasp()
# env = ParaHandRotateZ()
# env = TestTask()

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# 设置摄像机参数
camera = mj.MjvCamera()
# camera.type = mj.mjtCamera.mjCAMERA_TRACKING
# camera.trackbodyid = env.mj_model.body("palm").id  # 跟踪手掌
camera.distance = 1.8           # 缩短摄像机距离
camera.azimuth = 120           # 调整方位角以获得更好的视角
camera.elevation = -30          # 从略微向下的角度观察
camera.lookat[2] = 0.1          # 调整观察点高度

state = jit_reset(jax.random.PRNGKey(0))
# ctrl = jp.zeros(env.action_size)
ctrl = env._default_ctrl.copy()
action = jp.zeros(env.action_size)
action = action.at[12:16].set(-1.0)
motor_targets = env._default_ctrl + action * env._config.action_scale
motor_targets = jp.clip(motor_targets, env._act_lowers, env._act_uppers)
ctrl = motor_targets

# 分批仿真和渲染
frames = []
rollout = [state]
total_steps = 100  # 总仿真步数

# ctrl = ctrl.at[5].set(-15) 
for i in range(total_steps):
     state = jit_step(state, ctrl)
     rollout.append(state)

# 保存四元数数据到本地文件
# jp.save("./middle_finger_quats.npy", jp.stack(middle_quats))
frames = env.render(rollout, height=480, width=640, camera=camera)

output_path = f"./video/min_exam.mp4"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
media.write_video(output_path, frames, fps=30)
print(f"视频已保存到 {output_path}")
