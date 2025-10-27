import os, sys, time
import gc
gc.collect()
import jax
import jax.numpy as jp
import mujoco as mj
from mujoco import mjx
import mediapy as media

from absl import flags, app
from mujoco_playground._src import registry
from mujoco_playground._src.locomotion.g1.joystick import Joystick, default_config

# 强制使用 GPU 进行推理
os.environ["JAX_PLATFORM_NAME"] = "gpu"

def main(argv):
    # 定义环境名称标志
    _ENV_NAME = flags.DEFINE_string(
        "env_name",
        "G1JoystickFlatTerrain",
        # "G1JoystickRoughTerrain",
        f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
    )
    # 初始化环境
    env_cfg = registry.get_default_config(_ENV_NAME.value)
    env = registry.load(
        _ENV_NAME.value, 
        config=env_cfg,
        )

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # 设置摄像机参数
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0, -1, 0.5]  # 对准手部模型的中心位置
    camera.distance = 8.0           # 缩短摄像机距离
    camera.azimuth = 90             # 从侧面观察
    camera.elevation = -20          # 从略微向下的角度观察

    state = jit_reset(jax.random.PRNGKey(0))
    # 初始化控制信号
    ctrl = jp.zeros(env.mj_model.nu)  # 根据环境的控制维度动态设置控制信号

    # 分批仿真和渲染
    frames = []
    rollout = [state]
    batch_size = 50  # 每批次仿真步数
    total_steps = 300  # 总仿真步数

    # 使用g1模型观察是否会将显存/内存快速消耗殆尽
    for i in range(total_steps):
        state = jit_step(state, ctrl)
        rollout.append(state) # 不转移到CPU
        # rollout.append(jax.device_get(state)) # 转移到CPU
    frames = env.render(rollout, height=480, width=640, camera=camera)

    # 生成带时间戳的视频文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f"./video/g1_rollout_{timestamp}.mp4"

    # 保存视频到本地
    media.write_video(output_path, frames, fps=30)
    print(f"视频已保存到 {output_path}")

if __name__ == "__main__":
    app.run(main)
