import mujoco as mj
import mediapy as media
import numpy as np

# 加载模型
with open("./xmls/hand.xml", "r") as f:
    xml_str = f.read()
model = mj.MjModel.from_xml_string(xml_str)
data = mj.MjData(model)

# 创建渲染器
height, width = 720, 1280  # 设置分辨率
renderer = mj.Renderer(model, height=height, width=width)

# 设置摄像机参数
camera = mj.MjvCamera()
camera.type = mj.mjtCamera.mjCAMERA_FREE
camera.lookat[:] = [0, 0, 0.2]  # 对准手部模型的中心位置
camera.distance = 0.3           # 缩短摄像机距离
camera.azimuth = 90             # 从侧面观察
camera.elevation = -35          # 从略微向下的角度观察

# 渲染仿真过程
frames = []
for i in range(100):  # 仿真 100 步
    mj.mj_step(model, data)  # 执行仿真步
    renderer.update_scene(data, camera=camera)  # 更新场景
    frame = renderer.render()  # 渲染当前帧
    frames.append(frame)

# 保存视频到本地
output_path = "mujoco_simulation_output.mp4"
media.write_video(output_path, frames, fps=30)  # 设置帧率为 30 FPS
print(f"视频已保存到 {output_path}")

# 关闭渲染器
renderer.close()