import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math
import os
from datetime import datetime

# 加载数据
index_pos = np.load("./data/index_finger_pos.npy")  # 形状为 (100, 12)
index_quat = np.load("./data/index_finger_quats.npy")  # 形状为 (100, 16)
index_tendon_lengths = np.load("./data/index_tendon_lengths.npy")  # 形状为 (100,)

# 提取每个 link 的四元数
link_0_quat = index_quat[:, 0:4]  # 第 1 个 link 的四元数
link_1_quat = index_quat[:, 4:8]  # 第 2 个 link 的四元数
link_2_quat = index_quat[:, 8:12]  # 第 3 个 link 的四元数
link_3_quat = index_quat[:, 12:16]  # 第 4 个 link 的四元数


def quats_wxyz_to_rpy(quats_wxyz):
    """
    将形如 (N,4) 或 (4,) 的四元数 (w,x,y,z) 转为 scipy Rotation 所需的 (x,y,z,w)，
    并返回 Euler angles (roll, pitch, yaw) 以 'xyz' 顺序（固定世界坐标系，degrees=True）。
    """
    q = np.asarray(quats_wxyz)
    single = (q.ndim == 1)
    if single:
        q = q.reshape(1,4)
    # reorder to (x,y,z,w) for scipy
    q_xyzw = np.column_stack([q[:,1], q[:,2], q[:,3], q[:,0]])
    r = R.from_quat(q_xyzw)
    eulers = r.as_euler('xyz', degrees=True)  # shape (N,3)
    return eulers[0] if single else eulers

# 计算每个 link 在 MuJoCo 固定坐标系下的 roll,pitch,yaw
rpy_link_0 = quats_wxyz_to_rpy(link_0_quat)  # (T,3)
rpy_link_1 = quats_wxyz_to_rpy(link_1_quat)
rpy_link_2 = quats_wxyz_to_rpy(link_2_quat)
rpy_link_3 = quats_wxyz_to_rpy(link_3_quat)

# 提取 yaw 并做 unwrap（以获得连续曲线）
pitch_link_0 = np.degrees(np.unwrap(np.radians(rpy_link_0[:,1])))
pitch_link_1 = np.degrees(np.unwrap(np.radians(rpy_link_1[:,1])))
pitch_link_2 = np.degrees(np.unwrap(np.radians(rpy_link_2[:,1])))
pitch_link_3 = np.degrees(np.unwrap(np.radians(rpy_link_3[:,1])))

pitch_link_0 = pitch_link_0 - pitch_link_0[0]
pitch_link_1 = pitch_link_1 - pitch_link_1[0]
pitch_link_2 = pitch_link_2 - pitch_link_2[0]
pitch_link_3 = pitch_link_3 - pitch_link_3[0]

# 绘制偏航角与绳索长度的关系图（所有 link 在一张图上）
plt.figure(figsize=(10, 6))

ratio = 1
motor_angle = -ratio * index_tendon_lengths  # 取负值以符合实际物理意义，因为绳索越拉越短
motor_angle = motor_angle - motor_angle[0]  # 以初始位置为零点
plt.plot(motor_angle, pitch_link_0, label="Link 0", color="orange")
plt.plot(motor_angle, pitch_link_1, label="Link 1", color="green")
plt.plot(motor_angle, pitch_link_2, label="Link 2", color="blue")
plt.plot(motor_angle, pitch_link_3, label="Link 3", color="purple")

# 设置标题和标签
plt.title("Pitch vs Tendon Length (Excluding Base Link)")
plt.xlabel("Tendon Length")
plt.ylabel("Pitch (degrees)")

# 显示图例
plt.legend()

# 显示图像
plt.grid()
plt.tight_layout()
# 添加一个时间戳防止覆盖
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
fname = f"./figs/pitch_vs_tendon_length{ts}.png"
plt.savefig(fname, dpi=300)
print(f"Saved figure to {fname}")
plt.show()