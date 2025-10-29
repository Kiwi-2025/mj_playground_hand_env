import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# 加载数据
index_pos = np.load("./data/index_finger_pos.npy")  # 形状为 (100, 12)
index_quat = np.load("./data/index_finger_quats.npy")  # 形状为 (100, 16)
index_tendon_lengths = np.load("./data/index_tendon_lengths.npy")  # 形状为 (100,)

# 提取每个 link 的四元数
link_1_quat = index_quat[:, 4:8]  # 第 2 个 link 的四元数
link_2_quat = index_quat[:, 8:12]  # 第 3 个 link 的四元数
link_3_quat = index_quat[:, 12:16]  # 第 4 个 link 的四元数

# 计算偏航角（yaw）
def compute_yaw(quat, reference_quat):
    """
    计算相对第一步的偏航角（yaw）。
    quat: 当前步骤的四元数 (w, x, y, z)
    reference_quat: 第一步的四元数 (w, x, y, z)
    """
    # 转换为旋转矩阵
    r_current = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # (x, y, z, w)
    r_reference = R.from_quat([reference_quat[1], reference_quat[2], reference_quat[3], reference_quat[0]])
    
    # 计算相对旋转矩阵
    r_relative = r_reference.inv() * r_current
    
    # 提取偏航角（yaw）
    _, _, yaw = r_relative.as_euler('xyz', degrees=True)  # 返回 (roll, pitch, yaw)
    return yaw

# 计算每个 link 的偏航角
yaw_link_1 = [-compute_yaw(link_1_quat[i], link_1_quat[0]) for i in range(len(link_1_quat))]
yaw_link_2 = [-compute_yaw(link_2_quat[i], link_2_quat[0]) for i in range(len(link_2_quat))]
yaw_link_3 = [-compute_yaw(link_3_quat[i], link_3_quat[0]) for i in range(len(link_3_quat))]

# 绘制偏航角与绳索长度的关系图（所有 link 在一张图上）
plt.figure(figsize=(10, 6))

ratio = 1
motor_angle = -ratio * index_tendon_lengths  # 取负值以符合实际物理意义，因为绳索越拉越短
motor_angle = motor_angle - motor_angle[0]  # 以初始位置为零点
plt.plot(motor_angle, yaw_link_1, label="Link 1", color="green")
plt.plot(motor_angle, yaw_link_2, label="Link 2", color="blue")
plt.plot(motor_angle, yaw_link_3, label="Link 3", color="purple")

# 设置标题和标签
plt.title("Yaw vs Tendon Length (Excluding Base Link)")
plt.xlabel("Tendon Length")
plt.ylabel("Yaw (degrees)")

# 显示图例
plt.legend()

# 显示图像
plt.grid()
plt.tight_layout()
plt.show()