import numpy as np
import matplotlib.pyplot as plt

index_pos = np.load("./data/index_finger_pos.npy")  # 形状为 (100, 12)
index_quat = np.load("./data/index_finger_quats.npy")  # 形状为 (100, 16)
index_tendon_lengths = np.load("./data/index_tendon_lengths.npy")  # 形状为 (100,)

# 提取每个 link 的 x 和 z 坐标（忽略 y 坐标）
link_0 = index_pos[:, [0, 2]]  # 第 1 个 link 的 (x, z)
link_1 = index_pos[:, [3, 5]]  # 第 2 个 link 的 (x, z)
link_2 = index_pos[:, [6, 8]]  # 第 3 个 link 的 (x, z)
link_3 = index_pos[:, [9, 11]]  # 第 4 个 link 的 (x, z)

# 提取所有 y 坐标
y_coords = index_pos[:, [1, 4, 7, 10]].flatten()  # 所有 link 的 y 坐标

# 计算 y 坐标的极差和方差
y_range = np.ptp(y_coords)  # 极差
y_variance = np.var(y_coords)  # 方差

print(f"Y Coordinate Range (极差): {y_range:.4f}")
print(f"Y Coordinate Variance (方差): {y_variance:.4f}")

# 创建二维图像
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制每个 link 的轨迹
ax.plot(link_0[:, 0], link_0[:, 1], label="Base_Link", color="red", linestyle="-")
ax.plot(link_1[:, 0], link_1[:, 1], label="Link_1", color="green", linestyle="-")
ax.plot(link_2[:, 0], link_2[:, 1], label="Link_2", color="blue", linestyle="-")
ax.plot(link_3[:, 0], link_3[:, 1], label="Link_3", color="purple", linestyle="-")

# 每隔 20 步连接对应的点形成骨架
for i in range(0, len(link_0), 20):
    # 连接 link_0 -> link_1 -> link_2 -> link_3
    ax.plot(
        [link_0[i, 0], link_1[i, 0], link_2[i, 0], link_3[i, 0]],
        [link_0[i, 1], link_1[i, 1], link_2[i, 1], link_3[i, 1]],
        color="black",
        marker="o",
        label="Skeleton" if i == 0 else None  # 避免重复图例
    )

# 设置坐标轴标签
ax.set_xlabel("X")
ax.set_ylabel("Z")

# 设置标题
ax.set_title("Middle Finger Links Trajectory in XZ Plane with Skeleton")

# 显示图例
ax.legend()

# 显示图像
plt.show()