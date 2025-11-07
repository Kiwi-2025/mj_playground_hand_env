from etils import epath

ROOT_PATH = epath.Path(__file__).parent
PARA_HAND_XML = ROOT_PATH / "xmls" / "sphere_tac_5x5.xml"   # 球状触觉传感器，5x5阵列

# 任务对应的 XML 文件路径
TASK_XML_FILES = {
    "reorient": ROOT_PATH / "xmls" / "reorient" / "reorient_hand.xml",
}

# TODO: 确认自由度数量
NQ = 32  # 5fingers × 4DOF + 6DOF(base) = 26 DOF (32 values due to quaternion)

# 位置和速度自由度
NQ_POS = 32  # 修改为32: 完整的位置状态向量长度 
NQ_VEL = 31  # 修改为31: 速度状态向量长度(quaternion角速度用3个值)

# 实际控制自由度
NV = 26  # 修改为26: 实际物理自由度数量
NU = 20  # 修改为20: 可控制的关节数量(只有手指关节)
NT_FORCE=5

TACTILE_GEOM_NAMES = [
    f"{finger}_markerG_{i}_{j}"
    for finger in ["thumb", "index", "middle", "ring", "little"]
    # for i in range(9) for j in range(9)   # 9x9阵列
    for i in range(5) for j in range(5)     # 5x5阵列
]

TACTILE_BODY_NAMES = [
    f"{finger}_markerB_{i}_{j}"
    for finger in ["thumb", "index", "middle", "ring", "little"]
    # for i in range(9) for j in range(9)   # 9x9阵列
    for i in range(5) for j in range(5)     # 5x5阵列
]

INNER_SITE_NAMES = [
    f"inner_{finger}_{row}_{col}"
    for finger in ["thumb", "index", "middle", "ring","little"]
    for row in range(1, 4)
    for col in range(5)
]+[
    f"inner_palm_{i}"
    for i in range(1,5)
]

OUTER_SITE_NAMES = [
    f"outer_{finger}_{row}_{col}"
    for finger in ["thumb", "index", "middle", "ring","little"]
    for row in range(1, 4)
    for col in range(5)
]+[
    f"outer_palm_{i}"
    for i in range(1,5)
]

# change these joint names to fit xml file
JOINT_NAMES = [
    # thumb
    "thumb_joint_1",
    "thumb_joint_2",
    "thumb_universal_1",
    "thumb_universal_2",
    # index
    "index_joint_0",
    "index_joint_1",
    "index_joint_2",
    "index_swing",
    # middle
    "middle_joint_0",
    "middle_joint_1",
    "middle_joint_2",
    # ring
    "ring_joint_0",
    "ring_joint_1",
    "ring_joint_2",
    "ring_swing",
    # little
    "little_joint_0",
    "little_joint_1",
    "little_joint_2",
    "little_swing",
    # palm
    "palm_rotate_x",
    "palm_rotate_y",
    "palm_rotate_z",
    "palm_slide_x",
    "palm_slide_y",
    "palm_slide_z",
    
    # others
    "cube_freejoint",
]
    
# change these actuator names to fit xml file
ACTUATOR_NAMES = [
    # index
    "if_mcp_act",
    "if_rot_act",
    "if_pip_act",
    "if_dip_act",
    # middle
    "mf_mcp_act",
    "mf_rot_act",
    "mf_pip_act",
    "mf_dip_act",
    # ring
    "rf_mcp_act",
    "rf_rot_act",
    "rf_pip_act",
    "rf_dip_act",
    # thumb
    "th_cmc_act",
    "th_axl_act",
    "th_mcp_act",
    "th_ipl_act",
]

FINGERTIP_NAMES = [
    "th_tip",
    "if_tip",
    "mf_tip",
    "rf_tip",
]