from etils import epath

ROOT_PATH = epath.Path(__file__).parent
PARA_HAND_XML = ROOT_PATH / "xmls" / "sphere_tac_5x5.xml"   # 球状触觉传感器，5x5阵列

# 任务对应的 XML 文件路径
TASK_XML_FILES = {
    "reorient": ROOT_PATH / "xmls" / "reorient" / "reorient_hand.xml",
    # "rotateZ": ROOT_PATH / "xmls" / "rotateZ" / "rotateZ_hand_tac.xml",
    "rotateZ": ROOT_PATH / "xmls" / "rotateZ" / "rotateZ_hand_sim_pro.xml",
    # "rotateZ": ROOT_PATH / "xmls" / "rotateZ" / "rotateZ_hand_obj_notendon.xml",
    "grasp": ROOT_PATH / "xmls" / "grasp" / "grasp_hand.xml",
}

# TODO: 确认自由度数量
# NQ_POS = 25     # Number of Positions, 机器人系统的位置自由度数量
# NQ_VEL = 25     # Number of Velocities, 机器人系统的速度自由度数量
# NV = 25         # Number of Velocities, 机器人系统的速度自由度数量，表示机器人有16个关节
# NU = 18         # Number of Actuators, 机器人系统的执行器数量，表示机器人有16个执行器，在这里是说有16个motors
# NT_FORCE=5      # Number of Tactile Forces, 触觉传感器的数量，这里表示有5个触觉传感器，分别对应5根手指

# 训练用魔法手的关节、执行器、触觉传感器名称列表
NQ_POS = 20
NQ_VEL = 19
NV = 19
NU = 19
NT_FORCE = 5

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

# change these joint names to fit xml file， only load hand related joints
JOINT_NAMES = [
    # thumb
    "thumb_joint_0",
    "thumb_joint_1",
    "thumb_joint_2",
    "thumb_joint_3",
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
    # "palm_rotate_x",
    # "palm_rotate_y",
    # "palm_rotate_z",
    # "palm_slide_x",
    # "palm_slide_y",
    # "palm_slide_z",
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