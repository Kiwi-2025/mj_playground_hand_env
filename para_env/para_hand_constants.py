from etils import epath

ROOT_PATH = epath.Path(__file__).parent
# PARA_HAND_XML = ROOT_PATH / "xmls" / "origin_hand.xml"
# PARA_HAND_XML = ROOT_PATH  / "xmls" / "obj_hand.xml"
# PARA_HAND_XML = ROOT_PATH / "xmls" / "obj_hand_tac.xml"

# OOM 测试时使用的 XML
# PARA_HAND_XML = ROOT_PATH / "xmls" / "mesh_hand_tac.xml"    # mesh计算 + tac
# PARA_HAND_XML = ROOT_PATH / "xmls" / "mesh_hand_no_tac.xml" # mesh计算 + no tac
# PARA_HAND_XML = ROOT_PATH / "xmls" / "obj_hand_tac.xml"     # obj 计算 + tac
# PARA_HAND_XML = ROOT_PATH / "xmls" / "obj_hand_no_tac.xml"  # obj 计算 + no tac

# sensor测试使用的 XML
PARA_HAND_XML = ROOT_PATH / "xmls" / "obj_hand_sensor.xml"

NQ_POS = 25 # 32
NQ_VEL = 25 # 31
NV = 16
NU = 16
NT_FORCE=5

TACTILE_GEOM_NAMES = [
    f"{finger}_markerG_{i}_{j}"
    for finger in ["thumb", "index", "middle", "ring", "little"]
    for i in range(9)
    for j in range(9)
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


JOINT_NAMES = [
    # index
    "if_mcp",
    "if_rot",
    "if_pip",
    "if_dip",
    # middle
    "mf_mcp",
    "mf_rot",
    "mf_pip",
    "mf_dip",
    # ring
    "rf_mcp",
    "rf_rot",
    "rf_pip",
    "rf_dip",
    # thumb
    "th_cmc",
    "th_axl",
    "th_mcp",
    "th_ipl",
]

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