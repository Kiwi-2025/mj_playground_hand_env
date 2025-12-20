import mujoco
import numpy as np
from para_env import grasp_env
import sys
import os

def test_sensors():
    print("Initializing Environment...")
    try:
        # 初始化环境以获取模型
        # 注意：这里我们只用到 mj_model，不需要 jax 相关的初始化
        env = grasp_env.ParaHandGrasp()
        model = env.mj_model
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Failed to initialize environment: {e}")
        return

    print("="*60)
    print("SENSOR BINDING DIAGNOSTIC TOOL")
    print("="*60)

    # 1. 检查 Sensor 定义
    print("\n[1] Checking Sensor Definitions in XML")
    sensor_names = ["palm_pos", "cube_pos"]
    for name in sensor_names:
        try:
            sid = model.sensor(name).id
            objtype = model.sensor_objtype[sid]
            objid = model.sensor_objid[sid]
            
            # objtype: 1=body, 2=xbody, 3=geom, 4=site, 5=camera
            type_str = "Unknown"
            if objtype == 1: type_str = "Body"
            elif objtype == 4: type_str = "Site"
            
            # 获取绑定的对象名称
            if objtype == 1:
                target_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, objid)
            elif objtype == 4:
                target_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, objid)
            else:
                target_name = f"ID {objid}"

            print(f"  Sensor '{name}': ID={sid}, Type={type_str}, Attached To='{target_name}'")
            
            if name == "palm_pos" and type_str == "Site":
                # 如果绑定在 Site 上，检查 Site 的父 Body
                site_bodyid = model.site_bodyid[objid]
                site_body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, site_bodyid)
                print(f"    -> Site '{target_name}' is attached to Body '{site_body_name}'")
                
                # 检查 Site 的局部位置
                site_pos = model.site_pos[objid]
                print(f"    -> Site Local Pos (relative to body): {site_pos}")

        except Exception as e:
            print(f"  Error checking sensor '{name}': {e}")

    # 2. 动态测试：移动手掌
    print("\n[2] Dynamic Test: Moving Palm Joint")
    print("  Goal: Verify if 'palm_pos' sensor updates correctly when palm moves.")
    
    palm_joint_names = ["palm_slide_x", "palm_slide_y", "palm_slide_z"]
    try:
        palm_qpos_addrs = [model.jnt_qposadr[model.joint(name).id] for name in palm_joint_names]
    except KeyError as e:
        print(f"  Error: Could not find palm joints. {e}")
        return

    # 测试几个不同的位置偏移
    test_offsets = [
        [0.0, 0.0, 0.0],    # 默认
        [0.0, 0.0, -0.2],   # 向下
        [0.1, 0.0, 0.0],    # 向右
    ]

    palm_sensor_id = model.sensor("palm_pos").id
    palm_body_id = model.body("palm").id

    for offset in test_offsets:
        mujoco.mj_resetData(model, data)
        
        # 设置关节位置
        for i, val in enumerate(offset):
            data.qpos[palm_qpos_addrs[i]] = val
            
        mujoco.mj_forward(model, data)
        
        # 读取数据
        sensor_val = data.sensordata[model.sensor_adr[palm_sensor_id]:model.sensor_adr[palm_sensor_id]+3]
        body_pos = data.xpos[palm_body_id]
        
        print(f"\n  Set Palm Slide Offset: {offset}")
        print(f"    Body Global Pos:   {body_pos}")
        print(f"    Sensor Read (Site):{sensor_val}")
        print(f"    Difference:        {sensor_val - body_pos}")
        
        # 验证逻辑：Sensor(Site) 应该等于 BodyPos + Rotation * SiteLocalPos
        # 这里简单验证：如果 Body 动了 [0,0,-0.2]，Sensor 也应该动 [0,0,-0.2]
        if offset == [0.0, 0.0, 0.0]:
            base_sensor = sensor_val.copy()
            base_body = body_pos.copy()
        else:
            sensor_move = sensor_val - base_sensor
            body_move = body_pos - base_body
            print(f"    Movement Check: Body moved {body_move}, Sensor moved {sensor_move}")
            if np.allclose(sensor_move, body_move, atol=1e-4):
                print("    [PASS] Sensor moves consistently with Body.")
            else:
                print("    [FAIL] Sensor movement does not match Body movement!")

    # 3. 动态测试：移动方块
    print("\n[3] Dynamic Test: Moving Cube")
    cube_sensor_id = model.sensor("cube_pos").id
    cube_body_id = model.body("cube").id
    cube_joint_id = model.joint("cube_freejoint").id
    cube_qpos_addr = model.jnt_qposadr[cube_joint_id]

    test_cube_pos = [
        [0.0, 0.0, 0.05],
        [0.5, 0.5, 0.5]
    ]

    for pos in test_cube_pos:
        mujoco.mj_resetData(model, data)
        data.qpos[cube_qpos_addr:cube_qpos_addr+3] = pos
        data.qpos[cube_qpos_addr+3] = 1.0 # w
        
        mujoco.mj_forward(model, data)
        
        sensor_val = data.sensordata[model.sensor_adr[cube_sensor_id]:model.sensor_adr[cube_sensor_id]+3]
        body_pos = data.xpos[cube_body_id]
        
        print(f"\n  Set Cube Pos: {pos}")
        print(f"    Body Global Pos: {body_pos}")
        print(f"    Sensor Read:     {sensor_val}")
        
        if np.allclose(sensor_val, body_pos, atol=1e-4):
            print("    [PASS] Sensor matches Body position exactly.")
        else:
            print("    [FAIL] Sensor mismatch!")

    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_sensors()
