import jax
import jax.numpy as jp
import numpy as np
from para_env.reorient_env import ParaHandReorient

def check_array(name, arr, threshold=1e4):
    """检查数组中是否有 NaN, Inf 或异常大的值"""
    arr_np = np.array(arr) # 转为 numpy 方便打印
    if np.any(np.isnan(arr_np)):
        print(f"❌ [NaN DETECTED] {name} contains NaN!")
        print(f"   Values: {arr_np}")
        return True
    if np.any(np.isinf(arr_np)):
        print(f"❌ [Inf DETECTED] {name} contains Inf!")
        return True
    if np.any(np.abs(arr_np) > threshold):
        print(f"⚠️ [HUGE VALUE] {name} has values > {threshold}!")
        print(f"   Max value: {np.max(np.abs(arr_np))}")
        return True
    return False

def main():
    print("🔍 开始环境数值诊断...")
    env = ParaHandReorient()
    
    # 使用 JIT 编译 reset 和 step，模拟真实训练环境
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    rng = jax.random.PRNGKey(0)
    
    # 运行多次测试
    for i in range(20):
        print(f"--- Test Episode {i} ---")
        rng, key = jax.random.split(rng)
        
        # 1. 检查 Reset 后的状态
        try:
            state = jit_reset(key)
            # 强制同步以捕获错误
            state.data.qpos.block_until_ready()
        except Exception as e:
            print(f"💥 Crash during reset: {e}")
            break

        # 分解 qpos 查看具体是哪部分炸了
        # 假设前 N 个是手，后 7 个是方块 (3 pos + 4 quat)
        hand_qpos = state.data.qpos[:-7]
        cube_pos = state.data.qpos[-7:-4]
        cube_quat = state.data.qpos[-4:]
        
        hand_qvel = state.data.qvel[:-6]
        cube_vel = state.data.qvel[-6:]

        if check_array("Reset: Hand Qpos", hand_qpos): break
        if check_array("Reset: Cube Pos", cube_pos): break
        if check_array("Reset: Cube Quat", cube_quat): break
        if check_array("Reset: Hand Qvel", hand_qvel): break
        if check_array("Reset: Cube Vel", cube_vel): break
        
        # 2. 检查 Step 后的状态 (物理仿真第一步最容易炸)
        action = jp.zeros(env.action_size) # 零动作测试
        try:
            state = jit_step(state, action)
            state.data.qpos.block_until_ready()
        except Exception as e:
            print(f"💥 Crash during step: {e}")
            break
            
        if check_array("Step 1: Qpos", state.data.qpos): 
            print("   -> 物理仿真在第一步后发散，通常是发生了剧烈碰撞（穿模）。")
            break
        if check_array("Step 1: Qvel", state.data.qvel): 
            print("   -> 速度爆炸，检查 sim_dt 或初始接触力。")
            break
            
    print("✅ 诊断结束")

if __name__ == "__main__":
    # 开启 NaN 调试模式（会变慢，但报错更准）
    jax.config.update("jax_debug_nans", True)
    main()