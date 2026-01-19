
import os
import jax
import jax.numpy as jp
import numpy as np
import para_env 

def main():
    # 1. 设置环境名称和配置
    env_name = "ParaHandGrasp"
    print(f"Initializing environment: {env_name}")
    
    # 获取默认配置并设置 impl
    env_cfg = para_env.get_default_config(env_name)
    env_cfg.impl = "jax" # ConfigDict allow dot access usually, if not use string key

    # 加载环境
    env = para_env.load(env_name, config=env_cfg)

    # 2. 准备 JAX 随机 
    rng = jax.random.PRNGKey(0)
    reset_rng, step_rng = jax.random.split(rng)

    # 3. JIT 编译核心函数
    # 只需要编译 reset 和 step，以及用于提取数据的 probe 函数
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # 定义一个探测函数，从 state.data 中提取感兴趣的物理量
    # 注意：这些计算会在 Device (GPU) 上执行
    def probe_fn(state):
        data = state.data
        return {
            "cube_pos": env.get_cube_position(data),
            "palm_pos": env.get_palm_position(data),
            "cube_linvel": env.get_cube_linvel(data),
            "cube_angvel": env.get_cube_angvel(data),
            "fingertip_errors": env.get_fingertip_errors(data, env.get_cube_position(data)),
            "contact_forces": env.get_touch_forces(data),
            "pos_error": env.get_palm_position(data) - env.get_cube_position(data)
        }
    
    jit_probe = jax.jit(probe_fn)

    # 4. 执行 Reset
    print("\n=== Resetting Environment ===")
    state = jit_reset(reset_rng)
    
    # 5. 打印 Step 0 的数据
    metrics = jit_probe(state)
    print_metrics(0, metrics)

    # 6. 运行几个 Step 并观察变化
    print("\n=== Running Simulation (5 steps) ===")
    print("Action: Zero (Hold pose)")
    
    # 创建一个全零动作（维持默认姿态）
    # action_size = env.action_size # MJX env 可能没有直接暴露 action_size 属性，通常在 default_config 或 info 里
    # 我们可以查看 env.mj_model.nu (control inputs)
    # 但为了简单，我们直接用 zeros，长度通常是 env.mj_model.nu
    action = jp.zeros(env.mj_model.nu)

    for i in range(1, 6):
        state = jit_step(state, action)
        metrics = jit_probe(state)
        print_metrics(i, metrics)

def print_metrics(step, metrics):
    """辅助函数：格式化打印 DeviceArrray"""
    print(f"\n--- Step {step} ---")
    
    # 将 JAX 数组转换为 Numpy 数组以便格式化，并保留3位小数
    def fmt(arr):
        return np.array2string(np.array(arr), precision=4, suppress_small=True, floatmode='fixed')

    print(f"Cube Pos:       {fmt(metrics['cube_pos'])}")
    print(f"Palm Pos:       {fmt(metrics['palm_pos'])}")
    print(f"Pos Error:      {fmt(metrics['pos_error'])}  (Palm - Cube)")
    print(f"Cube LinVel:    {fmt(metrics['cube_linvel'])}")
    # print(f"Cube AngVel:    {fmt(metrics['cube_angvel'])}")
    # print(f"Fingertip Errs: {fmt(metrics['fingertip_errors'])}")
    # print(f"Contact Forces: {fmt(metrics['contact_forces'])}")
    
    # 如果你也想看 Observation 的原始值
    # print(f"Raw Obs (slice): {fmt(metrics['state'].obs['state'][:10])}...")

if __name__ == "__main__":
    # 限制只使用第一个 GPU，避免多卡并行问题
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
