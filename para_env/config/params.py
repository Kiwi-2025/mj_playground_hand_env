"""RL config for ParaHand Manipulation envs."""
from typing import Optional
from ml_collections import config_dict

def brax_ppo_config(
    env_name: str, impl: Optional[str] = None
) -> config_dict.ConfigDict:
    """Returns tuned Brax PPO config for the given environment."""
    # 导入 get_default_config 函数
    from para_env import get_default_config
    
    env_config = get_default_config(env_name)
    
    # 基础配置
    rl_config = config_dict.create(
        episode_length=env_config.episode_length,
        normalize_observations=True,
        action_repeat=env_config.action_repeat,
        reward_scaling=1.0,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(32, 32, 32, 32),
            value_hidden_layer_sizes=(256, 256, 256, 256, 256),
            policy_obs_key="state",
            value_obs_key="state",
        ),
        num_resets_per_eval=10,
    )
    
    # 根据环境名称添加特定参数
    # BRAX要求 batch_size == num_envs * unroll_length，在设计这些参数时候最好确保所有的参数之间都是可以整除的
    # 重新定位任务，将方块重新旋转定位到目标位置
    if env_name == "ParaHandReorient":
        rl_config.update(
            num_resets_per_eval=1,
            num_timesteps=20_000,
            num_evals=10,
            num_envs=8,
            unroll_length=32,
            batch_size=256,
            num_minibatches=4,
            num_updates_per_batch=2,
            learning_rate=3e-4,
            entropy_cost=1e-2,
            discounting=0.99,
            max_devices_per_host=8,
        )

    elif env_name == "ParaHandGrasp":
        rl_config.num_timesteps = 100_000_000
        rl_config.num_evals = 10
        rl_config.num_minibatches = 32
        rl_config.unroll_length = 40
        rl_config.num_updates_per_batch = 4
        rl_config.discounting = 0.97
        rl_config.learning_rate = 3e-4
        rl_config.entropy_cost = 1e-2
        rl_config.num_envs = 8192
        rl_config.batch_size = 256
        rl_config.num_resets_per_eval = 1
        rl_config.network_factory = config_dict.create(
            # policy_hidden_layer_sizes=(512, 256, 128),
            # value_hidden_layer_sizes=(512, 256, 128),
            policy_hidden_layer_sizes=(256, 128),
            value_hidden_layer_sizes=(256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        )
        
    elif env_name == "ParaHandRotateZ":
        rl_config.num_timesteps = 100_000_000
        rl_config.num_evals = 10
        rl_config.num_minibatches = 32
        rl_config.unroll_length = 40
        rl_config.num_updates_per_batch = 4
        rl_config.discounting = 0.97
        rl_config.learning_rate = 3e-4
        rl_config.entropy_cost = 1e-2
        rl_config.num_envs = 8192
        rl_config.batch_size = 256
        rl_config.num_resets_per_eval = 1
        rl_config.network_factory = config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        )
    
    else:
        raise ValueError(f"No RL config found for environment: {env_name}")
    return rl_config