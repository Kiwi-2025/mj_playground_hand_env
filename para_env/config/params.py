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
    )
    
    # 根据环境名称添加特定参数
    if env_name == "ParaHandReorient":
        rl_config.update(
            num_resets_per_eval=10,
            num_timesteps=10_000_000,
            num_evals=10,
            num_envs=2048,
            batch_size=1024,
            num_minibatches=32,
            num_updates_per_batch=4,
            learning_rate=3e-4,
            entropy_cost=1e-2,
            discounting=0.97,
            unroll_length=10,
            max_devices_per_host=8,
        )
    # 可以根据需要添加更多环境的配置
    # elif env_name == "AnotherParaHandEnv":
    #     rl_config.update(
    #         num_resets_per_eval=5,
    #         num_timesteps=5_000_000,
    #         num_evals=5,
    #         num_envs=1024,
    #         batch_size=512,
    #         num_minibatches=16,
    #         num_updates_per_batch=2,
    #         learning_rate=1e-4,
    #         entropy_cost=5e-3,
    #         discounting=0.95,
    #         unroll_length=5,
    #         max_devices_per_host=4,
    #     )
  
    return rl_config