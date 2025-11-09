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
            policy_hidden_layer_sizes=(128, 64),
            value_hidden_layer_sizes=(128, 64),
            policy_obs_key="state",
            value_obs_key="state",
            # activation=linen.relu
        ),
    )
    
    # 根据环境名称添加特定参数
    if env_name == "ParaHandReorient":
        rl_config.update(
            num_resets_per_eval=1,
            num_timesteps=100_000_000,
            num_evals=20,
            num_envs=8,#先放少一些
            batch_size=8,#先放少一些
            num_minibatches=32,
            num_updates_per_batch=5,
            learning_rate=3e-4,
            entropy_cost=1e-2,
            discounting=0.97,
            unroll_length=5,
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
    
    
    else:
        raise ValueError(f"No RL config found for environment: {env_name}")
  
    return rl_config