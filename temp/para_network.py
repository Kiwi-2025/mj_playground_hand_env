import dataclasses
import functools
from typing import Any, Callable, Literal, Mapping, Sequence, Tuple
from flax import linen
import jax
import jax.numpy as jnp

from brax.training import types
from brax.training.acme import running_statistics
from brax.training import distribution
from brax.training import networks
from brax.training.agents.ppo.networks import PPONetworks, MLP, CNN

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]

class MultipartHybridNetwork(linen.Module):
    mlp_layers: Sequence[int]
    output_size: int
    cnn_features: Sequence[int]
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    use_batch_norm: bool = False
    normalize_inputs: bool = False
    
    @linen.compact
    def __call__(self, x: Mapping[str, jnp.ndarray], training: bool = False) -> jnp.ndarray:
        cnn_outputs = []
        for key, value in x.items():
            if len(value.shape) > 2:  # Assuming images have more than 2 dimensions
                cnn_module = CNN(
                    features=self.cnn_features,
                    activation=self.activation,
                    kernel_init=self.kernel_init,
                    use_batch_norm=self.use_batch_norm,
                    normalize_inputs=self.normalize_inputs,
                )
                cnn_output = cnn_module(value, training=training)
                cnn_outputs.append(cnn_output)
            else:
                cnn_outputs.append(value)
            
        concatenated = jnp.concatenate(cnn_outputs, axis=-1)
        
        mlp_module = MLP(
            layer_sizes=list(self.mlp_layers) + [self.output_size],
            activation=self.activation,
            kernel_init=self.kernel_init,
        )
        return mlp_module(concatenated)

@dataclasses.dataclass
class FeedForwardNetwork:
  init: Callable[..., Any]
  apply: Callable[..., Any]

def normalizer_select(
    processor_params: running_statistics.RunningStatisticsState, obs_key: str
) -> running_statistics.RunningStatisticsState:
  return running_statistics.RunningStatisticsState(
      count=processor_params.count,
      mean=processor_params.mean[obs_key],
      summed_variance=processor_params.summed_variance[obs_key],
      std=processor_params.std[obs_key],
  )

def _get_obs_state_size(obs_size: types.ObservationSize, obs_key: str) -> int:
  obs_size = obs_size[obs_key] if isinstance(obs_size, Mapping) else obs_size
  return jax.tree_util.tree_flatten(obs_size)[0][-1]

def make_policy_network(
    param_size: int,
    obs_size: types.ObservationSize,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    layer_norm: bool = False,
    obs_key: str = 'state',
    distribution_type: Literal['normal', 'tanh_normal'] = 'tanh_normal',
    noise_std_type: Literal['scalar', 'log'] = 'scalar',
    init_noise_std: float = 1.0,
    state_dependent_std: bool = False,
    use_cnn: bool = True,  # 新增参数：是否使用CNN
    cnn_features: Sequence[int] = (16, 32, 32, 8),  # CNN配置
) -> FeedForwardNetwork:
  """Creates a policy network."""
  
  # 如果观测是字典且包含多个手指输入，使用混合网络
  if use_cnn and isinstance(obs_size, Mapping):
    finger_names = ["thumb", "index", "middle", "ring", "little"]
    has_finger_obs = any(finger in obs_size for finger in finger_names)
    
    if has_finger_obs:
      # 使用已定义的MultipartHybridNetwork
      policy_module = MultipartHybridNetwork(
          mlp_layers=list(hidden_layer_sizes),
          output_size=param_size,
          cnn_features=cnn_features,
          activation=activation,
          kernel_init=kernel_init,
          use_batch_norm=True,
          normalize_inputs=False,
      )
      
      def apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs, training=False)
      
      # 创建虚拟输入用于初始化
      dummy_obs = {}
      for finger in finger_names:
        if finger in obs_size:
          dummy_obs[finger] = jnp.zeros((1,) + tuple(obs_size[finger]))
      if 'state' in obs_size:
        dummy_obs['state'] = jnp.zeros((1, obs_size['state']))
      
      def init(key):
        return policy_module.init(key, dummy_obs, training=False)
      
      return FeedForwardNetwork(init=init, apply=apply)
  
  # 否则使用标准MLP
  if distribution_type == 'tanh_normal':
    policy_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm,
    )
  elif distribution_type == 'normal':
    policy_module = networks.PolicyModuleWithStd(
        param_size=param_size,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm,
        noise_std_type=noise_std_type,
        init_noise_std=init_noise_std,
        state_dependent_std=state_dependent_std,
    )
  else:
    raise ValueError(
        f'Unsupported distribution type: {distribution_type}. Must be one'
        ' of "normal" or "tanh_normal".'
    )

  def apply(processor_params, policy_params, obs):
    if isinstance(obs, Mapping):
      obs = preprocess_observations_fn(
          obs[obs_key], normalizer_select(processor_params, obs_key)
      )
    else:
      obs = preprocess_observations_fn(obs, processor_params)
    return policy_module.apply(policy_params, obs)

  obs_size = _get_obs_state_size(obs_size, obs_key)
  dummy_obs = jnp.zeros((1, obs_size))

  def init(key):
    policy_module_params = policy_module.init(key, dummy_obs)
    return policy_module_params

  return FeedForwardNetwork(init=init, apply=apply)


def make_value_network(
    obs_size: types.ObservationSize,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    obs_key: str = 'state',
    use_cnn: bool = True,  # 新增参数：是否使用CNN
    cnn_features: Sequence[int] = (16, 32, 32, 8),  # CNN配置
) -> FeedForwardNetwork:
  """Creates a value network."""
  
  # 如果观测是字典且包含多个手指输入，使用混合网络
  if use_cnn and isinstance(obs_size, Mapping):
    finger_names = ["thumb", "index", "middle", "ring", "little"]
    has_finger_obs = any(finger in obs_size for finger in finger_names)
    
    if has_finger_obs:
      # 使用已定义的MultipartHybridNetwork
      value_module = MultipartHybridNetwork(
          mlp_layers=list(hidden_layer_sizes),
          output_size=1,  # 值函数输出单个标量
          cnn_features=cnn_features,
          activation=activation,
          use_batch_norm=True,
          normalize_inputs=False,
      )
      
      def apply(processor_params, value_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        output = value_module.apply(value_params, obs, training=False)
        return jnp.squeeze(output, axis=-1)
      
      # 创建虚拟输入用于初始化
      dummy_obs = {}
      for finger in finger_names:
        if finger in obs_size:
          dummy_obs[finger] = jnp.zeros((1,) + tuple(obs_size[finger]))
      if 'state' in obs_size:
        dummy_obs['state'] = jnp.zeros((1, obs_size['state']))
      
      return FeedForwardNetwork(
          init=lambda key: value_module.init(key, dummy_obs, training=False),
          apply=apply
      )
  
  # 否则使用标准MLP
  value_module = MLP(
      layer_sizes=list(hidden_layer_sizes) + [1],
      activation=activation,
  )

  def apply(processor_params, value_params, obs):
    if isinstance(obs, Mapping):
      obs = preprocess_observations_fn(
          obs[obs_key], normalizer_select(processor_params, obs_key)
      )
    else:
      obs = preprocess_observations_fn(obs, processor_params)
    return jnp.squeeze(value_module.apply(value_params, obs), axis=-1)

  obs_size_value = _get_obs_state_size(obs_size, obs_key)
  dummy_obs = jnp.zeros((1, obs_size_value))
  
  return FeedForwardNetwork(
      init=lambda key: value_module.init(key, dummy_obs), 
      apply=apply
  )

def make_ppo_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: networks.ActivationFn = linen.swish,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'state',
    distribution_type: Literal['normal', 'tanh_normal'] = 'tanh_normal',
    noise_std_type: Literal['scalar', 'log'] = 'scalar',
    init_noise_std: float = 1.0,
    state_dependent_std: bool = False,
) -> PPONetworks:
  """Make PPO networks with preprocessor."""
  parametric_action_distribution: distribution.ParametricDistribution
  if distribution_type == 'normal':
    parametric_action_distribution = distribution.NormalDistribution(
        event_size=action_size
    )
  elif distribution_type == 'tanh_normal':
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
  else:
    raise ValueError(
        f'Unsupported distribution type: {distribution_type}. Must be one'
        ' of "normal" or "tanh_normal".'
    )
  policy_network = make_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation,
      obs_key=policy_obs_key,
      distribution_type=distribution_type,
      noise_std_type=noise_std_type,
      init_noise_std=init_noise_std,
      state_dependent_std=state_dependent_std,
  )
  value_network = make_value_network(
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=value_hidden_layer_sizes,
      activation=activation,
      obs_key=value_obs_key,
  )

  return PPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      parametric_action_distribution=parametric_action_distribution,
  )
