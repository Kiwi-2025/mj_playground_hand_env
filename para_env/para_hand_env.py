from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from functools import partial
from ml_collections import config_dict
import mujoco as mj
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward

from para_env import para_hand_constants as consts
from para_env.para_hand_base import ParaHandEnv

def default_config() -> config_dict.ConfigDict:
  """默认配置，参考LeapHandEnv,需要根据实际更改"""
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.002,

      action_scale=0.5,
      action_repeat=1,
      ema_alpha=1.0,
      episode_length=1000,
      success_threshold=0.1,
      history_len=1,
      obs_noise=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.05,
              cube_pos=0.02,
              cube_ori=0.1,
          ),
          random_ori_injection_prob=0.0,
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              orientation=5.0,
              position=0.5,
              termination=-100.0,
              hand_pose=-0.5,
              action_rate=-0.001,
              joint_vel=0.0,
              energy=-1e-3,
          ),
          success_reward=100.0,
      ),
      pert_config=config_dict.create(
          enable=False,
          linear_velocity_pert=[0.0, 3.0],
          angular_velocity_pert=[0.0, 0.5],
          pert_duration_steps=[1, 100],
          pert_wait_steps=[60, 150],
      ),
      impl='jax',
      nconmax=30 * 8192,
      njmax=128,
  )

class TestTask(ParaHandEnv):
  """测试任务"""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.PARA_HAND_XML.as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()
  
  def _post_init(self) -> None:

    self._lowers = self._mj_model.actuator_ctrlrange[:, 0]
    self._uppers = self._mj_model.actuator_ctrlrange[:, 1]
    
    # self.init_qpos=self._mj_model.qpos0
    self.init_qpos = jp.zeros(self.mjx_model.nq)
    # self.init_qvel=jp.zeros(consts.NQ_VEL)
    self.init_qvel = jp.zeros(self.mjx_model.nv)
    # self._obj_qid=mjx_env.get_qpos_ids(self.mj_model, ["obj_pos"])

     # 创建一个15x18的变换矩阵（15维动作→18维控制）
    action_to_ctrl_matrix = jp.zeros((15, self.mjx_model.nu))
    # 设置映射关系
    # 欠驱动绳索映射
    action_to_ctrl_matrix = action_to_ctrl_matrix.at[0, 0].set(1.0)                    # 控制0
    action_to_ctrl_matrix = action_to_ctrl_matrix.at[1:5, [5,7,8,10]].set(jp.eye(4))   # 控制5,7,8,10
    # 拇指关节映射
    action_to_ctrl_matrix = action_to_ctrl_matrix.at[5, 1].set(1.0)                    # 控制1
    action_to_ctrl_matrix = action_to_ctrl_matrix.at[5, 2].set(-1.0)                   # 控制2，反向
    action_to_ctrl_matrix = action_to_ctrl_matrix.at[6, 3].set(1.0)                    # 控制3
    action_to_ctrl_matrix = action_to_ctrl_matrix.at[6, 4].set(-1.0)                   # 控制4，反向
    # 侧摆映射
    action_to_ctrl_matrix = action_to_ctrl_matrix.at[7, 6].set(1.0)                    # 控制6
    action_to_ctrl_matrix = action_to_ctrl_matrix.at[8, 9].set(1.0)                    # 控制9
    action_to_ctrl_matrix = action_to_ctrl_matrix.at[8, 11].set(1.5)                   # 控制11，1.5倍
    # 手掌映射
    action_to_ctrl_matrix = action_to_ctrl_matrix.at[9:15, 12:18].set(jp.eye(6))       # 控制12-17
    
    self._action_to_ctrl_matrix = action_to_ctrl_matrix

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, obj_init_rng = jax.random.split(rng)
    qpos = self.init_qpos #重置所有关节的pos
    # Convert qpos to jax.Array to ensure it can be modified with .at
    # qpos = jp.array(self.init_qpos)
    #qpos[self._obj_qid] = para_hand_base.random_obj_pose_z(obj_init_rng)  #随机化物体的初始位置
    # qpos = qpos.at[self._obj_qid].set(para_hand_base.random_obj_pose_z(obj_init_rng))
    qvel = self.init_qvel #重置所有关节的vel为0
    
    ten_len_xy=0.015

    # Set initial tendon lengths for thumb joints
    
    ctrl = jp.zeros((self.mjx_model.nu,))
    data = mjx.make_data(self.mjx_model)
    data = data.replace(
      qpos=qpos,
      qvel=qvel,
      ctrl=ctrl
    )

    info = {
        "rng": rng,
        "step": 0,
        "steps_since_last_success": 0,
        "success_count": 0,
        "ctrl_full": jp.zeros(self.mjx_model.nu),
        "last_act": jp.zeros(15),
        "last_last_act": jp.zeros(15),
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["reward/success"] = jp.zeros((), dtype=float)
    metrics["steps_since_last_success"] = 0
    metrics["success_count"] = 0

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)

    # jax.debug.print("Model 自由度{}",self.mjx_model.nv)
    # jax.debug.print("Model 约束数量{}",self.mjx_model.nefc)
    # jax.debug.print("reset ctrl:{}", ctrl)

    return mjx_env.State(data, obs, reward, done, metrics, info)

  def _get_termination(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    del info  # Unused.
    nans = jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))
    return nans

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> mjx_env.Observation:
    
    return {
      "state": 0,
    }

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    data = mjx_env.step(model=self.mjx_model,data=state.data,action=action)

    obs = self._get_obs(data, state.info)
    reward = jp.array(0)  # 返回标量 0
    done = self._get_termination(data, state.info)

    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state
  

  # Reward terms.

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
  ) -> dict[str, jax.Array]:
    
    rewards = {
        "reach_finger": 0,
        "reach_palm": 0,
    }
    
    return rewards
  
  @property
  def xml_path(self) -> str:
    # 返回 XML 文件路径
    return consts.PARA_HAND_XML.as_posix()
  
  @property
  def action_size(self) -> int:
      return self._mjx_model.nu

  @property
  def mj_model(self):
      return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
      return self._mjx_model


def main():
    # 创建环境实例
    env = TestTask()
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    print("环境重置成功！")

if __name__ == "__main__":
    main()