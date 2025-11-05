"""ParaHand旋转任务环境"""
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
from para_env import para_hand_base
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

class ParaHandRotate(ParaHandEnv):
    """ParaHand旋转任务环境"""

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
        
        # get ids for relevant joints and geoms
        self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, consts.JOINT_NAMES)
        self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.JOINT_NAMES)
        self._cube_qids = mjx_env.get_qpos_ids(self.mj_model, ["cube_freejoint"])
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._cube_geom_id = self._mj_model.geom("cube").id

        # Initialize default pose and limits
        home_key = self._mj_model.keyframe("home") # TODO: 确认home keyframe是否存在
        self._init_q = jp.array(home_key.qpos)
        self._default_pose = self._init_q[self._hand_qids]
        self._lowers, self._uppers = self.mj_model.actuator_ctrlrange.T

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # Randomizes hand pos
        rng, pos_rng, vel_rng = jax.random.split(rng, 3)
        q_hand = jp.clip(
            self._default_pose + 0.1 * jax.random.normal(pos_rng, (consts.NQ,)),
            self._lowers,
            self._uppers,
        )
        v_hand = 0.0 * jax.random.normal(vel_rng, (consts.NV,))

        # Randomizes cube qpos and qvel
        rng, p_rng, quat_rng = jax.random.split(rng, 3)
        start_pos = jp.array([0.1, 0.0, 0.05]) + jax.random.uniform(
            p_rng, (3,), minval=-0.01, maxval=0.01
        )     
        start_quat = para_hand_base.uniform_quat(quat_rng)
        q_cube = jp.array([*start_pos, *start_quat])
        v_cube = jp.zeros(6)
        
        ten_len_xy=0.015

        # Set initial tendon lengths for thumb joints
        qpos = jp.concatenate([q_hand, q_cube])
        qvel = jp.concatenate([v_hand, v_cube])
        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=q_hand,
            mocap_pos=jp.array([-100.0, -100.0, -100.0]),  # Hide goal for task.
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        info = {
            "rng": rng,
            "step": 0,
            "steps_since_last_success": 0,
            "success_count": 0,
            "ctrl_full": jp.zeros(self.mjx_model.nu),
            "last_act": jp.zeros(15),
            "last_last_act": jp.zeros(15),
            "last_cube_angvel": jp.zeros(3),
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
        metrics["reward/success"] = jp.zeros((), dtype=float)
        metrics["steps_since_last_success"] = 0
        metrics["success_count"] = 0

        obs_history = jp.zeros((self._config.history_len, self.observation_size))
        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        data = mjx_env.step(model=self.mjx_model,data=state.data,action=action)

        obs = self._get_obs(data, state.info, state.obs["state"])
        done = self._get_termination(data, state.info)

        rewards = self._get_reward(data, action, state.info, state.metrics, done)
        reward = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        } # scale rewards with config scales constatnts
        reward = sum(reward.values()) * self.dt  # total reward
        
        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state
    

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        """
        check whether episode is done, e.g. due to nan values or cube falling below floor
        """
        # check invalid velocities or poses
        nans = jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))
        # check cube falling below floor
        fall_termination = self.get_cube_position(data)[2] < -0.05

        return fall_termination | nans

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any], obs_history: jax.Array
    ) -> mjx_env.Observation:
        joint_qpos = data.qpos[self._hand_qids]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_qpos = (
            joint_qpos
            + (2 * jax.random.uniform(noise_rng, shape=joint_qpos.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_pos
        ) # add some noise to joint positions

        state = jp.concatenate([
            noisy_joint_qpos,
            info["last_act"],
        ])
        obs_history = jp.roll(obs_history, state.size)
        obs_history = obs_history.at[: state.size].set(state)
        
        cube_pos = self.get_cube_position(data)
        palm_pos = self.get_palm_position(data)
        cube_pos_error = palm_pos - cube_pos
        cube_quat = self.get_cube_orientation(data)
        cube_angvel = self.get_cube_angvel(data)
        cube_linvel = self.get_cube_linvel(data)
        fingertip_positions = self.get_fingertip_positions(data)
        joint_torques = data.actuator_force
    
        privileged_state = jp.concatenate([
            state,
            joint_qpos,
            data.qvel[self._hand_dqids],
            joint_torques,
            fingertip_positions,
            cube_pos_error,
            cube_quat,
            cube_angvel,
            cube_linvel,
        ])
    
        return {
            "state": obs_history,
            "privileged_state": privileged_state,
        }

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        del metrics  # Unused.
        cube_pos = self.get_cube_position(data)
        palm_pos = self.get_palm_position(data)
        cube_pos_error = palm_pos - cube_pos
        cube_angvel = self.get_cube_angvel(data)
        cube_linvel = self.get_cube_linvel(data)
        return {
            "angvel": self._reward_angvel(cube_angvel, cube_pos_error),
            "linvel": self._cost_linvel(cube_linvel),
            "termination": done,
            "action_rate": self._cost_action_rate(
                action, info["last_act"], info["last_last_act"]
            ),
            "pose": self._cost_pose(data.qpos[self._hand_qids]),
            "torques": self._cost_torques(data.actuator_force),
            "energy": self._cost_energy(
                data.qvel[self._hand_dqids], data.actuator_force
            ),
        }

    def _cost_torques(self, torques: jax.Array) -> jax.Array:
        return jp.sum(jp.square(torques))
    
    def _cost_energy(
        self, qvel: jax.Array, qfrc_actuator: jax.Array
    ) -> jax.Array:
        return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))
    
    def _cost_linvel(self, cube_linvel: jax.Array) -> jax.Array:
        return jp.linalg.norm(cube_linvel, ord=1, axis=-1)
    
    def _reward_angvel(
        self, cube_angvel: jax.Array, cube_pos_error: jax.Array
    ) -> jax.Array:
        # Unconditionally maximize angvel in the z-direction.
        del cube_pos_error  # Unused.
        return cube_angvel @ jp.array([0.0, 0.0, 1.0])
    
    def _cost_action_rate(
        self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
    ) -> jax.Array:
        del last_last_act  # Unused.
        return jp.sum(jp.square(act - last_act))
    
    def _cost_pose(self, joint_angles: jax.Array) -> jax.Array:
        return jp.sum(jp.square(joint_angles - self._default_pose))

    # some necessary properties
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

def domain_randomize(model: mjx.Model, rng: jax.Array):
        mj_model = ParaHandRotate().mj_model
        cube_geom_id = mj_model.geom("cube").id
        cube_body_id = mj_model.body("cube").id
        hand_qids = mjx_env.get_qpos_ids(mj_model, consts.JOINT_NAMES)
        hand_body_names = [
            "palm",
            "if_bs",
            "if_px",
            "if_md",
            "if_ds",
            "mf_bs",
            "mf_px",
            "mf_md",
            "mf_ds",
            "rf_bs",
            "rf_px",
            "rf_md",
            "rf_ds",
            "th_mp",
            "th_bs",
            "th_px",
            "th_ds",
        ]
        hand_body_ids = np.array([mj_model.body(n).id for n in hand_body_names])
        fingertip_geoms = ["th_tip", "if_tip", "mf_tip", "rf_tip"]
        fingertip_geom_ids = [mj_model.geom(g).id for g in fingertip_geoms]

        @jax.vmap
        def rand(rng):
            rng, key = jax.random.split(rng)
            # Fingertip friction: =U(0.5, 1.0).
            fingertip_friction = jax.random.uniform(key, (1,), minval=0.5, maxval=1.0)
            geom_friction = model.geom_friction.at[fingertip_geom_ids, 0].set(
                fingertip_friction
            )

            # Scale cube mass: *U(0.8, 1.2).
            rng, key1, key2 = jax.random.split(rng, 3)
            dmass = jax.random.uniform(key1, minval=0.8, maxval=1.2)
            cube_mass = model.body_mass[cube_body_id]
            body_inertia = model.body_inertia.at[cube_body_id].set(
                model.body_inertia[cube_body_id] * dmass
            )
            dpos = jax.random.uniform(key2, (3,), minval=-5e-3, maxval=5e-3)
            body_ipos = model.body_ipos.at[cube_body_id].set(
                model.body_ipos[cube_body_id] + dpos
            )

            # Jitter qpos0: +U(-0.05, 0.05).
            rng, key = jax.random.split(rng)
            qpos0 = model.qpos0
            qpos0 = qpos0.at[hand_qids].set(
                qpos0[hand_qids]
                + jax.random.uniform(key, shape=(16,), minval=-0.05, maxval=0.05)
            )

            # Scale static friction: *U(0.9, 1.1).
            rng, key = jax.random.split(rng)
            frictionloss = model.dof_frictionloss[hand_qids] * jax.random.uniform(
                key, shape=(16,), minval=0.5, maxval=2.0
            )
            dof_frictionloss = model.dof_frictionloss.at[hand_qids].set(frictionloss)

            # Scale armature: *U(1.0, 1.05).
            rng, key = jax.random.split(rng)
            armature = model.dof_armature[hand_qids] * jax.random.uniform(
                key, shape=(16,), minval=1.0, maxval=1.05
            )
            dof_armature = model.dof_armature.at[hand_qids].set(armature)

            # Scale all link masses: *U(0.9, 1.1).
            rng, key = jax.random.split(rng)
            dmass = jax.random.uniform(
                key, shape=(len(hand_body_ids),), minval=0.9, maxval=1.1
            )
            body_mass = model.body_mass.at[hand_body_ids].set(
                model.body_mass[hand_body_ids] * dmass
            )

            # Joint stiffness: *U(0.8, 1.2).
            rng, key = jax.random.split(rng)
            kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
                key, (model.nu,), minval=0.8, maxval=1.2
            )
            actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
            actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)

            # Joint damping: *U(0.8, 1.2).
            rng, key = jax.random.split(rng)
            kd = model.dof_damping[hand_qids] * jax.random.uniform(
                key, (16,), minval=0.8, maxval=1.2
            )
            dof_damping = model.dof_damping.at[hand_qids].set(kd)

            return (
                geom_friction,
                body_mass,
                body_inertia,
                body_ipos,
                qpos0,
                dof_frictionloss,
                dof_armature,
                dof_damping,
                actuator_gainprm,
                actuator_biasprm,
            )

        (
            geom_friction,
            body_mass,
            body_inertia,
            body_ipos,
            qpos0,
            dof_frictionloss,
            dof_armature,
            dof_damping,
            actuator_gainprm,
            actuator_biasprm,
        ) = rand(rng)

        in_axes = jax.tree_util.tree_map(lambda x: None, model)
        in_axes = in_axes.tree_replace({
            "geom_friction": 0,
            "body_mass": 0,
            "body_inertia": 0,
            "body_ipos": 0,
            "qpos0": 0,
            "dof_frictionloss": 0,
            "dof_armature": 0,
            "dof_damping": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
        })

        model = model.tree_replace({
            "geom_friction": geom_friction,
            "body_mass": body_mass,
            "body_inertia": body_inertia,
            "body_ipos": body_ipos,
            "qpos0": qpos0,
            "dof_frictionloss": dof_frictionloss,
            "dof_armature": dof_armature,
            "dof_damping": dof_damping,
            "actuator_gainprm": actuator_gainprm,
            "actuator_biasprm": actuator_biasprm,
        })

        return model, in_axes

if __name__ == "__main__":
    env = ParaHandRotate()
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    print("环境重置成功！")