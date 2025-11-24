"""ParaHandReorient任务环境"""
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from functools import partial
from ml_collections import config_dict
import mujoco as mj
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward

from para_env import para_hand_constants as consts
from para_env import para_hand_base
from para_env.para_hand_base import ParaHandEnv

def default_config() -> config_dict.ConfigDict:
  """config for ParaHandReorient environment. Check existing config for details."""
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.002,

      action_scale=0.5,
      action_repeat=1,
      ema_alpha=1.0,
      episode_length=256,
      success_threshold=0.1,
      history_len=1,
      # 观测噪声配置
      obs_noise=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.05,
              cube_pos=0.02,
              cube_ori=0.1,
          ),
          random_ori_injection_prob=0.0,
      ),
      # 奖励函数配置
      reward_config=config_dict.create(
          scales=config_dict.create(
              orientation=5.0,
            #   position=0.5,
            #   termination=-100.0,
            #   hand_pose=-0.5,
            #   action_rate=-0.001,
            #   joint_vel=0.0,
            #   energy=-1e-3,
            #   is_success=10.0,
            #   fail_terminate=0.0,
          ),
          success_reward=100.0,
      ),
      # 扰动配置
      pert_config=config_dict.create(
          enable=False,
          linear_velocity_pert=[0.0, 3.0],
          angular_velocity_pert=[0.0, 0.5],
          pert_duration_steps=[1, 100],
          pert_wait_steps=[60, 150],
      ),

      impl='jax',
      nconmax=4096,
      njmax=256,
  )

class ParaHandReorient(ParaHandEnv):
    """ParaHand重定位任务环境"""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):  
        super().__init__(
            xml_path=consts.TASK_XML_FILES["reorient"].as_posix(),
            # change const xml to para change xml later
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()
    
    def _post_init(self) -> None:        
        # get ids for relevant joints and geoms
        self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, consts.JOINT_NAMES)
        self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.JOINT_NAMES)
        self._cube_qids = mjx_env.get_qpos_ids(self.mj_model, ["cube_freejoint"])
        # self._floor_geom_id = self._mj_model.geom("floor").id
        self._cube_geom_id = self._mj_model.geom("cube").id
        # debug info
        # print("hand qpos ids num:", self._hand_qids.shape)
        # print("hand qvel ids num:", self._hand_dqids.shape) 

        # self._target_sid=mjx_env.get_site_ids(self.mj_model, ["target"])
        self._palm_bid = self._mj_model.body("palm").id
        self._tips_bids = [self._mj_model.body(tip).id for tip in [
            "thumb_fingertip", "index_fingertip", "middle_fingertip", "ring_fingertip", "little_fingertip"
        ]]
        # 不清楚inner和outer sites的作用，暂时注释掉. If needed, can uncomment and add in xml files
        # self._inner_sids = [self._mj_model.site(name).id for name in consts.INNER_SITE_NAMES]
        # self._outer_sids = [self._mj_model.site(name).id for name in consts.OUTER_SITE_NAMES]
        
        # get ids for tactile geoms
        self._tactile_geom_ids = [self._mj_model.geom(name).id for name in consts.TACTILE_GEOM_NAMES]
        self._tactile_geom_body_ids = [self._mj_model.body(name).id for name in consts.TACTILE_BODY_NAMES]
        
        # Initialize default pose and limits
        home_key = self._mj_model.keyframe("home")
        self._init_q = jp.array(home_key.qpos)
        self._init_q_vel = jp.array(home_key.qvel)

        self._default_pose = self._init_q[self._hand_qids]
        self._default_hand_vel = self._init_q_vel[self._hand_qids]
        self._init_cube_pos = self._init_q[self._cube_qids][:3]
        self._init_cube_vel = self._init_q_vel[self._cube_qids][:3]
        self._lowers, self._uppers = self.mj_model.actuator_ctrlrange.T
        
        # DEBUG: print initial positions
        # jax.debug.print("init qpos: {}", self._init_q)
        # jax.debug.print("hand qid: {}", self._hand_qids)
        # jax.debug.print("hand dqid: {}", self._hand_dqids)


    def reset(self, rng: jax.Array) -> mjx_env.State:
        # Randomizes hand pos
        rng, pos_rng, vel_rng = jax.random.split(rng, 3)
        
        q_hand = self._default_pose
        # q_hand = jp.zeros(consts.NQ_POS)
        # q_hand = jp.clip(
        #     self._default_pose + 0.1 * jax.random.normal(pos_rng, (consts.NQ_POS,)),
        #     self._lowers,
        #     self._uppers,
        # )
        # jax.debug.print("q_hand shape:{}", q_hand.shape)
        
        # v_hand = 0.0 * jax.random.normal(vel_rng, (consts.NV,))
        # v_hand = jp.zeros(consts.NV)
        v_hand = self._default_hand_vel
        # jax.debug.print("v_hand shape:{}", v_hand.shape)
        
        # Randomizes cube qpos and qvel
        rng, p_rng, quat_rng = jax.random.split(rng, 3)
        # start_pos = jp.array([0.1, 0.0, 0.2]) + jax.random.uniform(
        #     p_rng, (3,), minval=-0.01, maxval=0.01
        # )
        # start_quat = para_hand_base.uniform_quat(quat_rng)
        
        # 固定初始位置和姿态进行测试
        start_pos = jp.array([0.0, 0.0, 0.21])  # 手掌的上表面大概是 z=0.2
        start_quat = jp.array([1.0, 0.0, 0.0, 0.0])
        q_cube = jp.array([*start_pos, *start_quat])
        v_cube = jp.zeros(6)
        
        ten_len_xy=0.015
        ctrl = jp.zeros((self.mjx_model.nu,))
        # ctrl = ctrl.at[1:5].set(ten_len_xy)
        
        # Set initial tendon lengths for thumb joints
        qpos = jp.concatenate([q_hand, q_cube])
        qvel = jp.concatenate([v_hand, v_cube])

        # jax.debug.print("reset qpos :{}", qpos)
        # jax.debug.print("reset qvel :{}", qvel)
        # jax.debug.print("reset ctrl :{}", ctrl)
        
        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
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
            "last_act": jp.zeros(consts.NU),
            "last_last_act": jp.zeros(consts.NU),
            "last_cube_angvel": jp.zeros(3),
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
        metrics["reward/total"] = jp.zeros((),dtype=float)
        # metrics["reward/success"] = jp.zeros((), dtype=float)
        # metrics["steps_since_last_success"] = 0
        # metrics["success_count"] = 0

        # obs_history存在循环递归计算的问题，暂时先不使用
        # obs_history = jp.zeros((self._config.history_len, self.observation_size))
        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        data = mjx_env.step(model=self.mjx_model,data=state.data,action=action)
        
        # get observations and done signal
        obs = self._get_obs(data, state.info)
        done = self._get_termination(data)

        # get rewards
        rewards = self._get_reward(data, action, state.info, state.metrics, done)
        reward = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        } # scale rewards with config scales constatnts
        
        # update metrics
        for k, v in reward.items():
            state.metrics[f"reward/{k}"] = v
        # state.metrics["steps_since_last_success"] = state.info["steps_since_last_success"]
        # state.metrics["success_count"] = state.info["success_count"]

        # update total reward metrics
        reward = sum(reward.values()) * self.dt  # total reward
        state.metrics["reward/total"] = reward

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state
    

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        """
        check whether episode is done, e.g. due to nan values or cube falling below floor
        """
        # check invalid velocities or poses
        # nans = jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))
        # check cube falling below floor
        fall_termination = self.get_cube_position(data)[2] < -0.05
        return fall_termination

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> mjx_env.Observation:
        joint_qpos = data.qpos[self._hand_qids]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        
        # TODO： add noise to cube position and orientation
        # noisy_joint_qpos = (
        #     joint_qpos
        #     + (2 * jax.random.uniform(noise_rng, shape=joint_qpos.shape) - 1)
        #     * self._config.noise_config.level
        #     * self._config.noise_config.scales.joint_pos
        # ) # add some noise to joint positions

        # 策略网络所能够观察到的状态
        # TODO: 我认为触觉信息也应该解包后放在这里
        state = jp.concatenate([
            joint_qpos,
            # noisy_joint_qpos,
            info["last_act"],
            cube_pos,
            cube_quat,
            cube_angvel,
            cube_linvel,
            # *self.get_tactile_info(data),
        ])
        
        # all these functions should be defined in the base class, and necessary sensors should be added to the xml
        cube_pos = self.get_cube_position(data)
        palm_pos = self.get_palm_position(data)
        cube_pos_error = palm_pos - cube_pos
        cube_quat = self.get_cube_orientation(data)
        cube_angvel = self.get_cube_angvel(data)
        cube_linvel = self.get_cube_linvel(data)
        fingertip_positions = self.get_fingertip_positions(data)
    
        # 供价值网络用于估算价值所需要的完整状态
        privileged_state = jp.concatenate([
            state,
            joint_qpos,
            data.qvel[self._hand_dqids],
            fingertip_positions,
            cube_pos_error,
            cube_quat,
            cube_angvel,
            cube_linvel,
        ])
    
        return {
            # **self.get_tactile_info(data), # 我觉得触觉信息不应该放在这里
            "privileged_state": privileged_state,
            "state": state,
        }

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        """计算奖励函数，结合参考设计进行改进。"""
        del metrics  # Unused.

        # 获取方块和目标的姿态
        cube_ori = self.get_cube_orientation(data)
        # TODO: replace with a easily change goal orientation function
        # goal_ori = self.orientation_target  # 目标姿态
        goal_ori = jp.array([0.0, 0.0, 0.0, 1.0])   # 绕 z 轴旋转 180 度
        quat_diff = math.quat_mul(cube_ori, math.quat_inv(goal_ori))
        quat_diff = math.normalize(quat_diff)  # 归一化四元数，防止出现除0的情况
        ori_error =  2.0 * jp.asin(jp.clip(math.norm(quat_diff[1:]), a_max=1.0))  # 姿态误差
        reward_orientation = reward.tolerance(ori_error, (0, 0.2), margin=jp.pi, sigmoid="linear")

        # # 获取方块的位置误差
        # cube_pos = self.get_cube_position(data)
        # cube_pos_mse = jp.linalg.norm(cube_pos - self._init_cube_pos)
        # reward_position = (
        #     self._config.reward_config.scales.position
        #     * (1 - reward.tolerance(cube_pos_mse, bounds=(0, 0.02), margin=10))
        # )

        # # 检查是否失败（方块掉落）
        # fail_terminate = cube_pos[2] < -0.05
        # reward_termination = (
        #     self._config.reward_config.scales.termination
        #     * fail_terminate
        #     * (self._config.episode_length - info["step"])
        # )

        # # 手指位置尽量少偏离初始状态
        # reward_hand_pose = (
        #     self._config.reward_config.scales.hand_pose
        #     * jp.sum(jp.square(data.qpos[self._hand_qids] - self._default_pose))
        # )

        # # 手指动作前后变化尽量小
        # reward_action_rate = (
        #     self._config.reward_config.scales.action_rate
        #     * jp.sum(jp.square(action - info["last_act"]))
        # )

        # # 手指运动速度尽量小
        # reward_energy = (
        #     self._config.reward_config.scales.energy * jp.sum(jp.square(action))
        # )

        # # 成功奖励（姿态误差小于阈值）
        # is_success = ori_error < self._config.success_threshold
        # is_success = self._config.reward_config.success_reward * is_success

        # 奖励信息
        rewards = {
            "orientation": reward_orientation,
            # "position": reward_position,
            # "hand_pose": reward_hand_pose,
            # "action_rate": reward_action_rate,
            # "energy": reward_energy,
            # "fail_terminate": fail_terminate,
            # "is_success": is_success,
        }
        return rewards

    # Additional sensors specially for this task env
    # TODO: check the sensor names and add these sensors to the xml if not exists
    def get_fingertip_positions(self, data:mjx.Data) -> jax.Array:
        """获取所有指尖的位置表示"""
        sensor_names = [
            "thumb_fingertip_pos",
            "index_fingertip_pos",
            "middle_fingertip_pos",
            "ring_fingertip_pos",
            "little_fingertip_pos",
        ]
        return jp.concatenate([
            mjx_env.get_sensor_data(self.mj_model, data, sensor_name)
            for sensor_name in sensor_names
        ])

    def get_cube_position(self, data:mjx.Data) -> jax.Array:
        """获取方块的位置表示""" 
        return mjx_env.get_sensor_data(self.mj_model, data, "cube_freejoint_frame_origin_pos")

    def get_cube_orientation(self, data:mjx.Data) -> jax.Array:
        """获取方块的四元数表示"""
        return mjx_env.get_sensor_data(self.mj_model, data, "cube_freejoint_frame_origin_quat")

    def get_cube_angvel(self, data:mjx.Data) -> jax.Array:
        """获取方块的角速度表示"""
        return mjx_env.get_sensor_data(self.mj_model, data, "cube_freejoint_angvel")

    def get_cube_linvel(self, data:mjx.Data) -> jax.Array:
        """获取方块的线速度表示"""
        return mjx_env.get_sensor_data(self.mj_model, data, "cube_freejoint_linvel")
    
    def get_palm_position(self, data:mjx.Data) -> jax.Array:
        """获取手掌的位置表示"""
        return mjx_env.get_sensor_data(self.mj_model, data, "palm_pos")
    
    # Tactile sensor processing
    def _find_contact_indices(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
        """Find contact indices for each tactile geom.
        
        Args:
        data: MuJoCo data
        
        Returns:
        Tuple of (contact_idx_valid, mask) arrays:
            - contact_idx_valid: Boolean array indicating which tactile sensors have contact
            - mask: Array of contact indices (-1 for no contact)
        """
        # marker点几何体ID与contact几何体的匹配检测
        tactile_ids = jp.array(self._tactile_geom_ids)[:, None]  # Shape: [n_tactile, 1]
        is_in_contact = (tactile_ids == data.contact.geom1[:data.ncon]) | (tactile_ids == data.contact.geom2[:data.ncon])
        
        # 选取contact序号的最小值作为marker点对应的contact序号
        masked_indices = jp.where(is_in_contact, 
                                jp.arange(data.ncon), 
                                jp.full_like(is_in_contact, data.ncon, dtype=int))
        
        # 生成mask并计算有效的接触
        mask = jp.where(jp.min(masked_indices, axis=1) < data.ncon, 
                        jp.min(masked_indices, axis=1), 
                        jp.full(len(self._tactile_geom_ids), -1))
        
        return mask >= 0, mask

    def _extract_contact_data(self, data: mjx.Data, contact_idx_valid: jax.Array, mask: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Extract contact distances and determine which tactile geoms are geom2.
        
        Args:
        data: MuJoCo data
        contact_idx_valid: Boolean array indicating which tactile sensors have contact
        mask: Array of contact indices (-1 for no contact)
        
        Returns:
        Tuple of (contact_dists, is_geom2):
            - contact_dists: Contact distances for each tactile sensor
            - is_geom2: Boolean array indicating if the tactile sensor is geom2 in the contact
        """
        # Extract contact distances directly
        contact_dists = jp.where(contact_idx_valid, data.contact.dist[mask], 0.0)
        
        # Determine if each tactile geom is geom2 in the contact directly
        is_geom2 = jp.where(
            contact_idx_valid, 
            data.contact.geom2[mask] == jp.array(self._tactile_geom_ids), 
            False
        )
        
        return contact_dists, is_geom2

    def _extract_normals(self, data: mjx.Data, contact_idx_valid: jax.Array, mask: jax.Array, is_geom2: jax.Array) -> jax.Array:
        """Extract contact normals in world frame and flip if necessary.
        
        Args:
        data: MuJoCo data
        contact_idx_valid: Boolean array indicating which tactile sensors have contact
        mask: Array of contact indices (-1 for no contact)
        is_geom2: Boolean array indicating if the tactile sensor is geom2 in the contact
        
        Returns:
        Contact normals in world frame
        """
        # Extract frame data and stack into normals directly
        normals_world = jp.where(
            contact_idx_valid[:, None],
            jp.stack([
                data.contact.frame[mask, 2, 0],
                data.contact.frame[mask, 2, 1], 
                data.contact.frame[mask, 2, 2]
            ], axis=-1),
            jp.zeros((len(self._tactile_geom_ids), 3))
        )
        
        # Flip normals where the tactile geom is geom2
        return normals_world * jp.where(is_geom2, -1.0, 1.0)[:, None]

    def _transform_normals_to_local(self, data: mjx.Data, contact_idx_valid: jax.Array, normals_world: jax.Array) -> jax.Array:
        """Transform normals from world to local body frames.
        
        Args:
        data: MuJoCo data
        contact_idx_valid: Boolean array indicating which tactile sensors have contact
        normals_world: Normals in world frame
        
        Returns:
        Normals in local body frames
        """
        # Get body orientations and convert quats to rotation matrices in one step
        body_rots = jax.vmap(math.quat_to_mat)(data.xquat[jp.array(self._tactile_geom_body_ids)])
        
        # Transform normals to local coordinates and apply mask in one operation
        return jp.where(
        contact_idx_valid[:, None],
        jax.vmap(lambda rot, normal: rot.T @ normal)(body_rots, normals_world),
        jp.zeros_like(normals_world)
        )

    def _group_by_finger(self, contact_forces: jax.Array) -> dict[str, jax.Array]:
        """Group contact forces by finger.
        
        Args:
        contact_forces: Contact forces for all tactile sensors
        
        Returns:
        Dictionary mapping finger name to contact forces
        """
        # Directly create the grid coordinates
        # TODO: 如果你更换了tactile传感器的排列方式，这里也需要相应修改
        x = jp.tile(jp.arange(5) * 0.001, 5)
        y = jp.repeat(jp.arange(5) * 0.001, 5)
        grid = jp.stack([x, y], axis=1)
        
        # Compute finger forces in one step
        geoms_per_finger = len(self._tactile_geom_ids) // 5
        
        # Create dictionary directly without intermediate lists
        return {
            name: jp.concatenate([grid, contact_forces[i*geoms_per_finger:(i+1)*geoms_per_finger]], axis=1)
            for i, name in enumerate(["thumb", "index", "middle", "ring", "little"])
        }

    def get_tactile_info(self, data: mjx.Data) -> dict[str, jax.Array]:
        """Get tactile information for all fingers.
        
        Args:
            data: MuJoCo data
        
        Returns:
            Dictionary mapping finger name to contact forces,
            contact forces are forces object apply on those tac balls, with shape [n_tactile_per_finger, 5].
            For example, if we use 25 tac balls per finger, the shape will be [25, 5]
        """
        # Find contact indices for tactile geoms
        contact_idx_valid, mask = self._find_contact_indices(data)
        # jax.debug.print("mask shape:{}", mask.shape)
        # Extract contact data
        contact_dists, is_geom2 = self._extract_contact_data(data, contact_idx_valid, mask)
        # print("contact_dists shape:", contact_dists.shape)
        # print("is_geom2 shape:", is_geom2.shape)

        # Extract normals in world frame
        normals_world = self._extract_normals(data, contact_idx_valid, mask, is_geom2)
        # print("normals_world shape:", normals_world.shape)

        # Transform normals to local body frames
        normals_local = self._transform_normals_to_local(data, contact_idx_valid, normals_world)
        # print("normals_local shape:", normals_local.shape)
        # Calculate contact forces
        contact_forces = normals_local * contact_dists[:, None]
        # print("contact_forces shape:", contact_forces.shape)
        # Group by finger
        return self._group_by_finger(contact_forces)

    # some necessary properties, donot change this part
    @property
    def xml_path(self) -> str:
        # 返回 XML 文件路径
        return consts.TASK_XML_FILES["reorient"].as_posix()
        # return consts.PARA_HAND_XML.as_posix()
    
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
        mj_model = ParaHandReorient().mj_model
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
    env = ParaHandReorient()
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    print("环境重置成功！")