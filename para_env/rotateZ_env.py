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
from para_env.para_hand_base import ParaHandEnv as BaseEnv 

def default_config() -> config_dict.ConfigDict:
  """config for ParaHandRotateZ environment. Check existing config for details."""
  return config_dict.create(
      action_scale=0.5,
      ctrl_dt=0.025,
      sim_dt=0.005,
      action_repeat=5,
      ema_alpha=1.0,
      episode_length=500,
      success_threshold=0.1,
      history_len=1,
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.05,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              angvel=1.0,
              linvel=0.0,
              pose=0.0,
              torques=0.0,
              energy=0.0,
              termination=-100.0,
              action_rate=0.0,
          ),
      ),
    #   pert_config=config_dict.create(
    #       enable=False,
    #       linear_velocity_pert=[0.0, 3.0],
    #       angular_velocity_pert=[0.0, 0.5],
    #       pert_duration_steps=[1, 100],
    #       pert_wait_steps=[60, 150],
    #   ),
      impl='jax',
      nconmax=30*8192,
      njmax=128,
  )

class ParaHandRotateZ(BaseEnv):
    """ParaHand旋转Z轴任务环境"""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):  
        super().__init__(
            xml_path=consts.TASK_XML_FILES["rotateZ"].as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()
    
    def _post_init(self) -> None:        
        # get ids for relevant joints and geoms
        self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, consts.JOINT_NAMES)
        self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.JOINT_NAMES)
        self._cube_qids = mjx_env.get_qpos_ids(self.mj_model, ["cube_freejoint"])
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._cube_geom_id = self._mj_model.geom("cube").id

        # debug info
        # jax.debug.print("hand qpos ids num: {}", self._hand_qids.shape)
        # jax.debug.print("hand qvel ids num: {}", self._hand_dqids.shape) 

        # self._target_sid=mjx_env.get_site_ids(self.mj_model, ["target"])
        self._palm_bid = self._mj_model.body("palm").id
        self._tips_bids = [self._mj_model.site(tip).id for tip in [
             "thumb_fingertip", "index_fingertip", "middle_fingertip", "ring_fingertip", "little_fingertip"
        ]]
        # self._inner_sids = [self._mj_model.site(name).id for name in consts.INNER_SITE_NAMES]
        # self._outer_sids = [self._mj_model.site(name).id for name in consts.OUTER_SITE_NAMES]

        # self._tactile_geom_ids = [self._mj_model.geom(name).id for name in consts.TACTILE_GEOM_NAMES]
        # self._tactile_geom_body_ids = [self._mj_model.body(name).id for name in consts.TACTILE_BODY_NAMES]
        
        # Initialize default pose and limits
        home_key = self._mj_model.keyframe("home")
        self._init_q = jp.array(home_key.qpos)
        self._init_q_vel = jp.array(home_key.qvel)
        self._default_pose = self._init_q[self._hand_qids]
        self._default_ctrl = jp.array(home_key.ctrl)
        
        hand_joint_ids = jp.array([self._mj_model.joint(name).id for name in consts.JOINT_NAMES])
        hand_joint_ranges =self._mj_model.jnt_range[hand_joint_ids]
        self._lowers, self._uppers = hand_joint_ranges.T
        
        # DEBUG: print initial positions
        # jax.debug.print("init qpos shape: {}", self._init_q.shape)
        # jax.debug.print("hand qid  shape: {}", self._hand_qids.shape)
        # jax.debug.print("hand dqid shape: {}", self._hand_dqids.shape)
        # jax.debug.print("default pose shape: {}", self._default_pose.shape)
        # jax.debug.print("default ctrl shape: {}", self._default_ctrl.shape)s
        # jax.debug.print("actuator lowers shape: {}", self._lowers.shape)
        # jax.debug.print("actuator uppers shape: {}", self._uppers.shape)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # Randomizes hand pos
        rng, pos_rng, vel_rng = jax.random.split(rng, 3)
        
        # q_hand = self._default_pose
        q_hand = jp.clip(
            self._default_pose + 0.1 * jax.random.normal(pos_rng, (consts.NQ_POS,)),
            self._lowers,
            self._uppers,
        )
        # jax.debug.print("q_hand shape:{}", q_hand.shape)
        
        v_hand = 0.0 * jax.random.normal(vel_rng, (consts.NV,))
        # v_hand = jp.zeros(consts.NV)
        # v_hand = self._default_hand_vel
        # jax.debug.print("v_hand shape:{}", v_hand.shape)
        
        # Randomizes cube qpos and qvel
        rng, p_rng, quat_rng = jax.random.split(rng, 3)
        start_pos = jp.array([0.0, 0.0, 0.2])
        # start_pos = jp.array([0.0, 0.0, 0.2]) + jax.random.uniform(
        #     p_rng, (3,), minval=-0.01, maxval=0.01
        # )
        # start_quat = para_hand_base.uniform_quat(quat_rng)
        start_quat = jp.array([1.0, 0.0, 0.0, 0.0])
        
        # 固定初始位置和姿态进行测试
        # start_pos = jp.array([0.0, 0.0, 0.21])
        # start_quat = jp.array([1.0, 0.0, 0.0, 0.0])
        q_cube = jp.array([*start_pos, *start_quat])
        v_cube = jp.zeros(6)
        
        # ten_len_xy=0.015
        # ctrl = jp.zeros((self.mjx_model.nu,))
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
            ctrl=self._default_ctrl,
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        
        info = {
            "rng": rng,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": data.ctrl,
            "last_cube_angvel": jp.zeros(3),
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
        metrics["reward/total"] = jp.zeros(())
    
        # TODO: change obs history size accordingly
        # obs_history = jp.zeros(self._config.history_len * 37) # 37 = state size (25 joint pos + 12 last act)
        obs_history = jp.zeros(self._config.history_len * 49) # 49 = state size (25 joint pos + 12 last act + 12 last last act)
        obs = self._get_obs(data, info, obs_history)
        reward, done = jp.zeros(2)

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        motor_targets = self._default_ctrl + action * self._config.action_scale
        # debug info
        # jax.debug.print("step action shape: {}", action.shape)
        # jax.debug.print("motor targets: {}", motor_targets)

        data = mjx_env.step(
            self.mjx_model, state.data, motor_targets, self.n_substeps
        )
        state.info["motor_targets"] = motor_targets
        
        # get observations and done signal
        obs = self._get_obs(data, state.info, state.obs["state"])
        done = self._get_termination(data)

        # get rewards
        rewards = self._get_reward(data, action, state.info, state.metrics, done)
        reward = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        } # scale rewards with config scales constatnts
        
        # update metrics
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["last_cube_angvel"] = self.get_cube_angvel(data)
        for k, v in reward.items():
            state.metrics[f"reward/{k}"] = v

        reward = sum(rewards.values()) * self.dt
        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        """
        check whether episode is done, e.g. due to nan values or cube falling below floor
        """
        fall_termination = self.get_cube_position(data)[2] < 0.1
        return fall_termination

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

        # 策略网络所能够观察到的状态
        # TODO: 这里需要更改obs_history的尺寸
        state = jp.concatenate([
            noisy_joint_qpos, # 25个关节位置
            info["last_act"], # 12个动作(执行器数量)
            info["last_last_act"], # 12个动作(执行器数量)
        ])
        obs_history = jp.roll(obs_history, state.size)
        obs_history = obs_history.at[: state.size].set(state)
        
        # all these functions should be defined in the base class, and necessary sensors should be added to the xml
        cube_pos = self.get_cube_position(data)
        palm_pos = self.get_palm_position(data)
        cube_pos_error = palm_pos - cube_pos
        cube_quat = self.get_cube_orientation(data)
        cube_angvel = self.get_cube_angvel(data)
        cube_linvel = self.get_cube_linvel(data)
        fingertip_positions = self.get_fingertip_positions(data)
        joint_torques = data.actuator_force
    
        # 供价值网络用于估算价值所需要的完整状态
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
            # **self.get_tactile_info(data), # 我觉得触觉信息不应该放在这里
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
        """计算奖励函数，结合参考设计进行改进。"""
        del metrics  # Unused.
        cube_pos = self.get_cube_position(data)
        palm_pos = self.get_palm_position(data)
        cube_pos_error = palm_pos - cube_pos
        cube_angvel = self.get_cube_angvel(data)
        cube_linvel = self.get_cube_linvel(data)

        # debug info
        # jax.debug.print("cube pos: {}", cube_pos)
        # jax.debug.print("palm pos: {}", palm_pos)
        # jax.debug.print("cube pos error: {}", cube_pos_error)
        # jax.debug.print("cube angvel: {}", cube_angvel)
        # jax.debug.print("cube linvel: {}", cube_linvel)

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
                data.qvel[self._hand_dqids], data.actuator_force[self._hand_qids]
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
        return mjx_env.get_sensor_data(self.mj_model, data, "cube_pos")

    def get_cube_orientation(self, data:mjx.Data) -> jax.Array:
        """获取方块的四元数表示"""
        return mjx_env.get_sensor_data(self.mj_model, data, "cube_quat")

    def get_cube_angvel(self, data:mjx.Data) -> jax.Array:
        """获取方块的角速度表示"""
        return mjx_env.get_sensor_data(self.mj_model, data, "cube_angvel")

    def get_cube_linvel(self, data:mjx.Data) -> jax.Array:
        """获取方块的线速度表示"""
        return mjx_env.get_sensor_data(self.mj_model, data, "cube_linvel")
    
    def get_palm_position(self, data:mjx.Data) -> jax.Array:
        """获取手掌的位置表示"""
        return mjx_env.get_sensor_data(self.mj_model, data, "palm_pos")
    
    # # Tactile sensor processing
    # def _find_contact_indices(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
    #     """Find contact indices for each tactile geom.
        
    #     Args:
    #     data: MuJoCo data
        
    #     Returns:
    #     Tuple of (contact_idx_valid, mask) arrays:
    #         - contact_idx_valid: Boolean array indicating which tactile sensors have contact
    #         - mask: Array of contact indices (-1 for no contact)
    #     """
    #     # marker点几何体ID与contact几何体的匹配检测
    #     tactile_ids = jp.array(self._tactile_geom_ids)[:, None]  # Shape: [n_tactile, 1]
    #     is_in_contact = (tactile_ids == data.contact.geom1[:data.ncon]) | (tactile_ids == data.contact.geom2[:data.ncon])
        
    #     # 选取contact序号的最小值作为marker点对应的contact序号
    #     masked_indices = jp.where(is_in_contact, 
    #                             jp.arange(data.ncon), 
    #                             jp.full_like(is_in_contact, data.ncon, dtype=int))
        
    #     # 生成mask并计算有效的接触
    #     mask = jp.where(jp.min(masked_indices, axis=1) < data.ncon, 
    #                     jp.min(masked_indices, axis=1), 
    #                     jp.full(len(self._tactile_geom_ids), -1))
        
    #     return mask >= 0, mask

    # def _extract_contact_data(self, data: mjx.Data, contact_idx_valid: jax.Array, mask: jax.Array) -> tuple[jax.Array, jax.Array]:
    #     """Extract contact distances and determine which tactile geoms are geom2.
        
    #     Args:
    #     data: MuJoCo data
    #     contact_idx_valid: Boolean array indicating which tactile sensors have contact
    #     mask: Array of contact indices (-1 for no contact)
        
    #     Returns:
    #     Tuple of (contact_dists, is_geom2):
    #         - contact_dists: Contact distances for each tactile sensor
    #         - is_geom2: Boolean array indicating if the tactile sensor is geom2 in the contact
    #     """
    #     # Extract contact distances directly
    #     contact_dists = jp.where(contact_idx_valid, data.contact.dist[mask], 0.0)
        
    #     # Determine if each tactile geom is geom2 in the contact directly
    #     is_geom2 = jp.where(
    #         contact_idx_valid, 
    #         data.contact.geom2[mask] == jp.array(self._tactile_geom_ids), 
    #         False
    #     )
        
    #     return contact_dists, is_geom2

    # def _extract_normals(self, data: mjx.Data, contact_idx_valid: jax.Array, mask: jax.Array, is_geom2: jax.Array) -> jax.Array:
    #     """Extract contact normals in world frame and flip if necessary.
        
    #     Args:
    #     data: MuJoCo data
    #     contact_idx_valid: Boolean array indicating which tactile sensors have contact
    #     mask: Array of contact indices (-1 for no contact)
    #     is_geom2: Boolean array indicating if the tactile sensor is geom2 in the contact
        
    #     Returns:
    #     Contact normals in world frame
    #     """
    #     # Extract frame data and stack into normals directly
    #     normals_world = jp.where(
    #         contact_idx_valid[:, None],
    #         jp.stack([
    #             data.contact.frame[mask, 2, 0],
    #             data.contact.frame[mask, 2, 1], 
    #             data.contact.frame[mask, 2, 2]
    #         ], axis=-1),
    #         jp.zeros((len(self._tactile_geom_ids), 3))
    #     )
        
    #     # Flip normals where the tactile geom is geom2
    #     return normals_world * jp.where(is_geom2, -1.0, 1.0)[:, None]

    # def _transform_normals_to_local(self, data: mjx.Data, contact_idx_valid: jax.Array, normals_world: jax.Array) -> jax.Array:
    #     """Transform normals from world to local body frames.
        
    #     Args:
    #     data: MuJoCo data
    #     contact_idx_valid: Boolean array indicating which tactile sensors have contact
    #     normals_world: Normals in world frame
        
    #     Returns:
    #     Normals in local body frames
    #     """
    #     # Get body orientations and convert quats to rotation matrices in one step
    #     body_rots = jax.vmap(math.quat_to_mat)(data.xquat[jp.array(self._tactile_geom_body_ids)])
        
    #     # Transform normals to local coordinates and apply mask in one operation
    #     return jp.where(
    #     contact_idx_valid[:, None],
    #     jax.vmap(lambda rot, normal: rot.T @ normal)(body_rots, normals_world),
    #     jp.zeros_like(normals_world)
    #     )

    # def _group_by_finger(self, contact_forces: jax.Array) -> dict[str, jax.Array]:
    #     """Group contact forces by finger.
        
    #     Args:
    #     contact_forces: Contact forces for all tactile sensors
        
    #     Returns:
    #     Dictionary mapping finger name to contact forces
    #     """
    #     # Directly create the grid coordinates
    #     # TODO: 如果你更换了tactile传感器的排列方式，这里也需要相应修改
    #     x = jp.tile(jp.arange(5) * 0.001, 5)
    #     y = jp.repeat(jp.arange(5) * 0.001, 5)
    #     grid = jp.stack([x, y], axis=1)
        
    #     # Compute finger forces in one step
    #     geoms_per_finger = len(self._tactile_geom_ids) // 5
        
    #     # Create dictionary directly without intermediate lists
    #     return {
    #         name: jp.concatenate([grid, contact_forces[i*geoms_per_finger:(i+1)*geoms_per_finger]], axis=1)
    #         for i, name in enumerate(["thumb", "index", "middle", "ring", "little"])
    #     }

    # def get_tactile_info(self, data: mjx.Data) -> dict[str, jax.Array]:
    #     """Get tactile information for all fingers.
        
    #     Args:
    #         data: MuJoCo data
        
    #     Returns:
    #         Dictionary mapping finger name to contact forces,
    #         contact forces are forces object apply on those tac balls, with shape [n_tactile_per_finger, 5].
    #         For example, if we use 25 tac balls per finger, the shape will be [25, 5]
    #     """
    #     # Find contact indices for tactile geoms
    #     contact_idx_valid, mask = self._find_contact_indices(data)
    #     # jax.debug.print("mask shape:{}", mask.shape)
    #     # Extract contact data
    #     contact_dists, is_geom2 = self._extract_contact_data(data, contact_idx_valid, mask)
    #     # print("contact_dists shape:", contact_dists.shape)
    #     # print("is_geom2 shape:", is_geom2.shape)

    #     # Extract normals in world frame
    #     normals_world = self._extract_normals(data, contact_idx_valid, mask, is_geom2)
    #     # print("normals_world shape:", normals_world.shape)

    #     # Transform normals to local body frames
    #     normals_local = self._transform_normals_to_local(data, contact_idx_valid, normals_world)
    #     # print("normals_local shape:", normals_local.shape)
    #     # Calculate contact forces
    #     contact_forces = normals_local * contact_dists[:, None]
    #     # print("contact_forces shape:", contact_forces.shape)
    #     # Group by finger
    #     return self._group_by_finger(contact_forces)

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
        mj_model = ParaHandRotateZ().mj_model
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
