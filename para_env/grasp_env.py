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
    return config_dict.create(
        action_scale=0.5,
        ctrl_dt=0.025,
        sim_dt=0.005,
        action_repeat=5,
        ema_alpha=1.0,
        episode_length=500,
        success_threshold=0.1,
        history_len=1,
        reward_config=config_dict.create(
            scales=config_dict.create(
                reach_palm=10.0,
                inner_dist=0.02,
                outer_dist=15.0,
                finger_bend=0.1,
                finger_prox=0.1,
                contact=30.0,
                lift=0.1,
                action_smooth=-0.1,
            ),
        ),
        impl='jax',
        nconmax=30*8192,
        njmax=64,
    )

class ParaHandGrasp(ParaHandEnv):
    """ParaHand抓取任务环境"""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):  
        super().__init__(
            xml_path=consts.TASK_XML_FILES["grasp"].as_posix(),
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
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._cube_geom_id = self._mj_model.geom("cube").id

        # debug info
        # jax.debug.print("hand qpos ids num: {}", self._hand_qids.shape)
        # jax.debug.print("hand qvel ids num: {}", self._hand_dqids.shape) 

        self._target_sid = self._mj_model.site("target").id
        self._palm_bid = self._mj_model.body("palm").id
        self._tips_sids = jp.array([self._mj_model.site(tip).id for tip in [
                "thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"
        ]])
        self._inner_sids = jp.array([self._mj_model.site(name).id for name in consts.INNER_SITE_NAMES])
        self._outer_sids = jp.array([self._mj_model.site(name).id for name in consts.OUTER_SITE_NAMES])

        # self._tactile_geom_ids = jp.array([self._mj_model.geom(name).id for name in consts.TACTILE_GEOM_NAMES])
        # self._tactile_geom_body_ids = jp.array([self._mj_model.body(name).id for name in consts.TACTILE_BODY_NAMES])
        
        # Initialize default pose and limits
        home_key = self._mj_model.keyframe("home")
        self._init_q = jp.array(home_key.qpos)
        self._init_q_vel = jp.array(home_key.qvel)
        self._default_pose = self._init_q[self._hand_qids]
        self._default_ctrl = jp.array(home_key.ctrl)
        
        hand_joint_ids = jp.array([self._mj_model.joint(name).id for name in consts.JOINT_NAMES])
        hand_joint_ranges =self._mj_model.jnt_range[hand_joint_ids]
        self._lowers, self._uppers = hand_joint_ranges.T

        # Get actuator ranges
        self._act_lowers, self._act_uppers = jp.array(self._mj_model.actuator_ctrlrange.T)
        
        # Cube dimensions (half-lengths)
        # Get cube size from model directly to avoid hardcoding
        self.cube_half_size = self._mj_model.geom_size[self._cube_geom_id]
        
        # DEBUG: print initial positions
        # jax.debug.print("init qpos shape: {}", self._init_q.shape)
        # jax.debug.print("hand qid  shape: {}", self._hand_qids.shape)
        # jax.debug.print("hand dqid shape: {}", self._hand_dqids.shape)
        # jax.debug.print("default pose shape: {}", self._default_pose.shape)
        # jax.debug.print("default ctrl shape: {}", self._default_ctrl.shape)
        # jax.debug.print("actuator lowers shape: {}", self._lowers.shape)
        # jax.debug.print("actuator uppers shape: {}", self._uppers.shape)


    def reset(self, rng: jax.Array) -> mjx_env.State:
        # Randomizes hand pos
        rng, pos_rng, vel_rng = jax.random.split(rng, 3)
        
        q_hand = self._default_pose
        # q_hand = jp.clip(
        #     self._default_pose + 0.1 * jax.random.normal(pos_rng, (consts.NQ_POS,)),
        #     self._lowers,
        #     self._uppers,
        # )
        
        v_hand = 0.0 * jax.random.normal(vel_rng, (consts.NV,))
        # DEBUG: print randomized hand states shape
        # jax.debug.print("q_hand shape:{}", q_hand.shape)
        # jax.debug.print("v_hand shape:{}", v_hand.shape)
        
        # Randomizes cube qpos and qvel
        rng, p_rng, quat_rng = jax.random.split(rng, 3)
        # start_pos = jp.array([0.1, 0.0, 0.2]) + jax.random.uniform(
        #     p_rng, (3,), minval=-0.01, maxval=0.01
        # )
        # start_quat = para_hand_base.uniform_quat(quat_rng)
        
        # 固定初始位置和姿态进行测试
        start_pos = jp.array([0.0, 0.0, 0.05]) # TODO：确保方块能够放在地上，否则会刷奖励
        start_quat = jp.array([1.0, 0.0, 0.0, 0.0])
        q_cube = jp.array([*start_pos, *start_quat])
        v_cube = jp.zeros(6)
        
        # ten_len_xy=0.015
        # ctrl = jp.zeros((self.mjx_model.nu,))
        # ctrl = ctrl.at[1:5].set(ten_len_xy)
        
        # Set initial tendon lengths for thumb joints
        qpos = jp.concatenate([q_hand, q_cube])
        qvel = jp.concatenate([v_hand, v_cube])

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
            "last_act": jp.zeros(consts.NU),
            "last_last_act": jp.zeros(consts.NU),
            "motor_targets": self._default_ctrl,
            "dist": jp.zeros(()),
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
        metrics["reward/total"] = jp.zeros(())
    
        # TODO: change obs history size accordingly
        obs_history = jp.zeros(self._config.history_len * 75) # joint_pos = 26
        obs = self._get_obs(data, info, obs_history)
        reward, done = jp.zeros(2)

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
            motor_targets = self._default_ctrl + action * self._config.action_scale
            motor_targets = jp.clip(motor_targets, self._act_lowers, self._act_uppers)
            # motor_targets = motor_targets.at[16:22].set(self._default_ctrl[16:22])
            # debug info
            # jax.debug.print("step action shape: {}", action.shape)
            # jax.debug.print("motor target:{},action:{}", motor_targets, action)
            # jax.debug.print("motor targets: {}", motor_targets)

            data = mjx_env.step(
                self.mjx_model, state.data, motor_targets, self.n_substeps
            )
            state.info["motor_targets"] = motor_targets
            
            # get observations and done signal
            obs = self._get_obs(data, state.info, state.obs["state"])
            done = self._get_termination(data)

            # get rewards
            raw_rewards = self._get_reward(data, action, state.info, state.metrics, done)
            scaled_rewards = {
                k: v * self._config.reward_config.scales[k] for k, v in raw_rewards.items()
            } # scale rewards with config scales constatnts
            
            # update metrics
            state.info["last_last_act"] = state.info["last_act"]
            state.info["last_act"] = action
            state.info["dist"] = jp.linalg.norm(self.get_cube_position(data) - self.get_palm_position(data))
            # state.info["last_cube_angvel"] = self.get_cube_angvel(data)
            for k, v in scaled_rewards.items():
                state.metrics[f"reward/{k}"] = v

            reward = sum(scaled_rewards.values()) * self.dt
            # reward = sum(scaled_rewards.values())

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
        # fall_termination = self.get_obj_position(data)[2] < -0.2
        return nans

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any], obs_history: jax.Array
    ) -> mjx_env.Observation:
        joint_qpos = data.qpos[self._hand_qids]
        info["rng"], noise_rng = jax.random.split(info["rng"])

        # all these functions should be defined in the base class, and necessary sensors should be added to the xml
        cube_pos = self.get_cube_position(data)
        palm_pos = self.get_palm_position(data)
        cube_pos_error = palm_pos - cube_pos
        cube_quat = self.get_cube_orientation(data)
        cube_angvel = self.get_cube_angvel(data)
        cube_linvel = self.get_cube_linvel(data)
        # fingertip_positions = self.get_fingertip_positions(data)
        fingertip_errors = self.get_fingertip_errors(data, cube_pos)
        contact_forces = self.get_touch_forces(data)

        # 策略网络所能够观察到的状态，暂时增强
        # TODO: 我认为触觉信息也应该解包后放在这里
        state = jp.concatenate([
            joint_qpos,
            # noisy_joint_qpos,
            fingertip_errors,
            contact_forces,
            cube_pos,
            palm_pos,
            # cube_quat,
            info["last_act"],
            # *self.get_tactile_info(data),
        ])
        obs_history = jp.roll(obs_history, state.size)
        obs_history = obs_history.at[: state.size].set(state)
    
        # 供价值网络用于估算价值所需要的完整状态
        privileged_state = jp.concatenate([
            state,
            # joint_qpos,
            data.qvel[self._hand_dqids],
            fingertip_errors,
            cube_pos_error,
            cube_quat,
            # cube_angvel,
            # cube_linvel,
            contact_forces
        ])
    
        return {
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
        """
        计算抓取任务的奖励函数,奖励包括以下内容:
        - 物体位姿奖励,物体的位姿与目标点之间的距离,鼓励灵巧手靠近物体并抓取物体
        - 内部标记点距离奖励:内侧的标记点的 SDF 值之和,鼓励灵巧手的手指和手掌内侧靠近物体
        - 外部标记点距离:外侧的标记点的 SDF 值之和与内侧的差值,使物体保持在手指和手掌的内侧
        - 手指弯曲度:在手掌与物体距离小于阈值时,对手指的弯曲动作进行奖励,以抓取物体
        - 接触状态:每根手指尖端视触觉传感器检测到接触数据时施加奖励,由于抓取时大拇指与其他手指处于对侧,因此大拇指对抓握的作用更大,给予更大的奖励
        - 抬升高度:当大拇指以及其他至少一根手指上的视触觉传感器检测到接触数据时(此时大概率接近或已经抓住物体),对灵巧手手掌的抬升动作进行奖励,鼓励灵巧手抬升物体
        """
        del done, metrics  # Unused.
        
        # 1. 获取状态信息
        target_pos = self.get_target_position(data)
        obj_pos = self.get_cube_position(data)
        obj_quat = self.get_cube_orientation(data)
        obj_pose = jp.concatenate([obj_pos, obj_quat])
        palm_pos = self.get_palm_position(data)

        # 2. 计算各项奖励
        
        # [Reach Reward] 手掌靠近物体
        palm_dist = jp.linalg.norm(palm_pos - obj_pos)
        # reward_reach_palm = palm_dist
        reward_reach_palm = reward.tolerance(
            palm_dist, bounds=(0.0, 0.05), margin=0.2, sigmoid='linear', value_at_margin=0.0
        )
        # jax.debug.print("palm dist: {}, reach reward: {}", palm_dist, reward_reach_palm)

        # [Inner Dist Reward] 内侧点靠近物体表面
        # SDF 接近 0 (表面) 或略微负值 (内部) 时奖励高
        inner_sdfs = self.get_cube_sdf(obj_pose, self.get_inner_sites_positions(data))
        reward_inner_dist = jp.sum(reward.tolerance(
            inner_sdfs, bounds=(-0.02, 0.02), margin=0.05, sigmoid='linear', value_at_margin=0.0
        ))
        # jax.debug.print("inner sdf: {}, inner dist reward: {}", inner_sdfs, reward_inner_dist)

        # [Outer Dist Reward] 外侧点远离物体
        # SDF > 0.03 时奖励为 1
        outer_sdfs = self.get_cube_sdf(obj_pose, self.get_outer_sites_positions(data))
        reward_outer_dist = jp.sum(reward.tolerance(
            outer_sdfs, bounds=(-0.02, 0.02), margin=0.05, sigmoid='linear', value_at_margin=0.0
        ))
        # jax.debug.print("outer sdf: {}, outer dist reward: {}", outer_sdfs, reward_outer_dist)

        # [Finger Bend Reward] 手指弯曲
        # 鼓励特定关节弯曲 (假设 action[12:16] 为关键关节)
        # reward_finger_bend = jp.sum(jp.abs(action[8:16]))
        is_close = palm_dist < 0.2
        reward_finger_bend = jp.where(is_close, jp.sum(action[8:16]), 0.0)
        # jax.debug.print("finger bend reward: {}", reward_finger_bend)
        
        # [Finger Proximity Reward] 鼓励指尖靠近方块
        # 计算每个指尖到方块中心的距离，并在距离较小时给予更高奖励
        tips_pos = self.get_tips_positions(data)
        tip_dists = jp.linalg.norm(tips_pos - obj_pos, axis=1)
        reward_finger_prox = jp.sum(reward.tolerance(
            tip_dists, bounds=(0.0, 0.01), margin=0.2, sigmoid='linear', value_at_margin=0.0
        ))
        # jax.debug.print("tip dists: {}, finger prox reward: {}", tip_dists, reward_finger_prox)
        
        # [Contact Reward] 接触奖励
        # 加权计算接触情况，大拇指权重更高
        weights = jp.array([5.0, 1.0, 1.0, 1.0, 1.0])
        contact_info = self.get_contact_info(data, contact_threshold=0.1)
        reward_contact = jp.sum(weights * contact_info) / jp.sum(weights)
        # jax.debug.print("contact info: {}, contact reward: {}", contact_info, reward_contact)

        # [Lift Reward] 抬升奖励
        # 仅在形成抓取 (大拇指+其他手指接触) 时，奖励抬升高度
        lift_condition = (contact_info[0] > 0.5) & (jp.sum(contact_info[1:]) >= 0.5)
        reward_lift = reward.tolerance(
            palm_pos[2], bounds=(0.2, float('inf')), margin=0.2, sigmoid='linear', value_at_margin=0.0
        )
        reward_lift = jp.where(lift_condition, reward_lift, 0.0)
        # jax.debug.print("palm height: {}, lift reward: {}", palm_pos[2], reward_lift)

        # [Action Smoothness Reward] 动作平滑性惩罚
        reward_action_smooth = jp.sum(jp.square(action - info["last_act"]))
        # jax.debug.print("action smooth reward: {}", reward_action_smooth)

        return {
            "reach_palm": reward_reach_palm,
            "inner_dist": reward_inner_dist,
            "outer_dist": reward_outer_dist,
            "finger_prox": reward_finger_prox,
            "finger_bend": reward_finger_bend,
            "contact": reward_contact,
            "lift": reward_lift,
            "action_smooth": reward_action_smooth,
        }

    # Additional sensors specially for this task env
    # TODO: check the sensor names and add these sensors to the xml if not exists
    # def get_fingertip_positions(self, data:mjx.Data) -> jax.Array:
    #     """获取所有指尖的位置表示"""
    #     sensor_names = [
    #         "thumb_fingertip_pos",
    #         "index_fingertip_pos",
    #         "middle_fingertip_pos",
    #         "ring_fingertip_pos",
    #         "little_fingertip_pos",
    #     ]
    #     return jp.concatenate([
    #         mjx_env.get_sensor_data(self.mj_model, data, sensor_name)
    #         for sensor_name in sensor_names
    #     ])
    
    def get_fingertip_errors(self, data:mjx.Data, cube_pos: jax.Array) -> jax.Array:
        """获取所有指尖与方块位置的误差表示"""
        sensor_names = [
            "thumb_fingertip_pos",
            "index_fingertip_pos",
            "middle_fingertip_pos",
            "ring_fingertip_pos",
            "little_fingertip_pos",
        ]
        return jp.concatenate([
            mjx_env.get_sensor_data(self.mj_model, data, sensor_name) - cube_pos
            for sensor_name in sensor_names
        ])
    
    # Neccessary getters 
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
    
    def get_target_position(self, data:mjx.Data) -> jax.Array:
        """获取目标位置""" 
        target_pos = data.site_xpos[self._target_sid]
        return jp.array(target_pos)
    
    def get_palm_quat(self, data: mjx.Data) -> jax.Array:
        """获取手掌四元数表示的位姿"""
        return mjx_env.get_sensor_data(self.mj_model, data, "palm_quat")
    
    def get_tips_positions(self, data: mjx.Data) -> jax.Array:
        """获取所有指尖的位置表示"""
        # TODO：这里直接写成常量
        sensor_names = [
            "thumb_fingertip_pos",
            "index_fingertip_pos",
            "middle_fingertip_pos",
            "ring_fingertip_pos",
            "little_fingertip_pos",
        ]
        return jp.stack([
            mjx_env.get_sensor_data(self.mj_model, data, sensor_name)
            for sensor_name in sensor_names
        ])
    
    def get_inner_sites_positions(self, data: mjx.Data) -> jax.Array:
        """获取inner sites的位置"""
        return data.site_xpos[self._inner_sids]
    
    def get_outer_sites_positions(self, data: mjx.Data) -> jax.Array:
        """获取outer sites的位置"""
        return data.site_xpos[self._outer_sids]
    
    def get_ten_len(self, data: mjx.Data) -> jax.Array:
        """获取tendon长度"""
        return mjx_env.get_sensor_data(self.mj_model, data, "tendon_length")
    
    def get_cube_sdf(self, cube_pose: jax.Array, points: jax.Array) -> jax.Array:
        """
        Calculate signed distance field (SDF) from points to a cube.
        Args:
            cube_pose: Array of shape (7,) with position and quaternion [x,y,z,qx,qy,qz,qw]
            points: Array of shape (N,3) containing N point coordinates
        Returns:
            Array of shape (N,) with signed distances to cube surface
        """
        # Extract translation and rotation
        pos = cube_pose[:3]
        quat = cube_pose[3:]
        
        # Convert quaternion to rotation matrix using MuJoCo math
        rot_matrix = math.quat_to_mat(quat)

        # Transform points to local space
        local_points = jp.einsum('ij,nj->ni', rot_matrix.T, points - pos)

        # Calculate distances per axis
        d = jp.abs(local_points) - self.cube_half_size

        # Calculate final distance
        outside_distance = jp.linalg.norm(jp.maximum(d, 0.0), axis=1)
        inside_distance = jp.minimum(jp.max(d, axis=1), 0.0)

        return outside_distance + inside_distance 
    
    # 使用简化的触觉传感器获得标量力
    def get_touch_forces(self, data: mjx.Data) -> jax.Array:
        """返回指尖触觉传感器检测到接触的合力的大小"""
        return jp.stack([
            mjx_env.get_sensor_data(self.mj_model, data, sensor_name) 
            for sensor_name in consts.TOUCH_SENSOR_NAMES
        ]).ravel()

    def get_contact_info(self, data: mjx.Data, contact_threshold: float) -> jax.Array:
        """根据触觉力判断是否接触"""
        touch_forces = self.get_touch_forces(data)
        # 阈值判断：力大于 contact_threshold 视为接触
        return jp.where(touch_forces > contact_threshold, 1.0, 0.0)
    
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
        mj_model = ParaHandGrasp().mj_model
        cube_geom_id = mj_model.geom("cube").id
        cube_body_id = mj_model.body("cube").id
        hand_qids = mjx_env.get_qpos_ids(mj_model, consts.JOINT_NAMES)
        # TODO: hand_body_names 需要检查
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
