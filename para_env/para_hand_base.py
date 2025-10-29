from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from para_env import para_hand_constants as consts

def get_assets() -> Dict[str, bytes]:
  assets = {}
  path = consts.ROOT_PATH
  mjx_env.update_assets(assets, path / "assets")
  mjx_env.update_assets(assets, path / "xmls", "*.xml")
  mjx_env.update_assets(assets, path / "xmls" / "objs", "*.obj")
  # mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "meshes")
  return assets

class ParaHandEnv(mjx_env.MjxEnv):
  
    def __init__(
        self,
        xml_path: str,
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)
        self._model_assets = get_assets()
        self._mj_model = mujoco.MjModel.from_xml_string(
            epath.Path(xml_path).read_text(), assets=self._model_assets
        )
        self._mj_model.opt.timestep = self._config.sim_dt

        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160

        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = xml_path

    # Sensors reading.
    def get_finger_quat(self, data:mjx.Data, finger_name: str) -> jax.Array:
        """获取指定手指所有links的四元数表示"""
        sensor_names = [
            f"{finger_name}_base_link_quat",
            f"{finger_name}_link_1_quat",
            f"{finger_name}_link_2_quat",
            f"{finger_name}_link_3_quat",
        ]
        return jp.concatenate([
            mjx_env.get_sensor_data(self.mj_model, data, sensor_name)
            for sensor_name in sensor_names
        ])

    def get_finger_pos(self, data:mjx.Data, finger_name: str) -> jax.Array:
        """获取指定手指所有links的位置表示"""
        sensor_names = [
            f"{finger_name}_base_link_frame_origin_pos",
            f"{finger_name}_link_1_frame_origin_pos",
            f"{finger_name}_link_2_frame_origin_pos",
            f"{finger_name}_link_3_frame_origin_pos",
        ]
        return jp.concatenate([
            mjx_env.get_sensor_data(self.mj_model, data, sensor_name)
            for sensor_name in sensor_names
        ])
    
    def get_tendon_length(self, data:mjx.Data, tendon_name: str) -> jax.Array:
        """获取指定肌腱的长度"""
        sensor_name = f"{tendon_name}_length"
        return mjx_env.get_sensor_data(self.mj_model, data, sensor_name)
    
    # Accessors: used to save files.
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

