from mujoco_playground import registry
from mujoco_playground import wrapper

env_name = 'PandaPickCubeOrientation'
env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)

print("环境配置：", env_cfg)
print("您的mujoco_menageri配置成功，路径是：", wrapper.MENAGERIE_PATH)