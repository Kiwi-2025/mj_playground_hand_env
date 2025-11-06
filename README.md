# README

### 项目结构
- para_env 为环境相关的配置文件，其中的`xmls`文件夹存储了相关的手的建模
  - xmls 存放了mjcf文件
  - config 存放了一些RL参数的配置文件
- tool 存放了一些有用的小工具
- data 存放了生成的一些数据
- sample 存放了一些测试代码以及之前的施工代码，可以作为使用示例参考

### 安装相关依赖
相关依赖包已经在`requirements.yml`中给出，该`requirements.yml`使用conda工具导出的，安装的时候有几个注意事项
- 安装指令为 `conda env create -f requirements.yml`，然后通过`conda activate mjx_env`激活环境
- jax 的版本与 cuda 相对应问题，参考 https://jax.net.cn/en/latest/installation.html 即可
- playground 的安装，不建议PyPI安装playground，建议通过clone后放到某个位置然后通过`pip -install -e .`将其设置为开发者模式，因为下面的方法需要篡改这个包
- mujoco_menagerie 的下载配置问题
  - 方案1：删除playground中检查mujoco_menagerie的函数，具体位置在 mjx_env.py 的 `ensure_menagerie_exists`函数中
  - 方案2：手动添加 mujoco_menagerie 到 `./mujoco_playground/mujoco_playground/external_dep/` 中
