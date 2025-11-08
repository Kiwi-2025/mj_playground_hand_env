# README

### Train Files
- `reorient_RL.py` 简单版本的训练脚本
- `train_reorient.py` 复杂版本的训练，支持命令行修改各种参数，支持记录点导入继续训练

### 项目结构
- para_env 为环境相关的配置文件，其中的`xmls`文件夹存储了相关的手的建模
  - xmls 存放了mjcf文件
    - reorient 手重定向任务的模型文件 
  - config 存放了一些RL参数的配置文件
- tool 存放了一些有用的小工具，比如快速生成不同触觉传感器规模的 xml 文件
- data 存放了生成的一些数据
- sample 存放了一些测试代码以及之前的施工代码，可以作为使用示例参考
- figs 存放了一些仿真中绘制的曲线

### 安装相关依赖
相关依赖包已经在`requirements.yml`中给出，该`requirements.yml`使用conda工具导出的，安装的时候有几个注意事项
- 安装指令为 `conda env create -f requirements.yml`，然后通过`conda activate mjx_env`激活环境
- jax 的版本与 cuda 相对应问题，参考 https://jax.net.cn/en/latest/installation.html 即可
- playground 的安装，不建议PyPI安装playground，建议通过clone后放到某个位置然后通过`pip install -e .`将其设置为开发者模式，因为下面的方法需要篡改这个包
- mujoco_menagerie 的下载配置问题
  - 方案1：删除playground中检查mujoco_menagerie的函数，具体位置在 mjx_env.py 的 `ensure_menagerie_exists`函数中
  - 方案2：手动添加 mujoco_menagerie 到 `./mujoco_playground/mujoco_playground/external_dep/` 中,直接从远程clone这个仓库的时候会发现并没有这个文件夹，很正常，这个文件夹是在第一次你尝试导入的时候自动添加的，这时我们只需要 mkdir 指令生成这个文件夹然后进去clone即可

### 特性

- **xml的路径**： playground 中使用的 xml 的路径是相对于运行的 python 文件的，这与 mujoco 刚好相反， mujoco 是直接相对于 xml 文件本身的，这一点非常奇怪？（我怀疑我在某个地方指定了路径地址导致出错，但是我还没有检查出来，所以我现在把路径改到相对于 train 文件）
- **get_body_ids**: mujoco 中之前经常使用的这个函数似乎已经停止使用了，现在只能够通过直接访问 ids 的方式来访问了，比较粗暴简单