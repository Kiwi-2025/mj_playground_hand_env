# README

### Train Files
- `reorient_RL.py` 简单版本的训练脚本
- `train_reorient.py` 复杂版本的训练，支持命令行修改各种参数，支持记录点导入继续训练

### 项目结构
- para_env 为环境相关的配置文件，其中的`xmls`文件夹存储了相关的手的建模
  - xmls 存放了mjcf文件
    - reorient 手重定向任务的模型文件 
  - config 存放了一些RL参数的配置文件
- tool 存放了一些有用的小工具
  - `xml_generate.py` 快速生成不同触觉传感器规模的 xml 文件，稍微改进可用于para过程
  - `DOF_calc.py` 用于校准xml文件的DOF数量，你只需要设计机器人，将复杂的DOF计算问题交给机器
- data 存放了生成的一些数据
- sample 存放了一些测试代码以及之前的施工代码，可以作为使用示例参考
- figs 存放了一些仿真中绘制的曲线

### 快捷安装相关依赖
相关依赖包已经在`requirements.yml`中给出，该`requirements.yml`使用conda工具导出的，安装的时候有几个注意事项
- 安装指令为 `conda env create -f requirements.yml`，然后通过`conda activate mjx_env`激活环境
- jax 的版本与 cuda 相对应问题，参考 https://jax.net.cn/en/latest/installation.html 即可
- playground 的安装，不建议PyPI安装playground，建议通过clone后放到某个位置然后通过`pip install -e ".[all]"`将其设置为开发者模式，因为下面的方法需要篡改这个包
- mujoco_menagerie 的下载配置问题
  - 方案1：删除playground中检查mujoco_menagerie的函数，具体位置在 mjx_env.py 的 `ensure_menagerie_exists`函数中
  - 方案2：手动添加 mujoco_menagerie 到 `./mujoco_playground/mujoco_playground/external_dep/` 中,直接从远程clone这个仓库的时候会发现并没有这个文件夹，很正常，这个文件夹是在第一次你尝试导入的时候自动添加的，这时我们只需要 mkdir 指令生成这个文件夹然后进去clone即可

### 从零安装相关依赖
如果您对上面我提到的安装步骤感到担忧，或者您希望适配自己的电脑，可以按照下面的步骤进行。徒手安装可以参考 [https://zhuanlan.zhihu.com/p/1913023899534354252] 但是不需要安装低版本jax而是使用3.12+的Python解释器
- 使用conda创建环境 `conda create -n <envName> python==3.12`
- 安装jax及相关 `pip install -U "jax[cuda12]"` 如果你需要安装基于cuda13的版本，`pip install -U "jax[cuda13]"`
- 手动安装mujoco_playground，参考上面关于playground的技巧
- 按道理到这里就能够完成安装


### 特性
- **xml的路径**： playground 中使用的 xml 的路径是相对于运行的 python 文件的，这与 mujoco 刚好相反， mujoco 是直接相对于 xml 文件本身的，这一点非常奇怪？（我怀疑我在某个地方指定了路径地址导致出错，但是我还没有检查出来，所以我现在把路径改到相对于 train 文件）
- **get_body_ids**: mujoco 中之前经常使用的这个函数似乎已经停止使用了，现在只能够通过直接访问 ids 的方式来访问了，比较粗暴简单

### 训练脚本参数说明
运行方式：
```bash
python train_reorient.py [--参数=值 ...]
```

tensorboard 可视化：
```bash
tensorboard --logdir logs/ParaHandReorient-*
```

#### 环境相关
- --env_name (str, 默认 ParaHandReorient)  
  选择具体环境，必须在 para_env.ALL_ENVS 中。
- --impl (jax|warp, 默认 jax)  
  选择 MJX 实现后端。
- --vision (bool, 默认 False)  
  使用视觉输入，包装环境为图像批渲染模式。
- --domain_randomization (bool, 默认 False)  
  启用域随机化（通过 para_env.get_domain_randomizer）。

#### 运行模式
- --play_only (bool, 默认 False)  
  仅推理/评估，不训练（强制 num_timesteps = 0）。
- --load_checkpoint_path (str|path)  
  恢复已有模型；传目录时自动取最后编号子目录。
- --suffix (str)  
  实验名后缀，便于区分日志。

#### 训练步数与评估
- --num_timesteps (int, 默认 1000000)  
  总训练交互步数（环境步，不含 action_repeat 展开）。
- --num_evals (int, 默认 5)  
  训练过程中评估次数。
- --run_evals (bool, 默认 True)  
  是否在训练中周期性评估。
- --num_videos (int, 默认 1)  
  训练结束生成的评估视频数量。
- --rscope_envs (int|None)  
  启用 rscope 可视化并行采样的环境数量。
- --deterministic_rscope (bool, 默认 True)  
  rscope rollout 是否使用确定性策略。

#### 强化学习超参数
- --reward_scaling (float, 默认 0.1)  
  奖励乘法缩放，调节梯度量级。
- --episode_length (int, 默认 1000)  
  每个环境单独 episode 截断步数。
- --normalize_observations (bool, 默认 True)  
  运行中统计均值方差对观察归一化。
- --action_repeat (int, 默认 1)  
  每个 action 在底层环境重复执行次数。
- --unroll_length (int, 默认 10)  
  每次采样的时间展开长度（GAE、优势计算窗口）。
- --num_envs (int, 默认 1024)  
  训练并行环境数量（影响吞吐与显存）。
- --num_eval_envs (int, 默认 128)  
  评估时的并行环境数量（vision 模式被强制为 num_envs）。
- --discounting (float, 默认 0.97)  
  奖励折扣因子 γ。
- --entropy_cost (float, 默认 5e-3)  
  策略熵正则系数，调节探索。
- --learning_rate (float, 默认 5e-4)  
  优化器学习率。
- --batch_size (int, 默认 256)  
  每次更新的样本批大小（来自多个 env * unroll 拼接）。
- --num_minibatches (int, 默认 8)  
  一个采样批再切分的 mini-batch 数。
- --num_updates_per_batch (int, 默认 8)  
  每个采样批上重复 PPO 更新次数（epochs）。
- --clipping_epsilon (float, 默认 0.2)  
  PPO ratio clip 边界。
- --max_grad_norm (float, 默认 1.0)  
  梯度裁剪 L2 范数上限。

#### 网络结构与观察键
- --policy_hidden_layer_sizes (list[int], 默认 64,64,64)  
  策略 MLP 隐层尺寸。
- --value_hidden_layer_sizes (list[int], 默认 64,64,64)  
  价值函数 MLP 隐层尺寸。
- --policy_obs_key (str, 默认 state)  
  从环境 obs 字典中取该键作为策略输入。
- --value_obs_key (str, 默认 state)  
  价值网络使用的 obs 键。

#### 日志与监控
- --use_wandb (bool, 默认 False)  
  发送指标到 Weights & Biases。
- --use_tb (bool, 默认 False)  
  记录到 TensorBoard。
- --log_training_metrics (bool, 默认 False)  
  额外记录训练内的 per-episode 指标（频繁开启会慢）。
- --training_metrics_steps (int, 默认 1000000)  
  训练指标打印/回调间隔步数。
- --seed (int, 默认 1)  
  随机种子。

#### 其它说明
- 输出目录：logs/<env_name>-<timestamp>-<suffix>/  
  子目录 checkpoints/ 存放模型与 config.json。
- 视觉模式下 env_cfg.vision_config.render_batch_size = num_envs。
- render_every 固定为 2，可在脚本下方手动调节加快视频生成。
- 失败排查：显存溢出优先降低 num_envs 或 unroll_length；学习不稳定调低 learning_rate 或缩小 clipping_epsilon。

#### 示例
```bash
# 基础训练
python train_reorient.py --env_name=ParaHandReorient

# 使用较少并行和更短 episode
python train_reorient.py --env_name=ParaHandReorient --num_envs=128 --episode_length=400 --num_timesteps=300000

# 加载并仅评估
python train_reorient.py --play_only --load_checkpoint_path=logs/ParaHandReorient-20250101-120000/checkpoints/5 --num_videos=2

# 开启视觉 + 域随机化
python train_reorient.py --vision --domain_randomization --num_envs=256
```

#### 获取可用环境列表
```python
import para_env
print(para_env.ALL_ENVS)
```

#### 修改与扩展
新增参数：在文件顶部追加 flags.DEFINE_*，然后在 main 中读取并写入 ppo_params 或 env_cfg。