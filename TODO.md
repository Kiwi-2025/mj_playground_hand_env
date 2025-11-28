# TODO List
## Issues to investigate and resolve

### 触觉传感器
- 按照原先触觉力提取器，要求长方体的传感器单元才可以，现在换成了球形，可能会出现影响
- 混合网络结构，需要一个CNN-MLP的混合网络结构

## Completed
- 多卡训练，目前看是没有生效的，需要修理
    - 你一段时间不管他好像自己就会吃满几个设备
    - 使用 CUDA_VISIBLE_DEVICES 环境变量指定 GPU 设备
    - 使用 `jax.devices()` 查看可用设备
- 理清楚obs,privileged_obs,和奖励计算之间的关系
    - 你需要了解 PPO 算法中如何使用观测值（obs）和特权观测值（privileged_obs）来计算奖励和更新策略。
- TensorBoard 转发问题
    - 使用`train_reorient.py --use_tb`启动训练脚本
    - 使用`tensorboard --logdir logs`启动TensorBoard服务
- TensorBoard 中出现了过多的训练指标，出现了不知道原因的eval环境标签
- 调整PPO超参数互相冲突问题，现在``num_minibatches``和``batch_size``互相冲突，导致无法同时设置较小的batch size和较多的minibatches。
batch_size=128 与 num_envs=4, unroll_length=5 不匹配。Brax 期望 batch_size = (num_envs * unroll_length)，或按 minibatch 划分后维度一致。
你的设定：num_envs * unroll_length = 4*5=20 << 128，Brax 在构造归一化统计期望“每个 batch 有 25 条并行轨迹”（128/5=25），但实际只有 4.

