### requirements
相关依赖包已经在`requirements.txt`中给出，安装的时候有几个注意事项
- jax 的版本与 cuda 相对应问题，参考 https://jax.net.cn/en/latest/installation.html 即可
- playground 的安装，可以参考 https://zhuanlan.zhihu.com/p/20347868805， 不建议PyPI安装
- mujoco_menagerie 的下载配置问题
  - 方案1：删除playground中检查mujoco_menagerie的函数，具体位置在 mjx_env.py 的 ensure_menagerie_exists 函数中
  - 方案2：手动添加 mujoco_menagerie 到 `./mujoco_playground/mujoco_playground/external_dep/` 中
