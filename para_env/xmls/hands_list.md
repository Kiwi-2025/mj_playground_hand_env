## Hand DOF
- 标准的手的自由度数是：
    NQ_POS = 25     # Number of Positions, 机器人系统的位置自由度数量
    NQ_VEL = 25     # Number of Velocities, 机器人系统的速度自由度数量
    NV = 25         # Number of Velocities, 机器人系统的速度自由度数量，表示机器人有16个关节
    NU = 18         # Number of Actuators, 机器人系统的执行器数量，表示机器人有16个执行器，在这里是说有16个motors
    NT_FORCE=5      # Number of Tactile Forces, 触觉传感器的数量，这里表示有5个触觉传感器，分别对应5根手指
- 一些道具的自由度数
    - cube   
        - NQ_POS: 7
        - NQ_VEL: 6
        - NV: 1
        - NU: 0


## Hands List

- hand.xml : 最早的手模型，有触觉传感器，不建议更改

### 测试OOM模型
- obj_hand_no_tac.xml : 注释了触觉传感器的部分，使用obj文件导入的手模型
- obj_hand_tac.xml : 含有触觉传感器的手模型，使用obj文件导入
- mesh_hand_no_tac.xml : 注释了触觉传感器的部分，使用mesh格式xml文件导入的手模型
- mesh_hand_tac.xml ： 含有触觉传感器的手模型，使用mesh格式xml文件导入

## 球状触觉传感器模型
- sphere_tac_5x5.xml ：含有触觉传感器的手模型，使用球体作为触觉传感器表示,5x5的传感器阵列
- sphere_tac_9x9.xml ：含有触觉传感器的手模型，使用球体作为触觉传感器表示,9x9的传感器阵列

## 强化学习环境
### Reorient任务
- reorient_hand.xml : 用于reorient任务的手模型,增加了各种传感器
### RotateZ任务
- rotatez_hand.xml : 用于rotatez任务的手模型,增加了各种传感器
- rotatez_hand_obj : 用obj导入了几个零件的外表面，没有触觉传感器，加载速度比较快