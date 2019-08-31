---
layout: post
title: "OPENAI Baeslines 详解（一）开始学习"
date: 2019-07-05
excerpt: "直接上手baseline，来训练自己的环境。"
tags: [openai, baselines, code]
comments: true
---


# OPENAI Baeslines 详解（一）开始学习

## baselines

OPENAI是强化学习的有力推进者，领导者。Beseline算法是OPENAI 在github上开源的强化学习标准程序。Beseline 顾名思义 ，构建新想法更先进的研究基础；同时，也是衡量新的算法的基准。

## 怎么去用baselines学习自己的环境

github上给出了固定环境的用法。

```python
python -m baselines.run --alg=ppo2 --env=Humanoid-v2 --network=mlp --num_timesteps=2e7 --ent_coef=0.1 --num_hidden=32 --num_layers=3 --value_network=copy
python -m baselines.run --alg=deepq --env=PongNoFrameskip-v4 --num_timesteps=1e6
python -m baselines.run --alg=<name of the algorithm> --env=<environment_id> [additional arguments]
```

如果说只想看看结果，那么这些一定是够用的。但是如果想进一步挖掘算法，然后训练自己的环境，那么这些肯定远远不能满足我们当前的需求。



但是可以看出baselines 都是使用 文件夹下的run函数 调用不同文件夹下的算法，来实现学习的。

run中需要建立一系列算法所用到的一系列超参数：

```python
env                            # gym的环境ID
env_type                       # 环境类型 一般默认为None。
seed                           # 随机种子
alg                            # 所调用的算法 默认为'ppo2'
num_timesteps                  # 总共step，默认为1e6
network                        # 网络类型 可选项mlp（全连接）, cnn, lstm, cnn_lstm, conv_only
state                          # 游戏状态 一般在mujoco等环境中使用
num_env                        # 并行运行的环境数目 一般最大是CPU的核心数目
reward_scale                   # 将Reward规划在某个范围，默认为1
save_path                      # 训练的网络所保存的路径
save_video_interval            # 所玩游戏视频每隔多少步保存一次
save_video_length              # 保存视频的长度
log_path                          # 保存训练数据路径
```



如果是用 命令行形式 则可以用github示范形式。实际上baselines 是利用了argparse 模块来解析了整个命令行中的表达式，所以如果需要用 pycharm等IDE 可以 写一个list 用来调用。例如 如果命令是：


``` python
python -m baselines.run --alg=ppo2 --env=Humanoid-v2 --network=mlp --num_timesteps=2e7 --ent_coef=0.1 --num_hidden=32 --num_layers=3 --value_network=copy
```

转化一下 可以变化为：

``` python
['run.py 所在的路径', '--alg=ppo2', '--env=Humanoid-v2', '--network=mlp', '--num_timesteps=2e7', '--ent_coef=0.1', '--num_hidden=32', '--num_layers=3', '--value_network=copy']
```

这样 直接把自己的环境ID一替换就好了。



同样我们可以不需要调用argparser，当然也没有默认帮你补充参数为默认参数，所以如果必须参数缺失，就会报错。我们直接可以在python中建立2个类然后直接定义这些参数。其中一个是arg 是训练的时候一些参数不可缺少的参数已经如下：


```python
class arg:
    def __init__(self):
        self.env = 'CartPole-v0'    
        self.env_type = None 
        self.alg = 'ppo2'                           
        self.num_timesteps = 1e6    
        self.network = 'mlp'   
        self.num_env = 0
        self.reward_scale = 1.0
        self.save_path =None
        self.log_path = None
```

其中第二个是 算法所利用的参数extra_args:

```python
class extra_args:
    def __init__(self):
        self.ent_coef = 0.1                   
        self.num_layers = 3                          
        self.num_hidden = 32    
        self.value_network = copy   
```

注意利用run的时候 ，最终调用的可以是类，而不是类的地址，所以需要利用args=arg()  将类生成。

#### run的主程序 MAIN

```python
def main(args):
    # argparser参数部分  如果利用类的方式 可以将这里注释掉
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    # 调用模型和环境
    model, env = train(args, extra_args)
    if args.play:
        logger.log("Running trained model")
        # 环境重置
        obs = env.reset()
        episode_rew = 0
        # GYM式循环不断训练
        while True:
            #  下一步
            obs, rew, done, _ = env.step(actions)
            episode_rew += rew[0] if isinstance(env, VecEnv) else rew
            env.render()
            done = done.any() if isinstance(done, np.ndarray) else done
            if done:
                print('episode_rew={}'.format(episode_rew))
                episode_rew = 0
                obs = env.reset()
    env.close()
    return model
```



#### Train

Train 函数主要是调用算法到 model中，然后再调用算法到learning 这个函数中

env   是调用环境的 主要还是与OPENAI的GYM相对接。

```python
def train(args, extra_args):
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))

    learn = get_learn_function(args.alg)

    env = build_env(args)
    # 选择网络 
    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)
    # model调用函数的model      
    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env
```

使用自己的网络



```
from baselines.common.models import register@register("your_network_name")def your_network_define(**net_kwargs):    ...    return network_fn
```





## 更简单的方式调用

Baselines 中包含很多东西是atari、mujoco、retro等，如果我们利用自己的环境，那么大可不必那么复杂的代码。

这里给出一个精简了的代码，供大家使用。 可以在我的github中下载使用。

```python
import os.path as osp
import gym
import tensorflow as tf
import numpy as np
import datetime
from baselines.common.vec_env import VecEnv
from baselines.common.cmd_util import make_vec_env
from baselines.common.tf_util import get_session
from baselines import logger
import run_util
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

# tensorflow 资源调用。
config = tf.ConfigProto(allow_soft_placement=True,  # 自行选择设备
                        intra_op_parallelism_threads=1,   # intra_op_parallelism_threads 单个运算内部，参数并行计算
                        inter_op_parallelism_threads=1)   # inter_op_parallelism_threads 多个运算之间，参数并行计算
# 备选项目device_count={"CPU": 4},
# config.gpu_options.per_process_gpu_memory_fraction = 0.4  #占用40%显存
# config.gpu_options.allow_growth = True                    # 动态分配显存
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'                  # GPU 设备选定
get_session(config=config)



# 环境参数设定部分
envID = 'CartPole-v0'
alg ='ppo2'
parallel = True
reward_scale = 1
seed = None
total_timesteps = 1e6
save_path = envID+'/'
log_path  = envID+'/'+datetime.datetime.now()+'/'
logger.configure(dir=log_path)

# 创建环境
env_type, env_id = run_util.get_env_type(envID)

if parallel:
    num_env = 1
    env = make_vec_env(env_id, env_type, num_env or 1, seed=seed, reward_scale=reward_scale)
else:
    env = gym.make(env_id)

print('Training {} on {}:{} with arguments \n{}'.format(alg, env_type, env_id, alg_kwargs))

# 创建agent

alg_kwargs= {'ent_coef':0.1,
             'num_hidden':32,
             'num_layers':3,
             'value_network': 'copy',
             'network': 'mlp'}                 # 这个地方可以直接写在  learn 中
learn = run_util.get_learn_function(alg)
model = learn(
    env=env,
    seed=seed,
    total_timesteps=total_timesteps,
    **alg_kwargs
)

# MPI part
rank = MPI.COMM_WORLD.Get_rank()


# 模型保存
if save_path is not None and rank == 0:
    save_path = osp.expanduser(save_path)
    model.save(save_path)


# 开始训练
logger.log("Running trained model")
obs = env.reset()
state = model.initial_state if hasattr(model, 'initial_state') else None
dones = np.zeros((1,))
episode_rew = 0
while True:
    if state is not None:
        actions, _, state, _ = model.step(obs, S=state, M=dones)
    else:
        actions, _, _, _ = model.step(obs)

    obs, rew, done, _ = env.step(actions)
    episode_rew += rew[0] if isinstance(env, VecEnv) else rew
    env.render()
    done = done.any() if isinstance(done, np.ndarray) else done
    if done:
        print('episode_rew={}'.format(episode_rew))
        episode_rew = 0
        obs = env.reset()

```



这个地方要用一个辅助代码：



```python
import gym
from collections import defaultdict
import re
from importlib import import_module


_game_envs = defaultdict(set)

def get_env_type( env_id ):
    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn
```





