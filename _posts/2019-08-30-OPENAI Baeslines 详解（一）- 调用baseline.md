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

首先，run中需要建立一系列算法所用到的一系列超参数：

```python
env                            # gym的环境ID
env_type                       # 环境类型 一般不用 
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
log_                           # 保存训练数据路径
```



如果是用 命令行形式 则可以用github示范形式。实际上baselines 是利用了argparse 模块来解析了整个命令行中的表达式。所以如果需要用 pycharm等IDE 可以 写一个list 


``` python
python -m baselines.run --alg=ppo2 --env=Humanoid-v2 --network=mlp --num_timesteps=2e7 --ent_coef=0.1 --num_hidden=32 --num_layers=3 --value_network=copy
```

转化一下 就是

['run.py 所在的路径', '--alg=ppo2', '--env=Humanoid-v2', '--network=mlp', '--num_timesteps=2e7', '--ent_coef=0.1', '--num_hidden=32', '--num_layers=3', '--value_network=copy']



我们直接可以在python中建立一个类然后直接定义这些参数。










```python
def main(args):
    # 参数部分
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