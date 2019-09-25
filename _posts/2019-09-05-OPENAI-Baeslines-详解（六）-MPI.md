---
layout: post
title: "OPENAI Baeslines 详解（六）并行环境采样"
date: 2019-07-05
excerpt: "环境交互这么慢，那我多个环境一起采！"
tags: [openai, baselines, code]
comments: true
---

# OPENAI Baeslines 详解（六）并行环境采样

OPENAI 采用了 自带的multiprocessing模块 和 MPI4PY 库，进行多个环境建模，然后同时对多个环境进行采样，这样减少了环境探索时采样的时常。



## multiprocessing 和 MPI 介绍

 在python中运用多个CPU的核心，是利用[`multiprocessing`](https://docs.python.org/zh-cn/3/library/multiprocessing.html?highlight=multi#module-multiprocessing) 模块来完成的，是Python 自带的模块，想学习的可以看一下的几个官方文档。

https://docs.python.org/zh-cn/3/library/multiprocessing.html?highlight=multi#module-multiprocessing

MPI的全称是Message Passing Interface

先上一份官方Tutorial       https://mpi4py.readthedocs.io/en/stable/tutorial.html   

那么再来一个简单版的     https://zhuanlan.zhihu.com/p/25332041



## OPENAI 的Multi-env

Multi-Env 主要是分为两个部分

+ 第一个是对环境进行包装，可以对一组动作进行step

+ 第二个是对agent 进行包装，可以同时利用一个policy 对多个env的observation输出action，这个地方比较简单，因为对于tensorflow， 只是 送进去一个batch 还是 单送进去一组输入的区别。

对环境进行包装需要通过这个形式。 

```
env = make_vec_env(env_id, env_type, num_env or 1, seed=seed, reward_scale=reward_scale)
```

在对环境包装的时候，其类似于环境进行一次外包装。这次外包装后，新的类具有原来类的所有方法，只不过新类的方法在调用过去类方法的同时，还增加了一些新的方法。

当然也可以认为，我们通过gym.make(env_id)创建的类， 是这个env的父类， 这个子类继承了所有方法（当然这个是不准确的）

**注意输入的环境是增加了Monitor。**

通过这种方式建立的env 是一个SubprocVecEnv 在子程序中并行运行多个环境并通过pipes与它们通信。

新的环境通过 remote.send（） 将数据送入 并通过 remote.reciever() 将三个环境反馈的数据送回来。

以上这些部分没有什么难度的地方。重点是如何解决以下几个问题。

## 重点问题：

1. 多环境并行的时候，没有办法使得所有环境 分开reset。也就是说 所有的环境 只能是交互相同的次数，无法使得每个环境交互达到done的时候reset，即使达到done的时候，还是要继续交互step一直到所有的 指定的步数结束。

   有些环境可以不用reset比如竖立摆那个环境，但是有环境是必须要用的。

2. 环境被包装之后在管道的另一头，无法直接通过env的方式调用。无法获得env的信息。

## 解决办法：

### 针对第一个问题：

 1. 当环境达到done的时候，无法reset，但是这样的结果只能是所采样的数据是无效的，那么此时只要阻止其进入经验池进行训练就好了 。以ddpg为例，在ddpg.py中

    ```python
    flag=[1,1,1,1]
    for d in range(env.num_envs): # 157行 
        if done[d]:
        		flag[d]=0
    
    for i in num_env：            # 140行
    	if flag[i]:
    		agent.store_transition(obs[i], action[i], r[i], new_obs[i], done[i])
    ```
然后，当所有的flag都为零的时候reset 就好了。

 2. 方法一是较为简单的方式，但是依然会有浪费的采样，那么更好的方式是什么呢。直接异步reset就OK了。

    ```
    if done[d]:
    	self.remotes[d].send(('reset', None))
    ```

    但这种方式没办法reset  agent中的附加噪声。那么也就是说会产生一种情况就是在某几个回合开始的时候开始的时候噪声非常的小。

### 针对第二个问题：

​	GYM的环境step输出有4个值， newob，reward，done，info  可以直接把所要提取的数据写在info中。Info是个字典，可以直接调用。


​    

​    

















