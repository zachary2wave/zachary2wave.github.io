---
layout: post
title: "OPENAI Baeslines 详解（零）综述"
date: 2019-07-05
excerpt: "OPENAI是强化学习的有力推进者，领导者。Beseline算法是OPENAI 在github上开源的强化学习标准程序。Beseline 顾名思义 ，构建新想法更先进的研究基础；同时，也是衡量新的算法的基准。"
tags: [openai, baselines, code]
comments: true
---





OPENAI是强化学习的有力推进者，领导者。Beseline算法是OPENAI 在github上开源的强化学习标准程序。Beseline 顾名思义 ，构建新想法更先进的研究基础；同时，也是衡量新的算法的基准。

# OPENAI Baeslines 详解（零）综述



[TOC]



##  OPENAI baseline

OPENAI是强化学习的有力推进者，领导者。在目前OPENAI已经将强化学习运用到炉火纯青的地步，其利用强化学习所做的DOTA2  AI不断击败人类，而使其名声大燥。

beseline算法是OPENAI 在github上开源的强化学习标准程序。其目的是使得这些算法将使研究社区更容易复制，改进和识别新想法，并将创建良好的baseline来构建其他新的更先进的研究。同时，可以确认的是算法中的DQN等算法已经经过各种测试，是可行的程序。通常当新的算法提出时，也被当做baseline，从而对比新的算法性能。这也是baseline的来源。

Baseline 作为OPENAI的官方公布程序，可信是一个重要的因素。因为很多算法的细节，可能从文章当中无法精确地得到，从而导致写程序的时候会产生大量的疑问。当然其也可以用在别的领域从而解决更多新的问题。

所以对baseline 进行改进 是很有必要的。

## 主要架构

+ 主程序：run result—plotter等
+ common
+ 算法
  + A2C( asynchronous advantage actor-critic  )   https://arxiv.org/abs/1602.01783
A2C是Asynchronous Advantage Actor Critic（A3C）的同步，确定性变体，具有相同的性能，但是更快的速度。 
  + ACER（ Sample Efficient Actor-Critic with Experience Replay ）https://arxiv.org/abs/1611.01224
  + ACKTR https://arxiv.org/abs/1708.05144
  ACKTR是一种比TRPO和A2C更具样本效率的强化学习算法，并且每次更新仅需要比A2C略多的计算。Actor-Critic方法，信任域优化（trust region optimization）来实现更一致的改进，以及分布式 Kronecker 因子分解以提高样本效率和可扩展性。
  + DDPG https://arxiv.org/abs/1509.02971

  +  DQN（Deep Q  Network）
  + Generative Adversarial Imitation Learning (GAIL) https://arxiv.org/abs/1606.03476
  + HER(Hindsight Experience Replay) https://arxiv.org/abs/1707.01495
  + PPO（Proximal Policy Optimization ） https://arxiv.org/abs/1707.06347
  + TRPO（Trust Region Policy Optimization）  https://arxiv.org/abs/1502.05477

## 使用指南

官网已经给出了具体使用指南。

```python
python -m baselines.run --alg=<name of the algorithm> --env=<environment_id> [additional arguments]
```

这个就不用详细说了

## 主要程序

主程序中有大量的 为了支持cmd直接运行所须的参数程序，主要是各种参数设定等等。

关键部分有两个函数：main 和 train

main 函数主要是 包含主体的循环。

train  函数主要是 调用算法和环境。

每一个算法都一个learning的接口来对接主程序，这个也成为将来要调用算法的时候的主要借口。

### main

其中main是最主要的部分，主程序 main 是跟OPENAI的GYM统一的步调。   具有循环 step、done、reset 几个步骤。

### Train

Train 函数主要是调用算法到 model中，然后再调用算法到learning 这个函数中

env   是调用环境的 主要还是与OPENAI的GYM相对接。

## common 文件夹

### common 文件夹 简述
#### 1、models  生成神经网络

用于神经网络  主要是有生成mlp （全连接网络）、CNN、IMPALA-CNN（https://arxiv.org/abs/1802.01561）、LSTM、CNN—LSTM 、impala_cnn_lstm。

####  2、Distributions 分布函数

主要是将变量映射成变成分布函数

- Box 连续空间->DiagGaussianPdType （对角高斯概率分布）
- Discrete离散空间->SoftCategoricalPdType（软分类概率分布）
- MultiDiscrete连续空间->SoftMultiCategoricalPdType （多变量软分类概率分布）
- 多二值变量连续空间->BernoulliPdType （伯努利概率分布）

分为两个大类 PD 和 PdType，具体请参考MADDPG中的。

#### 3、OPEN  Message Passing Library  MPI_xxx

并行计算编程的统一框架，其中调用了mpi4py，具体参看https://zhuanlan.zhihu.com/p/25332041。

####  4、XXX_util 类函数  辅助函数

主要有 plot_util 、math_util、misc_util、mlp_util、cmd_util、console_util .





 



 



 



 