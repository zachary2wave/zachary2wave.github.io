---
layout: post
title: "强化学习中使用OPENAI的GYM建立自己环境"
excerpt: "A bunch of text to test readability."
tags: 
  - sample post
  - readability
  - test
---
Reinforcement Learning 已经经过了几十年的发展，发展壮大。近些年来，跟随着机器学习的浪潮开始发展壮大。多次战胜围棋冠军柯洁，以及在DOTA2、星际争霸等游戏中超凡表现，成为了众人追捧的明星。目前OPENAI作为世界NO.1的AI研究机构，构建的GYM，成为衡量强化学习算法的标准工具。通过OPENAI 的GYM直接构建自己的环境，从而利用目前现有的算法，直接求解模型。

# 强化学习中使用OPENAI的GYM建立自己环境

## 综述
Reinforcement Learning 已经经过了几十年的发展，发展壮大。近些年来，跟随着机器学习的浪潮开始发展壮大。多次战胜围棋冠军柯洁，以及在DOTA2、星际争霸等游戏中超凡表现，成为了众人追捧的明星。目前OPENAI作为世界NO.1的AI研究机构，构建的GYM，成为衡量强化学习算法的标准工具。通过OPENAI 的GYM直接构建自己的环境，从而利用目前现有的算法，直接求解模型。

 __包含大量自我理解，肯定存在不正确的地方，希望大家指正__
## RL and GYM
RL 考虑的是agent如何在一个环境中采取行动，以最大化一些累积奖励。
其中主要包含的是2个交互：
1. agent对env作出动作 改变env
2. env 给出奖励和新的状态 给agent
其中GYM就是OPENAI所搭建的env。

具体的安装 和 介绍 主页很详细。
GYM主页 以及 DOC
[GYM](https://gym.openai.com/)
[GYM——DOC](https://gym.openai.com/docs/)

 安装好GYM之后，可以在annaconda 的 env 下的 环境名称 文件夹下 python sitpackage 下。

在调用GYM的环境的时候可以利用：

		'import gym'
		'env = gym.make('CartPole-v1')'

GYM的文件夹下 主要包含：
+ envs              所有环境都保存在这个文件下
+ spaces          环境所定义的状态、动作空间
+ utils               环境中使用的一组常用实用程序
+ warppers      包装
+ __init__          读取时初始化
+ core               核心环境，直接链接到给定的环境
GYM 创建的环境主要在envs中，在这个里面可以找到常用的几个环境，比如说cart-pole, MountainCar等等。
自我构建的GYM环境都应该在放在envs下子文件夹中的一个py文件中的类。
例如：
		`gym\envs\classic_control\cartpole.py`

## GYM registry
所有构建的环境都需要调用GYM库，然后再通过GYM库来调用所写的环境。所以需要现在GYM的内部构件一个内链接，指向自己构建的环境。
registry 主要在
1. envs下 \__init__ 文件下

		`register(`
    		`id='CartPole-v1',`
    		`entry_point='gym.envs.classic_control:CartPoleEnv',`
    		`max_episode_steps=500,`
    		`reward_threshold=475.0,`
		`)`
		
	 id 调用所构建的环境的名称 调用该环境的时候 所起的名字
	 注：名字包含一些特殊符号的时候，会报错
	 entry_point 所在的位置 
	 例如上述： 存在gym 文件夹下 classic_control文件夹下
	 算法所需的参数
	2 在所在文件夹下	
	建立 \__init__ 文件，在下面调用
	
		from gym.envs.classic_control.cartpole import CartPoleEnv
	
	其中是cartpole是环境所存在的文件名字，CartPoleEnv是该文件下的类。

 # GYM 环境构建
自我构建的环境为一个类。主要包含：变量、函数
## 必须的变量
这个类包含如下两个变量值：state 和 action 
对应的两个空间为observation _space 和 action _space
这两个空间必须要用 space 文件夹下的类在\__init__中进行定义。
其中 state是一个 object  一般为一个np.array  包含多个状态指示值。

## 必须存在的函数
+ step                  利用动作 环境给出的一下步动作 和 环境给出的奖励（核心）

	这个函数 承担了最重要的功能，是所构建环境所实现功能的位置
	 	输入为 动作  输出为 
 	1. 下一个状态值 object 
 	2. 反馈    float  值
 	3. done（终结标志） 布尔值   0 或者1 
 	4.  info（对调试有用的任何信息） any
+ reset				 重置环境
  将状态设置为初始状态，返回： 状态值
+ render              在图形界面上作出反应
	可以没有，但是必须存在
+ close                关闭图形界面
+ seed                 随机种子
	可以没有，但是必须存在
## 状态、动作空间的构建
主要分为离散空间和连续空间：
连续空间主要由spaces.Box定义，例如：

	self.action_space = spaces.Box(low=-10, high=10, shape=(1,2))

上面定义了一个取值范围在（-10，10）的变量 维度为1，2

离散空间主要有
+ spaces.Discrete，例如

		self.observation_space = spaces.Discrete(2)

	上面定义了一个变量空间范围为[0,2) 之间的整数
+ spaces.MultiBinary， 例如

 		self.observation_space = spaces.MultiBinary(2)

	上面定义了一个变量空间为0，1的2维整数变量

+ spaces.MultiBinary， 例如
		
		self.observation_space = MultiDiscrete（）

其他还可以定义一个元组或者字典 等变量空间。