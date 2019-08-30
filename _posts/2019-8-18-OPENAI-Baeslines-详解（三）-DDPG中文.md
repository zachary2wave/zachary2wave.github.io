---
layout: post
title:   " OPENAI-Baeslines-详解（三）-DDPG中文"
date:   2019-03-19
excerpt: "DDPG 深度确定性策略梯度下降算法。[论文链接](https://arxiv.org/abs/1509.02971)。采用了Actor-Critic 架构，可以有效的处理连续域的问题。同时，其actor的确定性动作输出，提高了采样的有效性。"
tag:
- GYM
- reinforcement learning 
comments: False
---

DDPG 深度确定性策略梯度下降算法。[论文链接](https://arxiv.org/abs/1509.02971)。采用了Actor-Critic 架构，可以有效的处理连续域的问题。

同时，其actor的确定性动作输出，提高了采样的有效性。

## Actor-Critic and DPG

强化学习算法的主要目标是去学习一个策略，来指导agent与环境交互从而得到更好的收益。策略$\pi_{\theta}(a|s)$是以$\theta$为参数的概率分布，代表不同状态下所采用的动作的概率分布。在学习的过程中不断的改变该函数的参数 $\theta$，从而改变应对环境的策略，以得到更好的奖励。当策略固定时，其所遍历的状态动作概率可以表示为


$$
p_\theta \left( {{{\bf{s}}_1},{{\bf{a}}_1}, \ldots ,{{\bf{s}}_T},{{\bf{a}}_T}} \right)_{p_\theta(\tau )} = p\left( {{{\bf{s}}_1}} \right)\prod\limits_{t = 1}^T {{\pi _\theta }} \left( {{{\bf{a}}_t}|{{\bf{s}}_t}} \right)p\left( {{{\bf{s}}_{t + 1}}|{{\bf{s}}_t},{{\bf{a}}_t}} \right)
$$
对单个状态而言，其到达概率为：

$$
\rho^{\pi}(s')=\int_{\mathcal{S}} \sum_{t=1}^{\infty} \gamma^{t-1} p_{1}(s) p\left(s \rightarrow s^{\prime}, t, \pi\right) \mathrm{d} s
$$

那么在策略$\pi_{\theta}(a|s)$下得到的期望收益可以表示为：

$$
\begin{aligned} J\left(\pi_{\theta}\right) &=\int_{\mathcal{S}} \rho^{\pi}(s) \int_{\mathcal{A}} \pi_{\theta}(s, a) r(s, a) \mathrm{d} a \mathrm{d} s \\ &=\mathbb{E}_{s \sim \rho^{\pi}, a \sim \pi_{\theta}}[r(s, a)] \end{aligned}
$$

实际上 DDPG是DPG算法利用深度神经网络去逼进 策略$\pi_{\theta}(a|s)$和期望$Q$。$Q$函数的更新 需要与DQN类似：

$$
Q^{*}(s, a)=Q(s, a)+\alpha\left(r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime}\right)-Q(s, a)\right)
$$

所以$Q$函数更新的loss可以表示为：
$$
L(\theta)=E\left[\left(r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime} ; \theta\right)-Q(s, a ; \theta)\right)^{2}\right]
$$
这样我们需要2组神经网络，其中一组 用来生成现在的状态S和动作A 另一组 用于生成 未来$Q$函数估值  $Q\left(s^{\prime}, a^{\prime} ; \theta\right)$ 一组用于更新当前$Q(s, a ; \theta)$网络。  

![1564451525193](image\1564451525193.png)



## Baseline 中的DDPG

DDPG文件夹下包含以下5个文件：

+ ddpg 主要程序  主要是 runner  
+ ddpg—learner  DDPG算法核心  主要是生成agent
+ memory 记忆库  
+ models   神经网络
+ noise      增加噪声

#### DDPG 主程序

#####  初始化 

 建立网络 63~65行 

```
memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape) # 创建记忆库
critic = Critic(network=network, **network_kwargs) # critic 网络
actor = Actor(nb_actions, network=network, **network_kwargs) # actor 网络
```

67~84行  创建noise模型 ，noise 主要作用是用于增大探索 

89行 调用ddpg—learner 创建agent    并开始循环与环境交互。

这里可以同时对多个环境 进行探索。

每个循环 有 epoch 、cycle、

每个epoch  需要有多个cycle  每个 cycle  中  rollout_step 次与环境交互  train_step 次进行训练。  

```python

for epoch in range(nb_epochs):   
    for cycle in range(nb_epoch_cycles):
```
**与环境交换阶段**
```python
        # reset环境
        if nenvs > 1:     
            agent.reset()
        for t_rollout in range(nb_rollout_steps):   
            # 输出动作
            action, q, _, _ = agent.step(obs, apply_noise=True, compute_Q=True)
         	  # 动作都是归一化在-1到1之间
            new_obs, r, done, info = env.step(max_action * action)  

            t += 1
            if rank == 0 and render:
                env.render()
            episode_reward += r
            episode_step += 1

            # 存进memory
            epoch_actions.append(action)
            epoch_qs.append(q)
            agent.store_transition(obs, action, r, new_obs, done) 
            # 新旧 状态更新
            obs = new_obs
            for d in range(len(done)):  # 对每一个agent进行reset
                if done[d]:
                    if nenvs == 1:
                        agent.reset()
```
**训练阶段**


```python
for t_train in range(nb_train_steps):
   	# 噪声更新
	if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
      distance = agent.adapt_param_noise()
      epoch_adaptive_distances.append(distance)
     # agent 训练
   	cl, al = agent.train()
```



#####　ddpg—learner

该类下，主要包含了各种DDPG中所需要包含的操作，包括利用状态值的actor 和critic 的 前向传播 

、保存数据到经验池、从经验池提取数据  进行 后向传播训练、噪声的增加以及初始化等工作。

###### 1、创建target—net、及其更新函数
创建target—network 120-126行
```python
target_actor = copy(actor)          
target_actor.name = 'target_actor'
self.target_actor = target_actor

target_critic = copy(critic)
target_critic.name = 'target_critic'
self.target_critic = target_critic
```

创建 target—net的更新  

```
# 先创建单个网络函数   36行定义的函数
def get_target_updates(vars, target_vars, tau)
# 返回的是两组操作op，一组是硬更新 一组是软更新。
# 每组更新都是一个对每一个参数 进行 更新。
	return tf.group(*init_updates), tf.group(*soft_updates)

# 2个网络的更新函数   149行 class 中定义的函数	
def setup_target_network_updates(self)
	可以得到self.target_init_updates  self.target_soft_updates 
```



######  2、actor 和critic 的 前向传播

128行  首先需要创建loss 以及 创建actor 与 critic之间的链接

```python
# actor
self.actor_tf = actor(normalized_obs0)
# critic 输入中的动作位置 为placeholder  
self.normalized_critic_tf = critic(normalized_obs0, self.actions)
self.critic_tf = denormalize(tf.clip_by_value(self.normalized_critic_tf,self.return_range[0], self.return_range[1]), self.ret_rms)

# critic 输入中的动作位置 为actor的输出
self.normalized_critic_with_actor_tf = critic(normalized_obs0, self.actor_tf, reuse=True)
self.critic_with_actor_tf =denormalize(tf.clip_by_value(self.normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]), self.ret_rms)

# target Q值计算
Q_obs1 = denormalize(target_critic(normalized_obs1, target_actor(normalized_obs1)), self.ret_rms)
self.target_Q = self.rewards + (1. - self.terminals1) * gamma * Q_obs1
```

259行 step 函数 是在每次交互过程中 ，根据当前状态 前向传输。根据当前状态 求取动作和Q值

```python
def step(self, obs, apply_noise=True, compute_Q=True):
	feed_dict = {self.obs0: U.adjust_shape(self.obs0, [obs])}# 送入数据
	# 利用网络计算动作和Q值 
      action, q = self.sess.run([actor_tf, self.critic_with_actor_tf],	feed_dict=feed_dict)  
      # 之后是为了增加噪声
      noise = self.action_noise()
	action += noise
      action = np.clip(action, self.action_range[0], self.action_range[1])
```

###### 3、反向传播

172 行 创建actor 网络训练

$\nabla_{\theta} J\left(\pi_{\theta}\right)$是$Q$对actor的参数求导数。 

采用的是 利用action 的输出作为输入的critic的输出 

因为经验回放 更新actor的时候是对当前actor的参数求导，所以必须对当前actor输入state 然后求得action 再将此时的action和state 送入critic ，并最后得到Q值 来更新 actor 参数。

```
def setup_actor_optimizer(self):
	self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)# Q值
	self.actor_grads = U.flatgrad(self.actor_loss, self.actor.trainable_vars, clip_norm=self.clip_norm) # 计算梯度
      self.actor_optimizer = MpiAdam(var_list=self.actor.trainable_vars,beta1=0.9, beta2=0.999, epsilon=1e-08)
```

183 行 创建critic 网络训练

更新critic的时候，从经验库中取得的数据，其reward 是当时state-action所得到的，而此时critic网络参数经由多次训练之后，发生了非常大的变化， 所以必须用当前的网络在计算一遍Q值然后，利用当前target 网络Q值和 当前 main 网络Q值 加上当时的reward 重新计算。

```python
def setup_critic_optimizer(self):
 	normalized_critic_target_tf = tf.clip_by_value(normalize(self.critic_target, self.ret_rms), self.return_range[0], self.return_range[1])
	self.critic_loss = tf.reduce_mean(tf.square(self.normalized_critic_tf - normalized_critic_target_tf))
	# 187-196 在这里会对critic的loss 增加 l2 约束。
      self.critic_grads = U.flatgrad(self.critic_loss, self.critic.trainable_vars, clip_norm=self.clip_norm)  # 计算梯度 
      self.critic_optimizer = MpiAdam(var_list=self.critic.trainable_vars,
            beta1=0.9, beta2=0.999, epsilon=1e-08) # 反向训练      
```

289 行 train

```
def train(self):
	# 经验池随机采样
	batch = self.memory.sample(batch_size=self.batch_size)
	ops = [self.actor_grads, self.actor_loss, self.critic_grads, self.critic_loss]
      # 根据采样数据重新计算Q值等。
      actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run(ops, feed_dict={
            self.obs0: batch['obs0'],
            self.actions: batch['actions'],
            self.critic_target: target_Q,
        })
      # 训练328 行
	self.actor_optimizer.update(actor_grads, stepsize=self.actor_lr)
      self.critic_optimizer.update(critic_grads, stepsize=self.critic_lr)
```

###### 4、噪声

噪声主要是为了增加action的探索作用。噪声主要有两种 一种是 静态参数的 一种是 动态参数（未使用）

噪声的生成主要是通过首先对actor 进行 copy   （155行函数）

```
def setup_param_noise(self, normalized_obs0):
	param_noise_actor = copy(self.actor)
     self.perturbed_actor_tf = param_noise_actor(normalized_obs0)
     
```

然后对copy后的actor的输出增加噪声

```
#50 行
def get_perturbed_actor_updates(actor, perturbed_actor, param_noise_stddev):
	# 增加均值为零 方差为param_noise_stddev的 高斯噪声
	updates.append(tf.assign(perturbed_var, var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)))

```

执行增加噪声, 在step函数中直接选择

```
#259行 
if self.param_noise is not None and apply_noise:
    actor_tf = self.perturbed_actor_tf  # 注意 这里只选择了参数固定的噪声。
else:
    actor_tf = self.actor_tf

```
其他函数 def reset(self):# 初始化噪声


###### 5、功能函数

```
# 初始化 将所有网络初始化、优化器初始化、硬更新一次target网络
def initialize(self, sess):
# 软更新target_net
def update_target_net(self):

# 通过从 数据库中采样数据并得到所有结果的函数
def setup_stats(self):
def get_stats(self):

```
