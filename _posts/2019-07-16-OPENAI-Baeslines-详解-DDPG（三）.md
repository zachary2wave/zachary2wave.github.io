---
layout: post
title:  OPENAI-Baeslines-详解-DDPG（三）
author: Xiangyu Zhang
---

The DDPG（deep deterministic policy gradient）was proposed in 2015 by Deep mind and Davide Silver. This algorithm was established on the architecture of Actor-Critic, which is efficient in dealing with continuous action domain problem. Meanwhile, by choosing action in determinate way, the sample  efficient is greatly improved.

DDPG 深度确定性策略梯度下降算法。[论文链接](https://arxiv.org/abs/1509.02971)。采用了Actor-Critic 架构，可以有效的处理连续域的问题。

同时，其actor的确定性动作输出，提高了采样的有效性。

##Actor-Critic and DDPG

The reinforcement learning is aiming at learning a policy to guide the agent interact with environment and to get a higher reward.   Traditionally, the policy is expressed by $\pi_{\theta}(a|s)$,  a statistical distribution that gives the possibility of the action $a$ that the agent chosen in state $s$, and $\theta$ is the parameter of  distribution that adjust to let the agent to find a better action.

Then the possible of a trajectory under policy $\pi$ can be expressed as: 
$$
\underbrace {{p_\theta }\left( {{{\bf{s}}_1},{{\bf{a}}_1}, \ldots ,{{\bf{s}}_T},{{\bf{a}}_T}} \right)}_{{p_\theta }(\tau )} = p\left( {{{\bf{s}}_1}} \right)\prod\limits_{t = 1}^T {{\pi _\theta }} \left( {{{\bf{a}}_t}|{{\bf{s}}_t}} \right)p\left( {{{\bf{s}}_{t + 1}}|{{\bf{s}}_t},{{\bf{a}}_t}} \right)
$$
Thereby the expectancy of reward under the policy is :
$$
\begin{aligned} J\left(\pi_{\theta}\right) &=\int_{\mathcal{S}} \rho^{\pi}(s) \int_{\mathcal{A}} \pi_{\theta}(s, a) r(s, a) \mathrm{d} a \mathrm{d} s \\ &=\mathbb{E}_{s \sim \rho^{\pi}, a \sim \pi_{\theta}}[r(s, a)] \end{aligned}
$$
where the $\rho^{\pi}(s)$ denote the discounted state distribution that the state happen 
$$
\rho^{\pi}(s')=\int_{\mathcal{S}} \sum_{t=1}^{\infty} \gamma^{t-1} p_{1}(s) p\left(s \rightarrow s^{\prime}, t, \pi\right) \mathrm{d} s
$$
Then we use the gradient $\nabla_{\theta} J\left(\pi_{\theta}\right)$ to update the $\theta$.  However, the gradient is instable because of the stochastic reward. To fix this, the most common way is that embedding the value method and replacing the reward by the value function $Q$, that is the expectancy of future reward.

Then, we get the actor-critic algorithm, which update the parameter $\theta$ by 
$$
\begin{aligned} \nabla_{\theta} J\left(\pi_{\theta}\right) &=\int_{\mathcal{S}} \rho^{\pi}(s) \int_{\mathcal{A}} \nabla_{\theta} \pi_{\theta}(a | s) Q^{\pi}(s, a) \mathrm{d} a \mathrm{d} s \\ &=\mathbb{E}_{s \sim \rho^{\pi}, a \sim \pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(a | s) Q^{\pi}(s, a)\right] \end{aligned}
$$
From the equation above, we need the value function $Q$ and the policy $\pi_{\theta}$. The policy function $\pi_{\theta}$, the actor, evaluates the action. From the [Davide Silver' work](http://proceedings.mlr.press/v32/silver14.pdf),  the  Deterministic Policy Gradient is provide being more sample efficient than stochastic case. Just like the Deep Q Network,  we can get the value function $Q$ can be approached by a neural network. 

强化学习算法的主要目标是去学习一个策略，来指导agent与环境交互从而得到更好的收益。策略$\pi_{\theta}(a|s)$是以$\theta$为参数的概率分布，代表不同状态下所采用的动作的概率分布。在学习的过程中不断的改变该函数的参数 $\theta$，从而改变应对环境的策略，以得到更好的奖励。当策略固定时，其所遍历的状态动作概率可以表示为
$$
\underbrace {{p_\theta }\left( {{{\bf{s}}_1},{{\bf{a}}_1}, \ldots ,{{\bf{s}}_T},{{\bf{a}}_T}} \right)}_{{p_\theta }(\tau )} = p\left( {{{\bf{s}}_1}} \right)\prod\limits_{t = 1}^T {{\pi _\theta }} \left( {{{\bf{a}}_t}|{{\bf{s}}_t}} \right)p\left( {{{\bf{s}}_{t + 1}}|{{\bf{s}}_t},{{\bf{a}}_t}} \right)
$$
对单个状态而言，其到达概率为：
$$
\rho^{\pi}(s')=\int_{\mathcal{S}} \sum_{t=1}^{\infty} \gamma^{t-1} p_{1}(s) p\left(s \rightarrow s^{\prime}, t, \pi\right) \mathrm{d} s
$$
那么在策略$\pi_{\theta}(a|s)$下得到的期望收益可以表示为：
$$
\begin{aligned} J\left(\pi_{\theta}\right) &=\int_{\mathcal{S}} \rho^{\pi}(s) \int_{\mathcal{A}} \pi_{\theta}(s, a) r(s, a) \mathrm{d} a \mathrm{d} s \\ &=\mathbb{E}_{s \sim \rho^{\pi}, a \sim \pi_{\theta}}[r(s, a)] \end{aligned}
$$
利用期望梯度 $\nabla_{\theta} J\left(\pi_{\theta}\right)$来更新参数$\theta$ , 从而改变策略调整在不同状态时得到的动作，进而获得最大收益。然而，由于梯度的随机不稳定的问题，导致式（4）存在不稳定，收敛差等问题。通常的办法是结合值函数的方法，即采用actor-critic架构，将式（4）中的奖励代替为$Q$ 值，可以得到：
$$
\begin{aligned} \nabla_{\theta} J\left(\pi_{\theta}\right) &=\int_{\mathcal{S}} \rho^{\pi}(s) \int_{\mathcal{A}} \nabla_{\theta} \pi_{\theta}(a | s) Q^{\pi}(s, a) \mathrm{d} a \mathrm{d} s \\ &=\mathbb{E}_{s \sim \rho^{\pi}, a \sim \pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(a | s) Q^{\pi}(s, a)\right] \end{aligned}
$$
从上式中可以看出actor-critic架构可以分为两个部分，其中actor部分代表所学习的策略$\pi_{\theta}(a|s)$，该当采用确定性的策略时，策略可以被拟合为一个输入为状态，输出为动作的函数。Critic部分是为了估计未来奖励期望Q，从而评估当前actor策略的好坏，并利用Critic所估计奖励的期望指导策略$\pi$的改变。

 



















## Baseline 中的DDPG

 