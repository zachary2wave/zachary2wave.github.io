---
layout: post
title:  OPENAI Baeslines 详解 DDPG（三）
author: Xiangyu Zhang
---

The DDPG（deep deterministic policy gradient）was proposed in 2015 by Deep mind and Davide Silver. This algorithm was established on the architecture of Actor-Critic, which is efficient in dealing with continuous action domain problem. Meanwhile, by choosing action in determinate way, the sample  efficient is greatly improved.

DDPG 深度确定性策略梯度下降算法。[论文链接](<https://arxiv.org/abs/1509.02971>)。采用了Actor-Critic 架构，可以有效的处理连续域的问题。

同时，其actor的确定性动作输出，提高了采样的有效性。

## DDPG原理

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
From the equation above, we need the value function $Q$ and the policy $\pi_{\theta}$.  The policy function $\pi_{\theta}$ 







Just like the Deep Q Network,  we can get the value function $Q$ can be approached by two neural network. Also, the 



 





 the  Deterministic Policy Gradient is 





















## Baseline 中的DDPG

 