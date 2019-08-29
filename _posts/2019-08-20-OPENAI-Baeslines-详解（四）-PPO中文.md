---
layout: post
title: "OPENAI Baeslines 详解（四）PPO"
date: 2019-08-20
excerpt: "2017年 [OpenAI](https://arxiv.org/abs/1707.06347)  和 [Deep Mind](https://arxiv.org/abs/1707.02286)先后推出了TRPO和PPO算法，该算法通过限制了new policy和 Old policy之间的KL散度大小（Kullback-Leibler Divergence，KLD），从而解决学习率过大引起不收敛的问题。"
tags: [openai, baselines, code]
comments: true
---

2017年 [OpenAI](https://arxiv.org/abs/1707.06347)  和 [Deep Mind](https://arxiv.org/abs/1707.02286)先后推出了TRPO和PPO算法，该算法通过限制了new policy和 Old policy之间的KL散度大小（Kullback-Leibler Divergence，KLD），从而解决学习率过大引起不收敛的问题。

# OPENAI-Baeslines-PPO

---

目录

[TOC]

---

## 基本原理

### TRPO

基于策略的强化学习的主要目标是找到一个“ **可以让带有折扣的未来期望的收益达到最大**  ” 的策略。带有折扣的未来期望的收益可以表示为：
$$
\eta(\pi)=\mathbb{E}_{s_{0}, a_{0}, \ldots}\left[\sum_{t=0}^{\infty} \gamma^{t} r\left(s_{t}\right)\right]
$$
TRPO的主要想法就是在每一步更新的策略的时候，新的策略都要比老的策略好。那么新旧策略之间的期望收益差可以表示为：
$$
\eta (\tilde \pi ) = \eta (\pi ) +\mathbb{E} \underbrace {{_{{\tau\sim\tilde \pi }}}\left[ {\sum\limits_{t = 0}^\infty  {{\gamma ^t}} {A_\pi }\left( {{s_t},{a_t}} \right)} \right]}_{{\rm{policy \quad  gap}}}
$$
Policy gap 是老策略的优势函数（advantage function）在新策略采样轨迹下的期望值。（2）式可以进一步表示为：
$$
\begin{aligned} \eta(\tilde{\pi}) &=\eta(\pi)+\sum_{t=0}^{\infty} \sum_{s} P\left(s_{t}=s | \tilde{\pi}\right) \sum_{a} \tilde{\pi}(a | s) \gamma^{t} A_{\pi}(s, a) \\&=\eta(\pi)+\sum_{s} \sum_{t=0}^{\infty} \gamma^{t} P\left(s_{t}=s | \tilde{\pi}\right) \sum_{a} \tilde{\pi}(a | s) A_{\pi}(s, a) \\ &=\eta(\pi)+\sum_{s} \rho_{\tilde{\pi}}(s) \sum_{a} \tilde{\pi}(a | s) A_{\pi}(s, a) \end{aligned}
$$
在上面的等式中， $\rho_{\tilde{\pi}}(s) =\sum_{t=0}^{\infty} \gamma^{t} P\left(s_{t}=s | \tilde{\pi}\right)$ 表示在新策略 $\tilde{\pi}$ 下状态$s_t$的带有折扣因子的概率。当没有得到新策略的时候，该式是非常难以求得的。 如果采用老策略的$\rho_{\pi}(s) $  来近似，则可以得到下式：
$$
L_{\pi}(\tilde{\pi})=\eta(\pi)+\sum_{s} \rho_{\pi}(s) \sum_{a} \tilde{\pi}(a | s) A_{\pi}(s, a)
$$
但是这里必须考虑到的是，新策略与老策略的差距非常小的时候，才有$L_{\pi}(\tilde{\pi})\approx \eta(\tilde{\pi})$。

如果，采用如下的方式来跟新策略：
$$
\pi_{\mathrm{new}}(a | s)=(1-\alpha) \pi_{\mathrm{old}}(a | s)+\alpha \pi^{\prime}(a | s)
$$
那么$L_{\pi}(\tilde{\pi})$ 和$\eta(\tilde{\pi})$ 之间的关系可以表示为：
$$
\eta\left(\pi_{\text { new }}\right) \geq L_{\pi_{\text { old }}}\left(\pi_{\text { new }}\right)-\frac{2 \epsilon \gamma}{(1-\gamma)^{2}} \alpha^{2}
$$
其中，$
\epsilon=\max _{s}\left|\mathbb{E}_{a \sim \pi^{\prime}(a | s)}\left[A_{\pi}(s, a)\right]\right|
$，需要被注意的是，这里 $\alpha$代表了步进的大小。如果采用从方差散度（total variation divergence, $D_{T V}(p \| q)=\frac{1}{2} \sum_{i}\left|p_{i}-q_{i}\right|$）的方式来衡量新旧策略的不同时, $\alpha$可以表示为
$$
\alpha = D_{\mathrm{TV}}^{\max }(\pi, \tilde{\pi})=\max _{s} D_{T V}(\pi(\cdot | s) \| \tilde{\pi}(\cdot | s))
$$
公式（5）就可以表示为：
$$
\eta\left(\pi_{\text { new }}\right) \geq L_{\pi_{\text { old }}}\left(\pi_{\text { new }}\right)-\frac{4 \epsilon \gamma}{(1-\gamma)^{2}} \alpha^{2}
$$
其中，$\epsilon=\max _{s, a}\left|A_{\pi}(s, a)\right|$。根据 $D_{T V}(p \| q)^{2} \leq D_{K L}(p \| q)$，将TV散度更换成KL散度的关系，这是由于将KL散度将更利于求解高斯分布等模型。 这样式（7）可以写成：
$$
\begin{aligned} \eta(\tilde{\pi}) & \geq L_{\pi}(\tilde{\pi})-C D_{\mathrm{KL}}^{\max }(\pi, \tilde{\pi}) \\ & \text { where } C=\frac{4 \epsilon \gamma}{(1-\gamma)^{2}} \end{aligned}
$$
在上式中，如果新策略等于旧策略，那么不等式变为等式，所以如果我们在每一次更新迭代的时候，都使得下届最大化而求得的新策略$\tilde{\pi}$，就一定是一个比旧策略更好的策略。这样，策略寻找问题变化成一个最优化问题，即：
$$
\begin{array}{l}{\text { maximize } \quad L_{\theta_{\text { old }}}(\theta)} \\ {\text { subject to } \quad D_{\mathrm{KL}}^{\max }\left(\theta_{\text { old }}, \theta\right) \leq \delta}\end{array}
$$
有5个方法化简该问题：

+ 从公式3中可以看出$L_{\theta_{\text { old }}}(\theta)$，中包含较多常数项$\eta(\pi)$。 同时，化简后面的advantage function可以得到：

$$
\sum_{a} \tilde{\pi}_{\theta}(a | s) A_{\theta_{o l d}}=\sum_{a} \tilde{\pi}_{\theta}(a | s)\left(Q_{\theta_{o l d}}(s, a)-V_{\theta_{o l d}}(s)\right)=\sum_{a} \tilde{\pi}_{\theta}(a | s) Q_{\theta_{o l d}}(s, a)-V_{\theta_{o l d}}
$$

​		上式中的$V_{old}$ 也是常数，所以直接最小化$\sum_{a}\tilde{\pi}_{\theta}(a | s) Q_{\theta_{o l d}}(s, a)$即可。 

+ 对目标函数中的状态概率求和项 ，用期望来表示，即$\sum_{s} \rho_{\pi}(s)$  代替为 $\mathbb{E}_{s \sim \rho}$.

+ 对目标函数中的状态概率求和项 ，用期望来表示，即$\sum_{a} \tilde{\pi}_{\theta}(a | s) Q_{\theta_{o l d}}(s, a)$ 代替为 $\mathbb{E}_{a \sim \tilde{\pi}_{\theta}}$. 

+ 现实中无法得到根据新的策略采样得到$\sum_{a} \tilde{\pi}_{\theta}(a | s) A_{\theta_{o l d}}$，所以利用 the importance sampling ，可以表示为：
  $$
  \begin{aligned} E_{a \sim \tilde{\pi}(\theta)}[f(\theta)] &=\int \tilde{\pi}(\theta) f(\theta) d \theta \\ &=\int \frac{q(\theta)}{q(\theta)} \tilde{\pi}(\theta) f(\theta) d \theta \\ &=\int q(\theta) \frac{\tilde{\pi}(\theta)}{q(\theta)} f(\theta) d \theta \\ &=E_{a \sim q(\theta)}\left[\frac{\tilde{\pi}(\theta)}{q(\theta)} f(\theta)\right] \end{aligned}
  $$
  利用旧策略的采样 来完成新策略的估计。
  
+ 约束条件中，KL散度约束在每一个状态s上，所以很难求得，所以最好的方法是利用平均，即：
  $$
  \overline{D}_{\mathrm{KL}}^{\rho}\left(\theta_{1}, \theta_{2}\right) :=\mathbb{E}_{s \sim \rho}\left[D_{\mathrm{KL}}\left(\pi_{\theta_{1}}(\cdot | s) \| \pi_{\theta_{2}}(\cdot | s)\right)\right]
  $$

最后最优问题化简为：
$$
\begin{array}{l}{\text { maximize } \mathbb{E}_{s \sim \rho_{\text { old }}, a \sim \pi(\theta_{\text { old }})}\left[\frac{\tilde{\pi}(\theta)}{{\pi(\theta_{\text { old }})}} Q_{\theta_{\text { old }}}(s, a)\right]} \\ {\text { subject to } \mathbb{E}_{s \sim \rho_{\theta_{\text { old }}}}\left[\overline{D}_{\mathrm{KL}}\left(\pi_{\theta_{\text { old }}}(\cdot | s) \| \tilde{\pi}_{\theta}(\cdot | s)\right)\right] \leq \delta}\end{array}
$$
为了求解上面的最优化问题，我们必须知道 $Q_{\theta_{\text { old }}}(s, a)$ and ${\pi}(\theta_{\text { old }})$。文中给出了2中方法来估计$Q$

第一种方法single path就是，在${\pi}(\theta_{\text { old }})$下直接找出许多条轨迹 然后求解，直接平均估计每一个状态的 $Q_{\theta_{\text { old }}}(s, a)$求解。第二种办法vine是，找一个状态，在该状态下求n次 找到估计 $Q_{\theta_{\text { old }}}(s, a)$。

解式（11）的最优化问题， 可以分为两步

1、 更新方向的选择，利用一阶模型近似约束条件和目标方程：

- 目标方程部分:
  让目标方程转化为一阶近似，令 

$$
\overline{A}(\theta')=\mathbb{E}_{s \sim \rho_{\text { old }}, a \sim \pi(\theta_{\text { old }})}\left[\frac{\tilde{\pi}(\theta)}{{\pi(\theta_{\text { old }})}} Q_{\theta_{\text { old }}}(s, a)\right]
$$

那么 $\overline{A}(\theta')$一阶线性近似是 $ \nabla_{\theta} \overline{A}(\theta)^{T}\left(\theta^{\prime}-\theta\right)$， 其中$\theta'$ 是下一步的策略参数，$\theta$ 是现在的策略参数。

- 约束条件部分：

KL散度的二阶近似为:
$$
\overline{D}_{\mathrm{KL}}\left(\theta_{\mathrm{old}}, \theta\right) \approx \frac{1}{2}\left(\theta-\theta_{\mathrm{old}}\right)^{T} A\left(\theta-\theta_{\mathrm{old}}\right)
$$
 其中A代表了Fisher information matrix:
$$
A_{i j}=\frac{\partial}{\partial \theta_{i}} \frac{\partial}{\partial \theta_{j}} \overline{D}_{\mathrm{KL}}\left(\theta_{\mathrm{old}}, \theta\right)
$$
 2、 更新策略参数在该方向上执行线性搜索，确保我们在满足非线性约束的同时改善非线性目标
$$
\begin{array}{l}{\theta^{\prime}=\theta+\alpha \mathbf{F}^{-1} \nabla_{\theta} J(\theta)} \\ {\alpha=\sqrt{\frac{2 \epsilon}{\nabla_{\theta} J(\theta)^{T} \mathbf{F} \nabla_{\theta} J(\theta)}}}\end{array}
$$

### PPO

PPO 是一种基于TRPO的算法，在求解上面一些方法，但是本质上还是TRPO

PPO提出了两种算法：

第一种是约束限制转移概率$r_{t}(\theta)=\frac{\pi_{\theta}\left(a_{t} | s_{t}\right)}{\pi_{\theta} \operatorname{old}\left(a_{t} | s_{t}\right)}$：
$$
L^{C L I P}(\theta)=\hat{\mathbb{E}}_{t}\left[\min \left(r_{t}(\theta) \hat{A}_{t}, \operatorname{clip}\left(r_{t}(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_{t}\right)\right]
$$
是吧变化量限制在某一个范围之内。这种方式是最直接的形式，因为直接把变化量限定在一个范围之内，但是其中的参数 $\epsilon$ 还是没有给出一个固定的指标。

另一种约束是KL散度上的约束
$$
L^{K L P E N}(\theta)=\hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}\left(a_{t} | s_{t}\right)}{\pi_{\theta_{\text { old }}}\left(a_{t} | s_{t}\right)} \hat{A}_{t}-\beta \operatorname{KL}\left[\pi_{\theta_{\text { old }}}\left(\cdot | s_{t}\right), \pi_{\theta}\left(\cdot | s_{t}\right)\right]\right]
$$
Let $
d=\hat{\mathbb{E}}_{t}\left[\mathrm{KL}\left[\pi_{\theta_{\mathrm{old}}}\left(\cdot | s_{t}\right), \pi_{\theta}\left(\cdot | s_{t}\right)\right]\right]
$

If $d<d_{\operatorname{targ}} / 1.5, \beta \leftarrow \beta / 2$

if $d>d_{\operatorname{targ}}\times  1.5, \beta \leftarrow \beta \times 2$

实际上， 当利用（16）作为loss function 的时候，**神经网络的loss function 必须还增加策略的熵升 和 值函数误差项**
$$
L_{t}^{C L I P+V F+S}(\theta)=\hat{\mathbb{E}}_{t}\left[L_{t}^{C L I P}(\theta)-c_{1} L_{t}^{V F}(\theta)+c_{2} S\left[\pi_{\theta}\right]\left(s_{t}\right)\right]
$$
其中 $S$ 代表了在该策略下的熵的奖励， 这是为了确保充分的探索。 $L_{t}^{V F}(\theta)=\left(V_{\theta}\left(s_{t}\right)-V_{t}^{\operatorname{targ}}\right)^{2}$

所以整个算法流程是 ：  利用老策略 采样  计算 优势函数  更新策略 再循环 

![1566278934110](image\1566278934110.png)

## Baseline中的PPO

baseline的PPO有两个版本一个版本为PPO1 一个版本为PPO2

### PPO1 版本

Baseline的PPO 主要分为以下3个部分：
+ 主程序部分： pposgd_simple  
  根据 env 和 提供的神经网络 创建好整个强化学习算法架构，并输出策略pi。

+ 神经网络部分： cnn_policy和mlp_policy
  创建神经网络模型

+ 概率分布部分： common.distributions
  根据当前所输出的动作和状态，建立对应

### 概率分布部分

由于采用的是随机策略PPO输出的结果都是概率分布的参数，然后再从中采样。所以在每个action输出的时候都是输出对应的概率分布的参数。这个部分主要的功能：

+ 将对应动作参数类型转化为概率分布
+ 为对应概率分布的参数生成必要的action 的placeholder
+ 求交叉熵以及散度。
+ 对概率分布采样得到动作。

### 神经网络部分

###### 动作节点的创建
神经网络首先调用概率分布函数，将动作建立为对应概率分布，并在调用的时候已经建立好对应的节点。
###### 网络中节点初始化

状态的创建：状态建立的时候 为了避免重复创建同名称的状态， 利用了 get_placeholder 这个函数创建。创建的时候 将所有的placeholder 都放入了一个字典里 类型为{"名称"：placeholder，数据类型，数据维度}。 创建好的状态都可以直接用 函数get_placeholder_cached 来进行调用。

然后对ob进行归一化，所有都归一到-5 到 5 之间。用RunningMeanStd 函数。



###### 网络创建

vf 网络： 输入为状态  输出为vpred

pol网络： 输入为状态  输出为动作

如果是高斯变量 那么网络输出的是均值 

如果不是高斯变量  那么网络输出的是概率参数

为了多次调用不同，在每次调用该类的过程中都需要先输入独特的名称 使得不同网络便于区分。 将网络利用with tf.variable_scope(name):命名。
+ Vf网络：critic网络

  每层都相等、输入为归一化之后的状态、输出为单个元素，均采用tanh。 需要参数num_hid_layers（网络层数）、hid_size（隐藏层节点个数）。

  利用for循环 循环生成。

```python
with tf.variable_scope('vf'):
     obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
     last_out = obz
     for i in range(num_hid_layers):
         last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
     
     self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]
```

+ pol网络 ：actor网络

  每层都相等、输入为归一化之后的状态、输出为动作，均采用tanh





### 主程序部分

PPO中主程序主要部分在learning中，重要参数有

- env :  环境
- policy_fn： 网络
- clip_param: 

##### 准备工作

######　网络输出输入

首先用所提供的 网络创建新旧policy

```python
pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
```

其中 policy_fn 是一个class（类） 具体看神经网络部分。

接着提取之前所创建的 参数 状态ob 和 动作ac

输入口：

```python
atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # 优势方程 （17式中的A）
ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])
```



######  定义图中的loss

计算loss所需的值：(107-111)

```python
kloldnew = oldpi.pd.kl(pi.pd) # 计算新老策略的KL散度 
ent = pi.pd.entropy()         # 计算新策略的熵值
meankl = tf.reduce_mean(kloldnew) # 平均KL散度
meanent = tf.reduce_mean(ent)     # 平均熵
```



计算（18）式的误差  包含  三项  $L^{C L I P}(\theta)$、$L_{t}^{V F}(\theta)$、$S$

+ 16式的$L^{C L I P}(\theta)$


  然后计算公式 （16）式所需要的值

  113 - 新老策略的概率π相除 然后得到 比率
  114 - 比率乘以优势函数  优势函数 也是一个 placeholder
  115 - 经过剪裁的比率 乘以优势函数
  116  对应（16）式惩罚项 $L^{C L I P}(\theta)$

```python
ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))
surr1 = ratio * atarg 
surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg  
pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) 
```

+ $L_{t}^{V F}(\theta)=\left(V_{\theta}\left(s_{t}\right)-V_{t}^{\operatorname{targ}}\right)^{2}$    这个地方$V_{t}^{\operatorname{targ}}$  是直接给出的  因为在前面定义了 一个placeholder


```python
vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
```

+ $S$    c1 是 -entcoeff 是输入参数 
  
  ```python
  pol_entpen = (-entcoeff) * meanent
  ```

###### 生成梯度、优化器所需要的函数

```
lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
adam = MpiAdam(var_list, epsilon=adam_epsilon)
assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)    for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)
```

1 、 计算梯度  

输入ob 状态, ac 动作 , atarg 优势函数, ret  , lrmult] 

计算得到pol_surr $L^{C L I P}(\theta)$ , pol_entpen $S$ , vf_loss  $L_{t}^{V F}(\theta)$  , meankl  新老策略KL散度均值 , meanent  新策略熵值 以及对应的神经网络梯度

2、 生成优化器

3、 新旧策略的迭代

4、 计算loss ：计算优化之后的LOSS 与1 的区别在于 不计算梯度



##### 更新迭代过程

###### 前期策略采样 和 优势函数估计

```
traj_segment_generator(pi, env, horizon, stochastic):
```

horizon 是采样的多少步

stochastic 标识 利用随机采样 即 pd.sample  或者 pd.mode

######  前向采样

```
seg = seg_gen.__next__()
add_vtarg_and_adv(seg, gamma, lam)
```

###### 后向更新

```
for _ in range(optim_epochs):
    losses = [] 
    for batch in d.iterate_once(optim_batchsize):
        *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
        adam.update(g, optim_stepsize * cur_lrmult)
        losses.append(newlosses)
```





















