---
layout: post
title: "OPENAI Baeslines 详解（一）开始学习"
date: 2019-07-05
excerpt: "你都不record tarjectory，你怎么知道问题在哪？"
tags: [openai, baselines, code]
comments: true
---

# OPENAI Baeslines 详解（五）保存数据

把环境的数据保存下来是找问题原因的一个关键技巧，利用baselines 的函数可以轻松地保存数据成各种形式。

Baseline有两种保存数据的方式：一种是建立**Monitor** 一种是** Callbacks**， 两种办法都是可行的。

当然你也可以用tensorboard 来观察你的整个训练过程。

###　Monitor

Monitor  监视器，相当于将env进行一层包装Wrapper，将env 放在监视之下。

```python
from baselines.bench import Monitor
env = Monitor(env, log_path, allow_early_resets=True)
# 输入的env为gym.make创建的，如果是多env环境会报错。
# log—path 是保存当前环境的地方。
```

完全未修改的监视器，只能输出 平均reward 、训练时常 和 所利用时间。

当然不能满足我们的需求。最简单的办法  修改源代码。

在bench中 找到montior ，然后找到step和update函数 。

update 的输入中，中加入任何你要记录的东西，并将其加入之后的字典变量epinfo。，比如说：

```
def update(self, ob, rew, done, info, action):  #58行
	
	epinfo = {"ob": ob, "action": action, 're': rew ,'done': done , "t": round(time.time() - self.tstart, 6)}、
```

并更新在step中的调用update的时候的输入 。

之后，需要在112行 中fieldname中加入：

```
self.logger = csv.DictWriter(self.f, fieldnames=('ob', 'action', 're', 'done', 't')+tuple(extra_keys))
```



### Callbacks









