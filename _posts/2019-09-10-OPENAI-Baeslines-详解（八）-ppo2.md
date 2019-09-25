---
layout: post
title: "OPENAI Baeslines 详解（八）OOP2"
date: 2019-07-05
excerpt: "并行环境的PPO"
tags: [openai, baselines, code]
comments: true
---

# OPENAI Baeslines 详解（八）PPO2

创建policy

```
policy = build_policy(env, network, **network_kwargs)
```

