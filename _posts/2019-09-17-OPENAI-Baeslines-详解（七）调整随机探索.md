---
layout: post
title: "OPENAI Baeslines 详解（七）调整随机探索"
date: 2019-09-17
excerpt: "让随机探索科学起来！！"
tags: [openai, baselines, code]
comments: true
---

# OPENAI Baeslines 详解（七）调整随机探索







创建policy

```
policy = build_policy(env, network, **network_kwargs)
```