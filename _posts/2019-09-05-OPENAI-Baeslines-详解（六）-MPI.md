---
layout: post
title: "OPENAI Baeslines 详解（六）MPI"
date: 2019-07-05
excerpt: "你都不record tarjectory，你怎么知道问题在哪？"
tags: [openai, baselines, code]
comments: true
---

# OPENAI Baeslines 详解（六）MPI







把环境的数据保存下来是找问题原因的一个关键技巧，利用baselines 的函数可以轻松地保存数据成各种形式。

Baseline有两种保存数据的方式：一种是建立Monitor 一种是 Callbacks， 两种办法都是可行的。



###　basesline 保存形式　







