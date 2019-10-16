---
layout: post
title: "OPENAI-Baeslines-详解(九)-加噪声"
date: 2019-08-16
excerpt: "加噪声是增加探索幅度最简单的方式。本章给出baseline中的加噪声的方式。"
tags: [openai, baselines, code]
comments: true
---



探索与利用一直是强化学习中最值得去研究的两个方向，如何保持一定探索幅度的情况下搜索到最优也是PPO中增加熵惩罚项的关键所在。加噪声是增加探索幅度最简单的方式。本这个博文给出baseline中如何调整增加噪声的。







