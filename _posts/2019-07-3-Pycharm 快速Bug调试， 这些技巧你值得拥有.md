---
layout: post
title: Pycharm 及 Python 小技巧
description: >
Pycharm 快速Bug调试， 这些技巧你值得拥有
tags: [Zee]
author: Xiangyu Zhang
---


# Why Python in ML？

Python已经占据深度学习的大半部分江山。虽然也有大量的库和程序使用C 或 JAVA ，但是依然感觉撼动不了Python，在机器学习中的地位。Github上，搜一下Deep Learning，这个数据，自然证明了一切。
![Github中Deep learning搜索结果](https://img-blog.csdnimg.cn/20190705185250875.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2t5c2d1cWZ4ZnI=,size_16,color_FFFFFF,t_70)
至于为什么，知乎已经讨论的很明确了。大家随便看看就好了。
[知乎]https://www.zhihu.com/question/30105838
# 编译器的选择
Python 作为一个影响如此之大的语言，各家的编译器支持已经是非常普遍的情况。
Xcode 、VS、Vim 、Sublime TEXT3 作为通用的编译器，环境搭建相当简单、上手容易，各程序猿的癖好不同，随意选择。
Pycharm、Jupyter Notebook 和 anaconda的spyder，可能是大多数人的选择吧。
各有优缺点，但是Pycharm在17年之后，不断改进对于 科学计算这一块的支持，现在的2019版本已经非常强大，并逐渐吸收了各个编译器的优点，虽然比MATLAB的差得多，但是实际上已经非常强大了。
# Pycharm 小技巧
这个地方可能许多人已经提到过，或正在使用中，我将我使用中最常用的几个技巧分享了。
## （一）Debug 中的执行语句。
调Bug应该是Bug 员，最头痛的一件事。 尤其是对Github Clone玩家而言，Python报错各种底层的错全部显示出来，底层代码又不能花个几小时去看，就算看了估计也不懂啥意思，再加上函数套函数 调用来调用去。
Bug猿 ，那么断点、Debug 是最好的办法。 通常如果保证低层代码有效的情况下，就Debug到出错的你写的行程序，然后！重点在于和MATLAB一样，不仅可以查看当前的变量，而且可以根据当前变量，进行计算执行语句。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190705191459479.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2t5c2d1cWZ4ZnI=,size_16,color_FFFFFF,t_70)
点开红色小框框， 然后输入你想要执行的语句。
举个例子：当Debug在这个地方 你又不想去庞大的self变量中去找到你想要的变量 那么！就可以Print（self.XXX）
同样 Tensorflow 的tensor一点都看不到 ，不爽怎么办， print（tf.session（XXXX））有木有！
同样，大多数的都是维度问题都可以用print(len(XXX))或者print（XXX.shape）
## （二）Alt+enter
+ 代码的规格化是相当重要的 ，不仅仅是对别人看代码有用 对自己写代码也是非常重要的。毕竟工整的代码 会极大的简化 写代码过程中的找的过程。选中代码Alt+Enter。轻松简单，还没有了令人苦恼的下波浪线。
+ 另外　还可以　增加　type hint  选中Python的 变量  



## （三） 分段执行
在眼红了jupyter note 可以分段执行了很久之后  pycharm 2018 的时候就加入了分段执行的功能。
不过这个功能建立在Python console + Ipython+scientific mode 的基础上
注意：只有打开了scientific mode（view->scientific mode）  并且你的Ipython是可用（直接点开下面的python console 如果正常出现IN[1] :  就说明可以 不行就pip install ipython）
 才可以使用

在每一段代码前都加入#%% 
然后在代码的右侧会自动的出现运行标志，这个相当有用 ，免除了重复读取大量数据的苦恼。同时运行完了之后，可以直接在下面的位置出现的in[x]:中输入语句，利用当前的变量工作，同时当前变量可以直接显示。 已经很接近matlab的用户体验了。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190706101829713.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2t5c2d1cWZ4ZnI=,size_16,color_FFFFFF,t_70)
## （四） 一切都可以Ctrl +点
看该函数的库， 用ctrl + 鼠标点一下该 函数自动跳到这个库中去，方便实用！！！简单灵活。
不知道这个函数干嘛的 ， ctrl+鼠标点一下 看看说明， 不用去查numpy、scipy、matplotlib 官网。
不知道这个函数输入是什么 ctrl+鼠标点一下 。

# Python 技巧
## try…… expect 语句
这个语句非常好用，相当于matlab 2018 之后的版本中，增加的在错误之后暂停的功能。

举例来说： 
1 当遇到了Bug， 不知道原因 ，尤其是在前几百次中都OK就这一次不OK的条件下，估计是哪个if 的BUG的时候！
try：
	错误语句
expect Exception
	print 或者保存下来所有的数据 更好的检测BUG原因
2 想提前终止循环（只能用一次！！！！！）
	try：
		循环语句
	expect KeyboardInterrupt
		剩下的程序
3 为了不造成 除以0的数字爆炸
	try：
		除法
	expect ZeroDivisionError
		新除法
## assert语句
这个就是 判断你的变量是不是符合你的标准。 相当于 
if 。。。。
else 
	raise error
最好的例子还是判断维度正确 或者 数据类型正确 否则 输入进某个网络中错误就找不到在哪里。
