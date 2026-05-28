# 课程计划：从零到 UNet —— 深度学习入门课程

## 课程定位
- 目标受众：完全没有深度学习基础的学习者
- 教学目标：理解并能实现 UNet 图像分割网络
- 教学风格：可视化优先、直觉先行、螺旋递进

## 技术栈
- HTML5 + CSS3 (Pico CSS 框架)
- KaTeX (数学公式渲染)
- Canvas API (动画和可视化)
- Alpine.js (轻量级交互)
- 原生 JS (神经网络可视化)

## 课程章节 (11 份 HTML)

| 编号 | 文件名 | 标题 | 核心内容 | 互动元素 |
|------|--------|------|----------|----------|
| 00 | index.html | 课程导航首页 | 课程介绍、学习路径、章节导航 | 进度条、导航卡片 |
| 01 | ch01-what-is-dl.html | 什么是深度学习 | AI/ML/DL关系、应用案例、课程预览 | 动态概念图 |
| 02 | ch02-perceptron.html | 感知机 | 权重、偏置、线性分类、XOR问题 | 可交互感知机模拟器 |
| 03 | ch03-mlp.html | 多层感知机 | 层叠、隐藏层、万能近似定理 | 可调层数的MLP可视化 |
| 04 | ch04-activation.html | 激活函数 | Sigmoid/ReLU/Tanh、非线性 | 函数图形交互绘制器 |
| 05 | ch05-loss.html | 损失函数 | MSE、交叉熵、损失曲面 | 3D损失曲面可视化 |
| 06 | ch06-optimization.html | 优化与梯度下降 | SGD、学习率、Adam | 梯度下降动画模拟器 |
| 07 | ch07-backprop.html | 反向传播 | 链式法则、计算图、autograd | 计算图交互演示 |
| 08 | ch08-cnn.html | 卷积神经网络 | 卷积、池化、经典架构 | 卷积操作动画演示 |
| 09 | ch09-encoder-decoder.html | 编码器-解码器 | 语义分割、上采样、FCN | 编码解码过程动画 |
| 10 | ch10-unet.html | UNet 架构 | U型结构、跳跃连接、应用 | UNet交互式架构图 |
| 11 | ch11-practice.html | 动手实践 | 代码实现、训练过程、评估 | 嵌入式代码编辑器 |

## 共享资源
- assets/css/style.css — 全局样式
- assets/js/common.js — 共享工具函数
- assets/js/nn-viz.js — 神经网络可视化库
- assets/js/conv-viz.js — 卷积可视化库
- assets/js/chart-viz.js — 图表可视化库

## 每页结构
1. 章节标题 + 导航
2. 互动演示区（先动手体验）
3. 直觉解释（用类比和图示）
4. 数学公式（KaTeX 渲染）
5. 代码示例
6. 本章小结 + 下一章链接
