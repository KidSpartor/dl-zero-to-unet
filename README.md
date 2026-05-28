# 从零到 UNet —— 深度学习入门课程

一套完整的交互式 HTML 课程，从零基础到理解并实现 UNet 图像分割网络。

## 快速开始

在浏览器中打开 `html/index.html` 即可开始学习。

```bash
# macOS
open html/index.html

# 或使用 Python 启动本地服务器
cd course && python3 -m http.server 8000
# 然后访问 http://localhost:8000/html/
```

## 课程结构

| 章节 | 文件 | 标题 | 互动内容 |
|------|------|------|----------|
| - | `index.html` | 课程导航 | 进度追踪、章节卡片 |
| 1 | `ch01-what-is-dl.html` | 什么是深度学习 | AI/ML/DL 关系图、应用案例标签页 |
| 2 | `ch02-perceptron.html` | 感知机 | 可交互感知机模拟器、训练动画 |
| 3 | `ch03-mlp.html` | 多层感知机 | 可调层数网络图、前向传播动画 |
| 4 | `ch04-activation.html` | 激活函数 | 函数/导数绘图器、可拖拽点 |
| 5 | `ch05-loss.html` | 损失函数 | 回归线+误差可视化、MSE/MAE切换 |
| 6 | `ch06-optimization.html` | 优化与梯度下降 | 损失曲面热力图、SGD/Momentum/Adam动画 |
| 7 | `ch07-backprop.html` | 反向传播 | 计算图交互演示、前向/反向传播动画 |
| 8 | `ch08-cnn.html` | 卷积神经网络 | 卷积核滑动动画、池化演示 |
| 9 | `ch09-encoder-decoder.html` | 编码器-解码器 | 编解码过程动画、信息瓶颈可视化 |
| 10 | `ch10-unet.html` | UNet 架构 | 完整UNet架构图、跳跃连接效果对比 |
| 11 | `ch11-practice.html` | 动手实践 | 代码实现、IoU/Dice交互计算器 |

## 技术栈

- HTML5 + CSS3 (无框架依赖)
- KaTeX (数学公式渲染)
- Canvas API (动画和可视化)
- Google Fonts (Inter + JetBrains Mono)
- 总外部依赖约 100KB

## 文件结构

```
course/
├── README.md
├── docs/                    # 调研文档和课程计划
│   ├── 01_research_fundamentals.md
│   ├── 02_research_unet.md
│   ├── 03_research_interactive.md
│   ├── 04_research_learning_path.md
│   └── 05_course_plan.md
├── html/                    # 课程页面 (12 份 HTML)
│   ├── index.html
│   ├── ch01-what-is-dl.html
│   ├── ...
│   └── ch11-practice.html
└── assets/                  # 共享资源
    ├── css/style.css
    └── js/
        ├── common.js        # 工具函数
        └── nn-viz.js        # 神经网络可视化库
```

## 特色

- 完全离线可用，无需服务器
- 深色模式自适应
- 响应式设计，支持移动端
- 每章都有可交互的 Canvas 演示
- 学习进度自动保存 (localStorage)
- 中文内容，KaTeX 数学公式
