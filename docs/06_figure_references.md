# 课程插图引用索引

本文档记录了"从零到 UNet"课程中所有经典教育图示的来源与引用信息。

---

## 第 1 章 · 什么是深度学习 (`ch01-what-is-dl.html`)

| 图名 | 说明 | 来源 | URL |
|------|------|------|-----|
| AI/ML/DL 层次关系图 | 深度学习的层次架构——深度学习是机器学习的子集，机器学习是人工智能的子集 | Dive into Deep Learning (d2l.ai) | https://d2l.ai/_images/deeplearning-architecture.svg |

---

## 第 2 章 · 感知机 (`ch02-perceptron.html`)

| 图名 | 说明 | 来源 | URL |
|------|------|------|-----|
| 人工神经元模型 | 经典人工神经元模型——输入经过加权求和后通过激活函数产生输出 | Wikipedia - Artificial Neuron | https://upload.wikimedia.org/wikipedia/commons/6/60/ArtificialNeuronModel_english.png |
| XOR 问题示意图 | XOR 问题是线性不可分的——无法用一条直线将两类点完全分开 | Dive into Deep Learning (d2l.ai) | https://d2l.ai/_images/output_mlp_9b92b4_6_0.svg |

---

## 第 3 章 · 多层感知机 (`ch03-mlp.html`)

| 图名 | 说明 | 来源 | URL |
|------|------|------|-----|
| MLP 架构图 | 多层感知机架构——包含输入层、隐藏层和输出层，每一层的神经元与下一层全连接 | Dive into Deep Learning (d2l.ai) | https://d2l.ai/_images/mlp.svg |

---

## 第 4 章 · 激活函数 (`ch04-activation.html`)

| 图名 | 说明 | 来源 | URL |
|------|------|------|-----|
| ReLU 激活函数 | ReLU 激活函数曲线——f(x) = max(0, x)，是目前最常用的激活函数 | Dive into Deep Learning (d2l.ai) | https://d2l.ai/_images/output_relu_721bc2_3_0.svg |

---

## 第 5 章 · 损失函数 (`ch05-loss.html`)

| 图名 | 说明 | 来源 | URL |
|------|------|------|-----|
| 3D 损失曲面 | 损失函数的三维曲面——训练目标是找到曲面上的最低点 | Dive into Deep Learning (d2l.ai) | https://d2l.ai/_images/loss_3d.svg |

---

## 第 7 章 · 反向传播 (`ch07-backprop.html`)

| 图名 | 说明 | 来源 | URL |
|------|------|------|-----|
| 反向传播计算图 | 反向传播计算图——梯度从损失函数出发，沿计算图反向传播到每个参数 | Dive into Deep Learning (d2l.ai) | https://d2l.ai/_images/backprop.svg |

---

## 第 8 章 · 卷积神经网络 (`ch08-cnn.html`)

| 图名 | 说明 | 来源 | URL |
|------|------|------|-----|
| LeNet-5 架构图 | LeNet-5 卷积神经网络架构（1998）——经典的 CNN 结构：卷积 → 池化 → 全连接 | Dive into Deep Learning (d2l.ai) | https://d2l.ai/_images/lenet.svg |
| CNN 特征层级可视化 | CNN 各层学到的特征可视化——从底层的边缘和纹理，到高层的物体部件和完整对象 | Zeiler & Fergus (2013) via Dive into Deep Learning | https://d2l.ai/_images/zeiler-fergus-1.svg |

---

## 第 9 章 · 编码器-解码器 (`ch09-encoder-decoder.html`)

| 图名 | 说明 | 来源 | URL |
|------|------|------|-----|
| 编码器-解码器架构图 | 编码器压缩输入提取语义特征，解码器恢复空间分辨率生成分割结果 | Dive into Deep Learning (d2l.ai) | https://d2l.ai/_images/encoder-decoder.svg |

---

## 第 10 章 · UNet 架构 (`ch10-unet.html`)

| 图名 | 说明 | 来源 | URL |
|------|------|------|-----|
| U-Net 架构图 | U-Net 架构（Ronneberger et al., 2015）——经典的 U 形编码器-解码器结构，蓝色箭头为跳跃连接 | Olaf Ronneberger et al. (2015) | https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png |

---

## 第 11 章 · 动手实践 (`ch11-practice.html`)

| 图名 | 说明 | 来源 | URL |
|------|------|------|-----|
| IoU 交并比示意图 | IoU（Intersection over Union）——预测区域与真实区域的交集除以并集 | PyImageSearch | https://b2633864.smushcdn.com/2633864/wp-content/uploads/2016/09/iou_equation.png?lossy=2&strip=1&webp=1 |

---

## 第 12 章 · UNet 变体 (`ch12-unet-variants.html`)

| 图名 | 说明 | 来源 | URL |
|------|------|------|-----|
| UNet++ 架构图 | UNet++ 通过密集嵌套的跳跃路径连接编码器和解码器，逐步融合不同层级的特征 | Zhou et al., UNet++ GitHub | https://raw.githubusercontent.com/MrGiovanni/UNetPlusPlus/master/Figures/fig_UNet++.png |
| Attention U-Net 注意力门控 | Attention U-Net 的注意力门控模块，自动学习聚焦于关键区域 | Oktay et al., Attention-Gated-Networks GitHub | https://raw.githubusercontent.com/ozan-oktay/Attention-Gated-Networks/master/figures/figure1.png |
| 3D U-Net / V-Net 架构图 | 3D 体积分割网络架构示意——将所有 2D 操作替换为 3D 版本 | Çiçek et al. (2016) / Milletari et al. (2016) | https://raw.githubusercontent.com/mattmacy/vnet.pytorch/master/images/diagram.png |
| TransUNet 架构图 | TransUNet 混合架构——ViT 编码器提取全局特征，CNN 解码器恢复空间细节 | Chen et al. (2021) | https://raw.githubusercontent.com/tamasino52/UNETR/main/Arche.JPG |
| Swin-UNet / Swin-UNETR 架构图 | 基于 Swin Transformer 的 U 形分割架构，使用窗口注意力机制 | MONAI SwinUNETR | https://raw.githubusercontent.com/Project-MONAI/research-contributions/main/SwinUNETR/BTCV/assets/swin_unetr.png |
| UNETR 架构图 | UNETR——将 ViT 作为编码器，通过跳跃连接与 CNN 解码器相连，专为 3D 医学图像分割设计 | Hatamizadeh et al., UNETR | https://raw.githubusercontent.com/tamasino52/UNETR/main/Arche.JPG |
| DDPM 扩散模型架构图 | 去噪扩散概率模型（DDPM）——UNet 学习在每一步预测并去除噪声 | Lilian Weng | https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM.png |
| 潜空间扩散模型架构图 | 潜空间扩散模型（Latent Diffusion）架构——先用 VAE 编码到潜空间，在潜空间中用 UNet 去噪 | Lilian Weng | https://lilianweng.github.io/posts/2021-07-11-diffusion-models/latent-diffusion-arch.png |

---

## 引用格式说明

所有图片均使用以下 HTML 格式嵌入：

```html
<figure class="figure">
  <img src="[URL]" alt="[description]" style="max-width:100%; border-radius:8px; margin:16px 0;">
  <figcaption style="font-size:0.85em; color:#888; text-align:center; margin-top:8px;">
    图: [description]. 来源: <a href="[source_url]" target="_blank">[source_name]</a>
  </figcaption>
</figure>
```

## 主要图片来源

| 来源 | 说明 | 链接 |
|------|------|------|
| Dive into Deep Learning (d2l.ai) | 开源深度学习教材，包含大量高质量图示 | https://d2l.ai |
| Wikipedia / Wikimedia Commons | 免费教育图示库 | https://en.wikipedia.org |
| UNet 原始论文 | Ronneberger et al. 2015 的官方页面 | https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/ |
| PyImageSearch | 计算机视觉教育网站 | https://pyimagesearch.com |
| CS231n (Stanford) | 斯坦福计算机视觉课程 | https://cs231n.github.io |
