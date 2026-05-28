# UNet and Related Architectures: A Research Survey

This document provides a structured overview of the UNet architecture, its key prerequisites, variants, related segmentation architectures, applications, and benchmark datasets. It is intended as a companion reference for the course materials.

---

## Table of Contents

1. [Foundational Concepts](#1-foundational-concepts)
2. [The Original UNet Paper](#2-the-original-unet-paper)
3. [UNet Variants and Improvements](#3-unet-variants-and-improvements)
4. [Related Segmentation Architectures](#4-related-segmentation-architectures)
5. [Transformer-Based UNet Variants](#5-transformer-based-unet-variants)
6. [Applications Beyond Biomedical Imaging](#6-applications-beyond-biomedical-imaging)
7. [Important Datasets and Benchmarks](#7-important-datasets-and-benchmarks)
8. [Key Takeaways](#8-key-takeaways)

---

## 1. Foundational Concepts

### 1.1 Encoder-Decoder Architecture

The encoder-decoder paradigm is a general architectural pattern where:

- **Encoder**: Progressively compresses the input into a lower-dimensional, high-level feature representation (bottleneck).
- **Decoder**: Reconstructs the desired output from the bottleneck, progressively upsampling back to the original resolution.

This concept originates from sequence-to-sequence models in NLP and has been widely adopted in computer vision for tasks such as segmentation, super-resolution, and image generation.

**Key Paper (RNN Encoder-Decoder):**

| Field | Detail |
|-------|--------|
| **Title** | Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation |
| **Authors** | Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio |
| **Year** | 2014 |
| **Venue** | EMNLP 2014 |
| **arXiv** | [1406.1078](https://arxiv.org/abs/1406.1078) |
| **Summary** | Introduced the RNN encoder-decoder framework and the GRU (Gated Recurrent Unit). The encoder maps an input sequence to a fixed-length vector, and the decoder generates the output sequence from that vector. This work laid the groundwork for attention mechanisms and encoder-decoder architectures in deep learning. |

### 1.2 Skip Connections (Residual Connections)

Skip connections allow gradients to bypass one or more layers, flowing directly from earlier layers to later layers. This addresses the vanishing gradient problem and enables training of very deep networks.

**Key Paper:**

| Field | Detail |
|-------|--------|
| **Title** | Deep Residual Learning for Image Recognition |
| **Authors** | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research) |
| **Year** | 2016 |
| **Venue** | CVPR 2016 |
| **arXiv** | [1512.03385](https://arxiv.org/abs/1512.03385) |
| **Summary** | Proposed residual learning: instead of learning a desired mapping H(x) directly, layers learn a residual F(x) = H(x) - x, so the output is F(x) + x via a shortcut connection. This enabled training networks with 152+ layers (ResNet-152) and won 1st place at ILSVRC 2015. Skip connections are a core component of UNet and many modern architectures. |

### 1.3 Fully Convolutional Networks (FCN)

FCNs replaced fully connected layers in classification networks with convolutional layers, enabling dense per-pixel predictions. This was a breakthrough for semantic segmentation.

**Key Paper:**

| Field | Detail |
|-------|--------|
| **Title** | Fully Convolutional Networks for Semantic Segmentation |
| **Authors** | Jonathan Long, Evan Shelhamer, Trevor Darrell |
| **Year** | 2015 |
| **Venue** | CVPR 2015 (extended version in IEEE TPAMI, 2017) |
| **arXiv** | [1411.4038](https://arxiv.org/abs/1411.4038) |
| **Summary** | Demonstrated that classification CNNs (AlexNet, VGGNet, GoogLeNet) can be adapted into fully convolutional networks for pixel-wise prediction. Introduced transposed convolution (deconvolution) layers for upsampling, and skip connections to fuse coarse semantic information with fine appearance information. Evaluated on PASCAL VOC, NYUDv2, and SIFT Flow. This paper is the direct predecessor to UNet. |

---

## 2. The Original UNet Paper

| Field | Detail |
|-------|--------|
| **Title** | U-Net: Convolutional Networks for Biomedical Image Segmentation |
| **Authors** | Olaf Ronneberger, Philipp Fischer, Thomas Brox |
| **Year** | 2015 |
| **Venue** | MICCAI 2015 (Medical Image Computing and Computer-Assisted Intervention) |
| **arXiv** | [1505.04597](https://arxiv.org/abs/1505.04597) |
| **DOI** | 10.1007/978-3-319-24574-4_28 |

### Key Contributions

1. **Symmetric U-Shaped Architecture**: The network consists of a contracting path (encoder) and an expanding path (decoder) that form a distinctive "U" shape.

2. **Skip Connections**: Feature maps from the encoder are concatenated with corresponding decoder feature maps at each resolution level, enabling the decoder to recover fine spatial details lost during downsampling.

3. **Encoder Path**: Repeated application of 3x3 convolutions (unpadded) followed by ReLU and 2x2 max-pooling. At each downsampling step, the number of feature channels is doubled.

4. **Decoder Path**: Upsampling via 2x2 up-convolution that halves the number of feature channels, concatenation with the corresponding cropped encoder feature map, followed by two 3x3 convolutions and ReLU.

5. **Data Augmentation with Elastic Deformations**: The paper emphasized that elastic deformations of training images were key to learning from very few annotated samples, which is critical in biomedical imaging where annotations are expensive.

6. **Overlap-Tile Strategy**: A tiling strategy that allows seamless segmentation of large images by using overlapping tiles, where the missing context is provided by mirroring the input image.

7. **Weighted Loss**: A pre-computed weight map is applied to the loss function to force the network to learn the separation borders between touching cells.

### Architecture Summary

```
Input (572x572x1)
    |
    v
[Encoder: Contracting Path]
    Conv 3x3 x2 -> MaxPool 2x2  (572->568->564, then 284)
    Conv 3x3 x2 -> MaxPool 2x2  (284->280->276, then 138)
    Conv 3x3 x2 -> MaxPool 2x2  (138->134->130, then 65)
    Conv 3x3 x2 -> MaxPool 2x2  (65->61->57, then 28)
    |
    v
[Bottleneck]
    Conv 3x3 x2  (28->24->20)
    |
    v
[Decoder: Expanding Path]
    UpConv 2x2 + Concat -> Conv 3x3 x2  (20->36+crop->32->28)
    UpConv 2x2 + Concat -> Conv 3x3 x2  (68->64->60)
    UpConv 2x2 + Concat -> Conv 3x3 x2  (136->132->128)
    UpConv 2x2 + Concat -> Conv 3x3 x2  (272->268->264)
    |
    v
[Output]
    Conv 1x1  (264->2, per-pixel classification)
```

---

## 3. UNet Variants and Improvements

### 3.1 3D U-Net

| Field | Detail |
|-------|--------|
| **Title** | 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation |
| **Authors** | Ozgun Cicek, Ahmed Abdulkader, Soeren Boettcher, Fabian Isensee, Jens Petersen, Klaus H. Maier-Hein |
| **Year** | 2016 |
| **Venue** | MICCAI 2016 |
| **arXiv** | [1606.06650](https://arxiv.org/abs/1606.06650) |
| **Summary** | Extended the 2D UNet to 3D volumetric segmentation by replacing all 2D operations with their 3D counterparts (3D convolutions, 3D max-pooling, 3D up-convolutions). Key innovation: learned dense volumetric segmentation from sparse annotations (only a few annotated slices per volume), using a weighted loss function. Designed for volumetric medical data such as CT and MRI scans. |

### 3.2 V-Net

| Field | Detail |
|-------|--------|
| **Title** | V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation |
| **Authors** | Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi |
| **Year** | 2016 |
| **Venue** | 3DV 2016 (International Conference on 3D Vision) |
| **arXiv** | [1606.04797](https://arxiv.org/abs/1606.04797) |
| **Summary** | Another 3D extension of UNet for volumetric segmentation. Key innovations: (1) residual connections within each stage (inspired by ResNet), (2) Dice loss function for handling class imbalance, which is pervasive in medical imaging. Demonstrated on prostate MRI segmentation. V-Net became a foundational work in 3D medical image segmentation. |

### 3.3 UNet++ (Nested U-Net)

| Field | Detail |
|-------|--------|
| **Title** | UNet++: A Nested U-Net Architecture for Medical Image Segmentation |
| **Authors** | Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, Jianming Liang |
| **Year** | 2018 (MICCAI 2018); extended in Medical Image Analysis, 2019 |
| **arXiv** | [1807.10165](https://arxiv.org/abs/1807.10165) |
| **DOI** | 10.1016/j.media.2019.02.001 |
| **Summary** | Introduced dense convolution blocks between the encoder and decoder, creating nested and dense skip pathways. This eliminates the semantic gap between encoder and decoder feature maps by progressively upsampling and refining features at each skip level. Supports deep supervision, enabling model pruning at test time. Achieved significant improvements on liver, cell, and lung nodule segmentation benchmarks. |

### 3.4 Attention U-Net

| Field | Detail |
|-------|--------|
| **Title** | Attention U-Net: Learning Where to Look for the Pancreas |
| **Authors** | Ozan Oktay, Jo Schlemper, Loic Le Folgoc, Matthew Lee, Mattias Heinrich, Kazunari Misawa, Kensaku Mori, Steven McDonagh, Niles Hammerla, Bernhard Kainz, Ben Glocker, Daniel Rueckert |
| **Year** | 2018 |
| **Venue** | MIDL 2018 (Medical Imaging with Deep Learning) |
| **arXiv** | [1804.03999](https://arxiv.org/abs/1804.03999) |
| **Summary** | Introduced attention gates (AGs) that can be seamlessly integrated into standard UNet architectures. The attention gates learn to suppress irrelevant features and highlight salient features useful for the specific task, without requiring extensive additional parameters. Demonstrated on pancreas segmentation in CT scans, where the target structure has large shape and size variability. |

### 3.5 nnU-Net (no-new-U-Net)

| Field | Detail |
|-------|--------|
| **Title** | nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation |
| **Authors** | Fabian Isensee, Paul F. Jaeger, Simon A. A. Kohl, Jens Petersen, Klaus H. Maier-Hein |
| **Year** | 2021 |
| **Venue** | Nature Methods, Volume 18, pp. 203-211 |
| **DOI** | [10.1038/s41592-020-01008-z](https://doi.org/10.1038/s41592-020-01008-z) |
| **Summary** | Not a new architecture per se, but a framework that automatically configures a UNet-based pipeline for any new segmentation task. nnU-Net adapts its architecture (2D, 3D full-resolution, 3D cascade), preprocessing, training scheme, postprocessing, and ensemble strategy based on dataset properties (image size, voxel spacing, class ratios). Benchmarked on 23 diverse public datasets, achieving state-of-the-art or competitive performance across all of them. Has become the de facto baseline for medical image segmentation. |
| **GitHub** | [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) |

---

## 4. Related Segmentation Architectures

### 4.1 SegNet

| Field | Detail |
|-------|--------|
| **Title** | SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation |
| **Authors** | Vijay Badrinarayanan, Alex Handa, Roberto Cipolla |
| **Year** | 2017 (BMVC 2015 for basic version; TPAMI 2017 for full version) |
| **Venue** | IEEE TPAMI 2017 |
| **Summary** | Encoder-decoder architecture based on VGG-16. Key innovation: uses max-pooling indices from the encoder for upsampling in the decoder, rather than learning upsampling parameters. This makes SegNet significantly more memory-efficient than alternatives like FCN or UNet. Designed for scene understanding applications including road scene segmentation and autonomous driving. |

### 4.2 DeepLab (v1, v2, v3, v3+)

| Field | Detail |
|-------|--------|
| **Title** | DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs |
| **Authors** | Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille |
| **Year** | 2017 (DeepLab v2); v1 at ICLR 2015 |
| **Venue** | IEEE TPAMI 2018 |
| **arXiv** | [1606.00915](https://arxiv.org/abs/1606.00915) (v2); [1706.05587](https://arxiv.org/abs/1706.05587) (v3+) |
| **Summary** | A family of architectures for semantic segmentation. Key innovations: (1) Atrous (dilated) convolution to control feature resolution without increasing parameters, (2) Atrous Spatial Pyramid Pooling (ASPP) for multi-scale processing, (3) Fully connected CRF for boundary refinement. DeepLab v3+ adds a decoder module on top of v3. Built on ResNet-101 backbone. Primarily designed for general scene segmentation (PASCAL VOC, Cityscapes). |

### 4.3 PSPNet (Pyramid Scene Parsing Network)

| Field | Detail |
|-------|--------|
| **Title** | Pyramid Scene Parsing Network |
| **Authors** | Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia |
| **Year** | 2017 |
| **Venue** | CVPR 2017 |
| **arXiv** | [1612.01105](https://arxiv.org/abs/1612.01105) |
| **Summary** | Introduced Pyramid Pooling Module to aggregate contextual information at different scales. Uses global average pooling at multiple grid scales and concatenates the results to capture both local and global context. Won 1st place in ImageNet Scene Parsing Challenge 2016. |

---

## 5. Transformer-Based UNet Variants

The recent integration of Vision Transformers with UNet-style architectures has produced a new family of hybrid and pure-transformer segmentation models.

### 5.1 TransUNet

| Field | Detail |
|-------|--------|
| **Title** | TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation |
| **Authors** | Jieneng Chen, Yongyi Lu, Qihang Yu, Xiangde Luo, Ehsan Adeli, Yan Wang, Le Lu, Alan L. Yuille, Yuyin Zhou |
| **Year** | 2021 |
| **arXiv** | [2102.04306](https://arxiv.org/abs/2102.04306) |
| **Summary** | One of the pioneering works combining Vision Transformers with UNet. Uses a ViT encoder to capture global context from image patches, with a CNN-based hybrid approach and a cascaded upsampler (CUP) decoder with skip connections. Addresses the limitation of pure CNNs (limited receptive field) while maintaining fine-grained spatial information through skip connections. Evaluated on Synapse multi-organ CT and ACDC cardiac segmentation. |

### 5.2 Swin-UNet

| Field | Detail |
|-------|--------|
| **Title** | Swin-UNet: U-like Pure Transformer for Medical Image Segmentation |
| **Authors** | Hu Cao, Yueyue Wang, Joy Chen, Dongsheng Jiang, Xiaopeng Zhang, Qi Tian, Manning Wang |
| **Year** | 2022 |
| **Venue** | MLMI Workshop, MICCAI 2022 |
| **arXiv** | [2105.05537](https://arxiv.org/abs/2105.05537) |
| **Summary** | The first pure Transformer architecture (no CNN components) built in a UNet-like encoder-decoder design for medical image segmentation. Uses Swin Transformer blocks as the encoder and bottleneck, with a symmetric decoder using patch expanding layers and skip connections. Demonstrated on Synapse multi-organ CT and ACDC cardiac segmentation. |

### 5.3 UNETR

| Field | Detail |
|-------|--------|
| **Title** | UNETR: Transformers for 3D Medical Image Segmentation |
| **Authors** | Ali Hatamizadeh, Yucheng Tang, Vishwesh Nath, Dong Yang, Andriy Myronenko, Bennett Landman, Holger Roth, Daguang Xu |
| **Year** | 2022 |
| **Venue** | WACV 2022 (IEEE/CVF Winter Conference on Applications of Computer Vision) |
| **arXiv** | [2103.10504](https://arxiv.org/abs/2103.10504) |
| **Summary** | Uses a Vision Transformer (ViT) as the encoder for 3D volumetric medical image segmentation, with a contracting-expanding structure and skip connections. The transformer captures long-range dependencies in 3D volumes. Evaluated on BTCV multi-organ segmentation and Synapse datasets. |

---

## 6. Applications Beyond Biomedical Imaging

While UNet was originally designed for biomedical image segmentation, its architecture has been adopted far beyond that domain.

### 6.1 Diffusion Models for Image Generation

UNet has become the standard denoising backbone in modern generative AI. Key applications include:

- **Stable Diffusion** (Stability AI): Uses a UNet-based denoiser operating in latent space for text-to-image generation.
- **DALL-E 2** (OpenAI): Employs diffusion with UNet-like architecture for text-to-image synthesis.
- **Imagen** (Google): Uses a UNet denoiser for high-fidelity text-to-image generation.
- **Stable Video Diffusion (SVD)**: Extends to video generation using 3D UNets with temporal attention.

The UNet works well in diffusion models because of its multi-scale feature extraction (via skip connections), effective denoising at various noise levels, and flexibility in conditioning (text embeddings, class labels, etc.).

### 6.2 Other Applications

| Application | Description |
|------------|-------------|
| **Super-Resolution** | Enhancing image resolution (e.g., SR3) |
| **Image Inpainting** | Filling in missing or masked regions |
| **Image-to-Image Translation** | Style transfer, colorization, domain adaptation |
| **Video Generation** | Temporal diffusion with 3D UNets |
| **Autonomous Driving** | Road scene segmentation, obstacle detection |
| **Satellite/Aerial Imagery** | Land use classification, building detection |
| **Industrial Inspection** | Defect detection in manufacturing |
| **Audio Generation** | Diffusion for waveform or spectrogram synthesis |
| **Point Cloud Processing** | 3D shape segmentation and generation |
| **Molecular Design** | Drug discovery and protein structure prediction |
| **Weather/Climate Modeling** | Forecasting with diffusion-based models |
| **Robotics** | Diffusion policies for action planning |

---

## 7. Important Datasets and Benchmarks

### 7.1 Biomedical/Medical Imaging

| Dataset | Description | Task | Reference |
|---------|-------------|------|-----------|
| **ISBI Cell Tracking Challenge** | Microscopy image sequences with ground truth annotations for cell segmentation and tracking. Includes multiple cell types (Fluo-N2DH-SIM+, PhC-C2DH-U373, DIC-C2DH-HeLa). | Cell segmentation and tracking | [celltrackingchallenge.net](http://celltrackingchallenge.net/) |
| **ISBI 2012 EM Segmentation** | Electron microscopy images of Drosophila larvae neural structures. One of the original benchmarks used in the UNet paper. | Neuronal structure segmentation | Arganda-Carreras et al., 2015 |
| **Synapse Multi-organ CT** | Abdominal CT scans with annotations for multiple organs. Widely used for evaluating TransUNet, Swin-UNet, UNETR. | Multi-organ segmentation | MICCAI Challenge |
| **ACDC (Automatic Cardiac Diagnosis Challenge)** | Cardiac MRI with annotations for left ventricle, right ventricle, and myocardium. | Cardiac segmentation | Bernard et al., 2018 |
| **BTCV (Beyond the Cranial Vault)** | Abdominal CT with 13 organ annotations. | Multi-organ segmentation | MICCAI Challenge |

### 7.2 General/Scene Understanding

| Dataset | Description | Task | Reference |
|---------|-------------|------|-----------|
| **Cityscapes** | 5,000 high-quality annotated + 20,000 coarsely annotated images from 50 cities. 30 semantic classes (road, building, car, person, vegetation, etc.). | Urban scene segmentation | [cityscapes-dataset.com](https://www.cityscapes-dataset.com/) |
| **PASCAL VOC 2012** | 20 object classes + background. Standard benchmark for object detection and segmentation. | Semantic/instance segmentation | Everingham et al. |
| **ADE20K** | Over 20,000 images with dense annotations spanning 150 categories. | Scene parsing | Zhou et al., 2017 |
| **NYUDv2** | RGB-D indoor scene dataset with depth maps and semantic labels. | Indoor segmentation | Silberman et al., 2012 |

---

## 8. Key Takeaways

1. **UNet's Core Innovation**: The symmetric encoder-decoder structure with skip connections was the key insight -- skip connections allow the decoder to recover fine spatial details that would otherwise be lost during the encoder's downsampling.

2. **FCN as a Precursor**: Fully Convolutional Networks (Long et al., 2015) demonstrated the feasibility of dense pixel-wise prediction with CNNs. UNet improved upon FCN by introducing a more effective decoder path with skip connections that concatenate (rather than add) features.

3. **Skip Connections are Universal**: Skip connections, pioneered in the context of segmentation by FCN and UNet, and formalized by ResNet, are now a fundamental building block of virtually all modern deep learning architectures, including Transformers.

4. **2D to 3D**: The 3D U-Net and V-Net extended the architecture to volumetric data, which is essential for medical imaging (CT, MRI) where 3D context matters.

5. **Attention Mechanisms**: Attention U-Net showed that learning where to focus (spatial attention) can improve segmentation of structures with high variability in shape and size.

6. **Self-Configuration**: nnU-Net demonstrated that careful engineering of the entire pipeline (preprocessing, architecture selection, training, postprocessing) matters as much as architectural innovations.

7. **Transformer Integration**: TransUNet, Swin-UNet, and UNETR represent the latest evolution, combining the local feature extraction of convolutions with the global context modeling of Transformers.

8. **Beyond Biomedical**: UNet has become the backbone of modern generative AI (diffusion models), demonstrating that its multi-scale architecture with skip connections is a general-purpose design principle, not limited to segmentation.

---

## References (Complete List)

| # | Short Citation | Full Title | Authors | Year | Venue | Link |
|---|---------------|------------|---------|------|-------|------|
| 1 | Ronneberger et al., 2015 | U-Net: Convolutional Networks for Biomedical Image Segmentation | Olaf Ronneberger, Philipp Fischer, Thomas Brox | 2015 | MICCAI 2015 | [arXiv:1505.04597](https://arxiv.org/abs/1505.04597) |
| 2 | Long et al., 2015 | Fully Convolutional Networks for Semantic Segmentation | Jonathan Long, Evan Shelhamer, Trevor Darrell | 2015 | CVPR 2015 | [arXiv:1411.4038](https://arxiv.org/abs/1411.4038) |
| 3 | He et al., 2016 | Deep Residual Learning for Image Recognition | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun | 2016 | CVPR 2016 | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |
| 4 | Cicek et al., 2016 | 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation | Ozgun Cicek et al. | 2016 | MICCAI 2016 | [arXiv:1606.06650](https://arxiv.org/abs/1606.06650) |
| 5 | Milletari et al., 2016 | V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation | Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi | 2016 | 3DV 2016 | [arXiv:1606.04797](https://arxiv.org/abs/1606.04797) |
| 6 | Badrinarayanan et al., 2017 | SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation | Vijay Badrinarayanan, Alex Handa, Roberto Cipolla | 2017 | IEEE TPAMI | [mi.eng.cam.ac.uk/projects/segnet/](https://mi.eng.cam.ac.uk/projects/segnet/) |
| 7 | Chen et al., 2017 | DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs | Liang-Chieh Chen et al. | 2017 | TPAMI 2018 | [arXiv:1606.00915](https://arxiv.org/abs/1606.00915) |
| 8 | Zhao et al., 2017 | Pyramid Scene Parsing Network | Hengshuang Zhao et al. | 2017 | CVPR 2017 | [arXiv:1612.01105](https://arxiv.org/abs/1612.01105) |
| 9 | Zhou et al., 2018 | UNet++: A Nested U-Net Architecture for Medical Image Segmentation | Zongwei Zhou et al. | 2018 | MICCAI 2018 / MedIA 2019 | [arXiv:1807.10165](https://arxiv.org/abs/1807.10165) |
| 10 | Oktay et al., 2018 | Attention U-Net: Learning Where to Look for the Pancreas | Ozan Oktay et al. | 2018 | MIDL 2018 | [arXiv:1804.03999](https://arxiv.org/abs/1804.03999) |
| 11 | Isensee et al., 2021 | nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation | Fabian Isensee et al. | 2021 | Nature Methods | [DOI:10.1038/s41592-020-01008-z](https://doi.org/10.1038/s41592-020-01008-z) |
| 12 | Chen et al., 2021 | TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation | Jieneng Chen et al. | 2021 | arXiv | [arXiv:2102.04306](https://arxiv.org/abs/2102.04306) |
| 13 | Cao et al., 2022 | Swin-UNet: U-like Pure Transformer for Medical Image Segmentation | Hu Cao et al. | 2022 | MLMI/MICCAI 2022 | [arXiv:2105.05537](https://arxiv.org/abs/2105.05537) |
| 14 | Hatamizadeh et al., 2022 | UNETR: Transformers for 3D Medical Image Segmentation | Ali Hatamizadeh et al. | 2022 | WACV 2022 | [arXiv:2103.10504](https://arxiv.org/abs/2103.10504) |
| 15 | Cho et al., 2014 | Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation | Kyunghyun Cho et al. | 2014 | EMNLP 2014 | [arXiv:1406.1078](https://arxiv.org/abs/1406.1078) |

---

## Suggested Reading Order

For a systematic understanding, the recommended reading order is:

1. **Cho et al., 2014** -- Understand the encoder-decoder concept
2. **Long et al., 2015** (FCN) -- See how CNNs were adapted for dense prediction
3. **Ronneberger et al., 2015** (UNet) -- The core architecture
4. **He et al., 2016** (ResNet) -- Understand skip connections formally
5. **Cicek et al., 2016** / **Milletari et al., 2016** (3D U-Net / V-Net) -- 3D extensions
6. **Badrinarayanan et al., 2017** (SegNet) / **Chen et al., 2017** (DeepLab) -- Related architectures
7. **Zhou et al., 2018** (UNet++) / **Oktay et al., 2018** (Attention U-Net) -- Key improvements
8. **Isensee et al., 2021** (nnU-Net) -- State-of-the-art engineering
9. **Chen et al., 2021** / **Cao et al., 2022** / **Hatamizadeh et al., 2022** -- Transformer-based variants

---

*Document compiled on 2026-05-28.*
