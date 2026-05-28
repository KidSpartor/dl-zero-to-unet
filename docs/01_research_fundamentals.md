# Deep Learning Fundamentals -- Teaching Resources Research

> Research compiled for course design: a complete beginner's path to understanding U-Net.
> Target audience: zero-knowledge learners with no prior ML/DL experience.

---

## Table of Contents

1. [Foundational Topics Roadmap](#1-foundational-topics-roadmap)
2. [Introductory Textbooks](#2-introductory-textbooks)
3. [Online Courses](#3-online-courses)
4. [Classic / Seminal Papers](#4-classic--seminal-papers)
5. [Chinese-Language Resources](#5-chinese-language-resources)
6. [Supplementary Resources](#6-supplementary-resources)
7. [Recommended Learning Path](#7-recommended-learning-path)

---

## 1. Foundational Topics Roadmap

Before a learner can understand U-Net (Ronneberger et al., 2015), they need to build up
knowledge in a specific sequence. Below is the prerequisite chain, from zero to U-Net.

### 1.1 Mathematical Prerequisites

| Topic | Why It Matters | Key Concepts |
|-------|---------------|--------------|
| **Linear Algebra** | Neural networks are built on matrix operations | Vectors, matrices, dot products, matrix multiplication, eigenvalues, transpose |
| **Calculus (single-variable)** | Gradient descent requires derivatives | Derivatives, chain rule, partial derivatives, gradients |
| **Probability & Statistics** | Loss functions and data distributions | Random variables, probability distributions, expectation, Bayes' theorem |
| **Python Programming** | All DL frameworks use Python | NumPy basics, functions, loops, classes |

**Recommended resource:** *Mathematics for Machine Learning* by Deisenroth, Faisal, and Ong
(Free PDF: http://mml-book.github.io/) -- Covers all the above in a self-contained way,
assuming only high-school math. Published by Cambridge University Press.

### 1.2 Core Deep Learning Concepts (in learning order)

```
Step 1: Perceptron & Single Neuron
  |
  v
Step 2: Activation Functions (sigmoid, ReLU, tanh, softmax)
  |
  v
Step 3: Loss Functions (MSE, cross-entropy)
  |
  v
Step 4: Gradient Descent (learning rate, batch, SGD, Adam)
  |
  v
Step 5: Backpropagation (chain rule applied to networks)
  |
  v
Step 6: Multi-Layer Perceptrons (MLPs / feedforward networks)
  |
  v
Step 7: Regularization (dropout, batch norm, weight decay, data augmentation)
  |
  v
Step 8: Convolutional Neural Networks (CNNs)
  |       - Convolution operation, filters/kernels
  |       - Pooling (max, average)
  |       - Stride, padding
  |       - Feature maps, receptive fields
  |       - Classic architectures: LeNet, AlexNet, VGG, ResNet
  |
  v
Step 9: Encoder-Decoder Architectures
  |       - Fully Convolutional Networks (FCN)
  |       - Transposed convolution (up-convolution)
  |       - Skip connections
  |
  v
Step 10: U-Net Architecture
          - Contracting path (encoder)
          - Expanding path (decoder)
          - Skip connections for precise localization
          - Biomedical image segmentation
```

### 1.3 Detailed Topic Breakdown

#### 1.3.1 Perceptron
- The simplest neural network unit: `output = activation(sum(w_i * x_i) + bias)`
- Historical context: Rosenblatt (1958)
- Limitations: can only learn linearly separable functions (Minsky & Papert, 1969)
- Multi-layer perceptrons overcome this limitation

#### 1.3.2 Activation Functions
- **Sigmoid**: `1 / (1 + e^-x)` -- outputs (0, 1), used historically, vanishing gradient problem
- **Tanh**: outputs (-1, 1), zero-centered, also suffers vanishing gradients
- **ReLU**: `max(0, x)` -- most commonly used in hidden layers, simple and effective
- **Leaky ReLU**: variant that allows small negative values
- **Softmax**: used in output layers for multi-class classification, outputs sum to 1
- **Key insight**: activation functions introduce non-linearity, enabling networks to learn complex patterns

#### 1.3.3 Loss Functions
- **Mean Squared Error (MSE)**: for regression tasks, `L = (1/n) * sum((y_pred - y_true)^2)`
- **Binary Cross-Entropy**: for binary classification
- **Categorical Cross-Entropy**: for multi-class classification
- **Dice Loss / IoU Loss**: commonly used in segmentation (relevant to U-Net)
- **Key insight**: loss functions quantify how wrong the model's predictions are

#### 1.3.4 Gradient Descent
- Optimization algorithm: `w = w - learning_rate * dLoss/dw`
- **Batch Gradient Descent**: uses entire dataset per update (slow, stable)
- **Stochastic Gradient Descent (SGD)**: uses one sample per update (fast, noisy)
- **Mini-batch Gradient Descent**: compromise between batch and SGD
- **Adam optimizer**: adaptive learning rate, combines momentum and RMSProp (Kingma & Ba, 2014)
- **Learning rate**: critical hyperparameter -- too high diverges, too low converges slowly

#### 1.3.5 Backpropagation
- The algorithm that makes training deep networks possible
- Applies the chain rule of calculus to compute gradients layer by layer
- Gradients flow backward from output to input
- Key paper: Rumelhart, Hinton & Williams (1986), "Learning representations by back-propagating errors," *Nature*
- Historical note: concepts existed earlier (Werbos 1974, Parker 1985) but this paper brought it to mainstream

#### 1.3.6 Convolutional Neural Networks (CNNs)
- Specialized architecture for spatial data (images, audio)
- **Convolutional layer**: applies learnable filters/kernels to detect features (edges, textures, patterns)
- **Pooling layer**: reduces spatial dimensions (max pooling, average pooling)
- **Stride**: step size of the filter as it slides across the input
- **Padding**: adding zeros around the border to control output size
- **Feature maps**: output of applying a filter to the input
- **Receptive field**: the region of the input that affects a particular output neuron
- **Key insight**: parameter sharing and local connectivity make CNNs efficient for images

#### 1.3.7 Encoder-Decoder & Segmentation
- **Fully Convolutional Networks (FCN)**: Long et al. (2015) -- replaced fully connected layers with convolutions for pixel-wise prediction
- **Transposed convolution**: learnable upsampling (also called deconvolution, though technically a misnomer)
- **Skip connections**: concatenate encoder features with decoder features to preserve spatial detail
- **Semantic segmentation**: assigning a class label to every pixel in an image

---

## 2. Introductory Textbooks

### 2.1 "Deep Learning" -- Goodfellow, Bengio & Courville (2016)

| Field | Details |
|-------|---------|
| **URL** | https://www.deeplearningbook.org/ (free online) |
| **Also known as** | "The Deep Learning Bible" / "The Flower Book" (花书, in Chinese) |
| **Level** | Intermediate (needs some math background) |
| **Format** | Book (online PDF, also available in print) |

**Table of Contents (abbreviated):**
- Part I -- Applied Math and Machine Learning Basics: Linear Algebra, Probability, Numerical Computation, ML Basics
- Part II -- Modern Practical Deep Networks: Feedforward Networks, Regularization, Optimization, CNNs, RNNs, Practical Methodology, Applications
- Part III -- Deep Learning Research: Autoencoders, Representation Learning, Generative Models, etc.

**Why it's good:**
- Written by three pioneers of the field (Bengio won the 2018 Turing Award)
- Comprehensive and authoritative
- Covers both theory and practice
- Free online access

**Fit in learning path:** Best as a reference book or for learners who want deep theoretical understanding. Not ideal as the *first* resource for a complete beginner due to mathematical density.

---

### 2.2 "Dive into Deep Learning" (D2L) -- Aston Zhang, Zachary Lipton, Mu Li, Alexander Smola

| Field | Details |
|-------|---------|
| **URL** | https://d2l.ai/ (English), https://zh.d2l.ai/ (Chinese) |
| **GitHub** | https://github.com/d2l-ai/d2l |
| **Level** | Beginner to Intermediate |
| **Format** | Interactive book with runnable Jupyter notebooks |
| **Frameworks** | PyTorch, TensorFlow, JAX, MXNet |

**Topics covered:**
- Foundations: data manipulation, linear algebra, calculus, probability, automatic differentiation
- Classical ML: linear regression, softmax regression, multilayer perceptrons
- CNNs: LeNet, AlexNet, VGG, GoogLeNet, ResNet, DenseNet, batch normalization
- RNNs: LSTMs, GRUs, bidirectional RNNs, sequence-to-sequence
- Transformers & Attention: multi-head attention, self-attention, vision transformers
- Optimization: SGD, momentum, Adagrad, RMSProp, Adam
- Computer Vision: object detection, semantic segmentation, neural style transfer
- NLP: Word2vec, GloVe, BERT

**Why it's good:**
- **Best resource for hands-on learners** -- every section is a runnable notebook
- Covers theory AND code side by side
- Multi-framework support (PyTorch recommended for U-Net course)
- Adopted by universities including CMU and Stanford
- Excellent Chinese translation
- Completely free and open-source
- Updated regularly (2023+ editions include transformers, BERT, etc.)

**Fit in learning path:** **Primary recommendation** as the main textbook for this course. Start with chapters 1-3 (foundations), then 7-8 (MLP), then 9 (CNNs), then the segmentation sections.

---

### 2.3 "Neural Networks and Deep Learning" -- Michael Nielsen

| Field | Details |
|-------|---------|
| **URL** | http://neuralnetworksanddeeplearning.com/ |
| **Level** | Beginner |
| **Format** | Free online book |
| **Language** | English |

**Chapters:**
1. Using neural nets to recognize handwritten digits
2. How the backpropagation algorithm works
3. Improving how neural networks learn
4. A visual proof that neural nets can compute any function
5. Why are deep neural networks hard to train?
6. Deep learning

**Why it's good:**
- Extremely clear and intuitive explanations
- Focuses on building understanding from first principles
- Uses MNIST handwritten digits as a running example (relevant to image understanding)
- Beautiful visual explanations of backpropagation
- Gentle mathematical pace -- accessible to beginners

**Fit in learning path:** Excellent **first book** for someone with zero knowledge. Covers perceptrons through CNNs with great clarity. Can be read in 1-2 weeks. Pairs well with 3Blue1Brown videos.

---

### 2.4 "Understanding Deep Learning" -- Simon J.D. Prince (2023)

| Field | Details |
|-------|---------|
| **URL** | https://udlbook.github.io/ (free PDF, slides, videos, exercises) |
| **Publisher** | MIT Press |
| **Level** | Beginner to Intermediate |
| **Format** | Book + companion YouTube lectures + exercises |

**Topics covered:**
- Introduction and supervised learning
- Shallow and deep neural networks
- Loss functions, fitting models, gradient descent
- Gradients and initialization
- Measuring performance, generalization
- Regularization (dropout, batch norm)
- CNNs, residual networks
- Transformers and attention
- Generative models (VAEs, GANs, diffusion)
- Graph neural networks
- Reinforcement learning

**Why it's good:**
- **Most modern textbook** (2023) -- covers transformers, diffusion models
- Beautifully illustrated
- Companion lecture videos on YouTube
- Exercises with solutions included
- Free online access
- Written specifically for accessibility

**Fit in learning path:** Strong alternative to D2L for learners who prefer a more traditional textbook format. Excellent companion videos make it highly self-study friendly.

---

### 2.5 "Deep Learning with Python" -- Francois Chollet (2nd ed., 2021)

| Field | Details |
|-------|---------|
| **Publisher** | Manning Publications |
| **ISBN** | 978-1617296864 |
| **Level** | Beginner (code-first) |
| **Framework** | TensorFlow / Keras |
| **Format** | Book (paid, but very affordable) |

**Topics covered:**
- Neural network fundamentals with Keras
- CNNs for computer vision
- RNNs for sequence data
- Text processing, transformers
- Generative deep learning (GANs, VAEs)
- Best practices for real-world applications

**Why it's good:**
- Written by the **creator of Keras** -- extremely practical
- Code-driven approach: you build things immediately
- Clear, intuitive explanations
- Updated 2nd edition covers modern techniques
- Great for people who learn by doing

**Fit in learning path:** Good supplementary resource for learners who prefer Keras/TensorFlow. Less theoretical depth than other options but excellent for rapid practical skill-building.

---

### 2.6 "Mathematics for Machine Learning" -- Deisenroth, Faisal & Ong

| Field | Details |
|-------|---------|
| **URL** | http://mml-book.github.io/ (free PDF) |
| **Publisher** | Cambridge University Press |
| **Level** | Beginner (math foundations) |
| **Format** | Book (free online, also in print) |

**Topics covered:**
- Part 1 -- Mathematical Foundations: Linear Algebra, Analytic Geometry, Matrix Decompositions, Vector Calculus, Probability & Distributions, Continuous Optimization
- Part 2 -- ML Topics: Linear Regression, Dimensionality Reduction (PCA), Density Estimation, Classification

**Why it's good:**
- Self-contained math prerequisites for deep learning
- Assumes only high-school mathematics
- Bridges the gap between basic math and what DL requires
- Free and well-written

**Fit in learning path:** Recommended as a **prerequisite companion** for learners who need to brush up on math. Not required if the learner already has university-level math.

---

## 3. Online Courses

### 3.1 Andrew Ng's Deep Learning Specialization (Coursera / deeplearning.ai)

| Field | Details |
|-------|---------|
| **URL** | https://www.coursera.org/specializations/deep-learning |
| **Platform** | Coursera (certificates available, paid; audit for free) |
| **Instructor** | Andrew Ng (Stanford / Google Brain / Baidu / Coursera co-founder) |
| **Duration** | ~5 months at 5 hours/week |
| **Level** | Beginner-friendly |
| **Language** | English (Chinese subtitles available) |

**Course sequence (5 courses):**
1. **Neural Networks and Deep Learning** -- NN basics, shallow/deep networks
2. **Improving Deep Neural Networks** -- Regularization, optimization (Adam, RMSProp), batch norm, hyperparameter tuning
3. **Structuring Machine Learning Projects** -- Error analysis, transfer learning, multi-task learning
4. **Convolutional Neural Networks** -- CNN foundations, ResNet, Inception, object detection, face recognition, neural style transfer
5. **Sequence Models** -- RNNs, LSTMs, word embeddings, attention, transformers

**Why it's good:**
- **Best structured introduction** for complete beginners
- Andrew Ng is an exceptionally clear teacher
- Builds intuition before diving into math
- Well-paced with quizzes and programming assignments
- Covers CNNs in depth (Course 4 directly relevant to U-Net)
- Global community of learners

**Fit in learning path:** **Primary course recommendation** alongside the D2L textbook. Courses 1 and 4 are most directly relevant to the U-Net learning path.

---

### 3.2 fast.ai -- "Practical Deep Learning for Coders"

| Field | Details |
|-------|---------|
| **URL** | https://course.fast.ai/ |
| **Platform** | Self-hosted (also YouTube) |
| **Instructor** | Jeremy Howard |
| **Duration** | ~7 weeks (self-paced) |
| **Level** | Beginner-friendly (assumes basic Python) |
| **Cost** | Free |
| **Companion book** | *Deep Learning for Coders with fastai and PyTorch* (also free online) |

**Course curriculum:**
1. Getting Started -- Training your first model
2. Deployment -- Putting models into production
3. Neural Net Foundations -- How neural nets work
4. NLP with RNNs/Transformers
5. Tabular Data
6. Collaborative Filtering
7. Convolutions & CNNs
8. ResNets & **U-Nets** (directly relevant!)
9. Generative Models (GANs, diffusion)
10. Stable Diffusion from Scratch

**Why it's good:**
- **Top-down teaching approach**: build first, understand later
- Covers U-Net explicitly (lesson 8)
- Extremely practical -- students train real models from lesson 1
- Uses PyTorch and the fastai library
- Regularly updated (2022+ version includes diffusion models)
- Vibrant community forum (forums.fast.ai)

**Fit in learning path:** Excellent alternative for learners who prefer hands-on work. The top-down approach means students see U-Net-like architectures early, which can be motivating. **Directly relevant** to the course goal.

---

### 3.3 CS231n -- Stanford "Convolutional Neural Networks for Visual Recognition"

| Field | Details |
|-------|---------|
| **URL** | https://cs231n.stanford.edu/ |
| **Platform** | Stanford (lecture videos on YouTube) |
| **Instructors** | Fei-Fei Li, Justin Johnson, Serena Yeung |
| **Level** | Intermediate (more academic/theoretical) |
| **Cost** | Free (materials and videos) |

**Topics:**
- Image classification (kNN, SVM, Softmax, neural networks)
- Backpropagation, optimization
- CNN architectures (AlexNet, VGG, GoogLeNet, ResNet)
- Training techniques (batch norm, dropout)
- Object detection (R-CNN, YOLO)
- Semantic segmentation
- Generative models (GANs, VAEs, diffusion)
- Vision transformers, self-supervised learning

**Assignments:**
1. Assignment 1: kNN, SVM, Softmax, two-layer NN from scratch
2. Assignment 2: Fully-connected nets, Batch Norm, Dropout, CNNs (PyTorch)
3. Assignment 3: RNNs, Transformers, Generative models

**Why it's good:**
- **Gold standard** university course for computer vision + deep learning
- Excellent assignments that build understanding through implementation
- Covers segmentation (relevant to U-Net)
- Academic rigor with practical assignments
- Free lecture recordings available on YouTube

**Fit in learning path:** Best for learners who want academic depth. More challenging than Andrew Ng's courses. Recommended after completing a more introductory course.

---

### 3.4 MIT 6.S191 -- Introduction to Deep Learning

| Field | Details |
|-------|---------|
| **URL** | https://introtodeeplearning.com/ |
| **Platform** | MIT (videos on YouTube: MIT Deep Learning channel) |
| **Instructors** | Alexander Amini, Ava Soleimany |
| **Duration** | Short (intensive January term) |
| **Level** | Beginner to Intermediate |
| **Cost** | Free |

**Lectures:**
1. Intro to Deep Learning
2. Deep Sequence Modeling
3. Deep Computer Vision
4. Deep Generative Modeling
5. Deep Reinforcement Learning
6. New Frontiers

**Software Labs:**
1. Deep Learning in Python / Music Generation
2. Facial Detection Systems (with bias mitigation)
3. Fine-Tuning an LLM

**Why it's good:**
- Concise and modern (updated yearly)
- Covers the breadth of deep learning quickly
- Hands-on labs with TensorFlow
- Great for getting a broad overview before diving deep

**Fit in learning path:** Good **starting point** for a quick orientation, or as a complement to a more detailed course. Too brief to be a sole resource.

---

### 3.5 3Blue1Brown -- Neural Networks Video Series

| Field | Details |
|-------|---------|
| **URL** | https://www.3blue1brown.com/ (also on YouTube) |
| **Creator** | Grant Sanderson |
| **Duration** | 4 videos (~1 hour total) |
| **Level** | Absolute beginner |
| **Cost** | Free |

**Videos:**
1. But what is a Neural Network? -- neurons, layers, MNIST digit recognition
2. Gradient descent, how neural networks learn -- cost functions, gradient descent
3. What is backpropagation really doing? -- intuition with visualizations
4. Backpropagation calculus -- rigorous mathematical derivation

**Why it's good:**
- **Best visual intuition builder** available anywhere
- Stunning animations make abstract concepts concrete
- Uses MNIST handwritten digits (directly relevant to image understanding)
- Can be watched in a single sitting
- No prerequisites beyond basic arithmetic

**Fit in learning path:** **Watch first, before anything else.** These 4 videos should be the very first resource a complete beginner encounters. They build the "aha!" moments that make everything else easier to understand.

---

### 3.6 Google Machine Learning Crash Course

| Field | Details |
|-------|---------|
| **URL** | https://developers.google.com/machine-learning/crash-course |
| **Platform** | Google (self-hosted) |
| **Level** | Beginner |
| **Cost** | Free |

**Why it's good:**
- Fast-paced overview of ML fundamentals
- Interactive visualizations
- Good starting point before a deeper course
- Uses TensorFlow

**Fit in learning path:** Optional quick start for learners who want a broad survey before committing to a longer course.

---

## 4. Classic / Seminal Papers

These are the landmark papers that introduced key concepts. Listed in recommended reading order.

### 4.1 Foundational Papers

#### Backpropagation -- Rumelhart, Hinton & Williams (1986)
| Field | Details |
|-------|---------|
| **Title** | Learning representations by back-propagating errors |
| **Published in** | *Nature*, Vol. 323, pp. 533-536, October 1986 |
| **DOI** | 10.1038/323533a0 |
| **Authors** | David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams |

**Key contribution:** Popularized the backpropagation algorithm for training multi-layer neural networks. Demonstrated that gradient descent could efficiently train hidden layers using the chain rule. Launched the modern connectionist movement.

---

#### LeNet-5 -- LeCun et al. (1998)
| Field | Details |
|-------|---------|
| **Title** | Gradient-based learning applied to document recognition |
| **Published in** | *Proceedings of the IEEE*, 86(11):2278-2324, 1998 |
| **DOI** | 10.1109/5.726791 |
| **Authors** | Yann LeCun, Leon Bottou, Yoshua Bengio, Patrick Haffner |

**Key contribution:** One of the earliest successful CNNs. Introduced the convolution-pooling-fc architecture pattern for handwritten digit recognition. Demonstrated that feature learning outperforms hand-crafted features.

---

### 4.2 The Deep Learning Revolution

#### AlexNet -- Krizhevsky, Sutskever & Hinton (2012)
| Field | Details |
|-------|---------|
| **Title** | ImageNet Classification with Deep Convolutional Neural Networks |
| **Published in** | *NeurIPS 2012* |
| **DOI** | 10.1145/3065386 |
| **Authors** | Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton |

**Key contribution:** Won the 2012 ImageNet competition by a large margin, igniting the deep learning revolution. Used ReLU activations, dropout, and data augmentation on GPUs. Demonstrated that deep CNNs dramatically outperform traditional computer vision methods.

---

#### VGGNet -- Simonyan & Zisserman (2014)
| Field | Details |
|-------|---------|
| **Title** | Very Deep Convolutional Networks for Large-Scale Image Recognition |
| **Published in** | *ICLR 2015* (arXiv:1409.1556, 2014) |
| **Authors** | Karen Simonyan, Andrew Zisserman |

**Key contribution:** Showed that network depth is critical for performance. Used small (3x3) convolution filters consistently. Simple and uniform architecture that became a standard feature extractor for many downstream tasks.

---

#### GoogLeNet / Inception -- Szegedy et al. (2014)
| Field | Details |
|-------|---------|
| **Title** | Going Deeper with Convolutions |
| **Published in** | *CVPR 2015* (arXiv:1409.4842, 2014) |
| **Authors** | Christian Szegedy et al. (Google) |

**Key contribution:** Introduced the "Inception module" -- parallel convolution branches of different sizes concatenated together. Achieved deeper networks with fewer parameters through clever architecture design.

---

#### ResNet -- He et al. (2015)
| Field | Details |
|-------|---------|
| **Title** | Deep Residual Learning for Image Recognition |
| **Published in** | *CVPR 2016* (arXiv:1512.03385, 2015) |
| **DOI** | 10.1109/CVPR.2016.90 |
| **Authors** | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun |

**Key contribution:** Introduced **residual connections** (skip connections) enabling training of very deep networks (152+ layers). Solved the degradation problem where deeper networks performed worse. The skip connection concept directly influenced U-Net's architecture.

**Relevance to U-Net:** Skip connections are a core component of U-Net's architecture.

---

### 4.3 Segmentation Papers (Direct U-Net Predecessors)

#### Fully Convolutional Networks (FCN) -- Long, Shelhamer & Darrell (2015)
| Field | Details |
|-------|---------|
| **Title** | Fully Convolutional Networks for Semantic Segmentation |
| **Published in** | *CVPR 2015* |
| **DOI** | 10.1109/CVPR.2015.7298965 |
| **Authors** | Jonathan Long, Evan Shelhamer, Trevor Darrell |

**Key contribution:** Showed that classification CNNs can be converted to segmentation networks by replacing fully connected layers with convolutional layers. Introduced skip connections for combining coarse and fine predictions. **Direct predecessor of U-Net.**

---

#### U-Net -- Ronneberger, Fischer & Brox (2015)
| Field | Details |
|-------|---------|
| **Title** | U-Net: Convolutional Networks for Biomedical Image Segmentation |
| **Published in** | *MICCAI 2015* (arXiv:1505.04597) |
| **DOI** | 10.1007/978-3-319-24574-4_28 |
| **Authors** | Olaf Ronneberger, Philipp Fischer, Thomas Brox |

**Key contribution:** The encoder-decoder architecture with skip connections for biomedical image segmentation. Contracting path captures context, expanding path enables precise localization. Achieved state-of-the-art with very few training images through data augmentation (especially elastic deformations).

**Prerequisites to understand this paper:**
- CNN fundamentals (convolution, pooling)
- VGG-style architecture understanding
- Encoder-decoder concept
- Transposed convolution
- Skip connections (from ResNet / FCN)
- Semantic segmentation task definition
- Data augmentation strategies

---

### 4.4 Training Technique Papers

#### Dropout -- Srivastava et al. (2014)
| Field | Details |
|-------|---------|
| **Title** | Dropout: A Simple Way to Prevent Neural Networks from Overfitting |
| **Published in** | *JMLR*, 15(1):1929-1958, 2014 |
| **Authors** | Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov |

**Key contribution:** Regularization technique that randomly drops neurons during training to prevent co-adaptation and overfitting. Simple yet extremely effective.

---

#### Batch Normalization -- Ioffe & Szegedy (2015)
| Field | Details |
|-------|---------|
| **Title** | Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift |
| **Published in** | *ICML 2015* (arXiv:1502.03167) |
| **Authors** | Sergey Ioffe, Christian Szegedy |

**Key contribution:** Normalizes layer inputs to stabilize and accelerate training. Enables higher learning rates and reduces sensitivity to initialization. Now a standard component in most architectures.

---

#### Adam Optimizer -- Kingma & Ba (2014)
| Field | Details |
|-------|---------|
| **Title** | Adam: A Method for Stochastic Optimization |
| **Published in** | *ICLR 2015* (arXiv:1412.6980) |
| **Authors** | Diederik P. Kingma, Jimmy Ba |

**Key contribution:** Combines the benefits of AdaGrad and RMSProp. Adaptive learning rate optimizer that works well in practice with little hyperparameter tuning. The most widely used optimizer in deep learning.

---

### 4.5 Other Important Papers

#### Attention Is All You Need -- Vaswani et al. (2017)
| Field | Details |
|-------|---------|
| **Title** | Attention Is All You Need |
| **Published in** | *NeurIPS 2017* (arXiv:1706.03762) |
| **DOI** | 10.48550/arXiv.1706.03762 |
| **Authors** | Ashish Vaswani et al. (Google Brain) |

**Key contribution:** Introduced the Transformer architecture with self-attention mechanisms. Became the foundation for BERT, GPT, and modern NLP/vision models. Relevant for understanding modern extensions of U-Net (e.g., TransUNet).

---

#### Generative Adversarial Networks (GAN) -- Goodfellow et al. (2014)
| Field | Details |
|-------|---------|
| **Title** | Generative Adversarial Nets |
| **Published in** | *NeurIPS 2014* (arXiv:1406.2661) |
| **Authors** | Ian J. Goodfellow et al. |

**Key contribution:** Introduced the adversarial training framework (generator vs. discriminator). Spawned an entire subfield of generative models. Relevant background for understanding advanced U-Net variants and medical image synthesis.

---

## 5. Chinese-Language Resources

### 5.1 Textbooks

#### "Neural Networks and Deep Learning" (神经网络与深度学习) -- Qiu Xipeng (邱锡鹏)

| Field | Details |
|-------|---------|
| **URL** | https://nndl.ai/ (also https://nndl.github.io/) |
| **GitHub Stars** | 18.8k+ |
| **Author** | Xipeng Qiu (邱锡鹏), Fudan University |
| **Publisher** | China Machine Press (机械工业出版社) |
| **Rating** | 9.4 on Douban |
| **Cost** | Free online |

**The "Dandelion Book Series" (蒲公英书系列) includes:**
1. **Neural Networks and Deep Learning (2nd Edition)** -- Core textbook, covers theory from perceptron to Transformer
2. **Neural Networks and Deep Learning: Cases and Practice** -- Hands-on companion with PyTorch and PaddlePaddle implementations
3. **Neural Networks and Deep Learning (General Education Edition)** -- Introductory version with weakened math, for non-specialists
4. **Large Models and Intelligent Agents** -- Forward-looking book on LLMs and AI agents

**Topics covered:** CNNs, RNNs, network optimization, regularization, attention mechanisms, transformers, LLMs

**Why it's good:** The most authoritative Chinese-language deep learning textbook. Systematic coverage from basics to advanced topics. The General Education Edition is perfect for beginners.

**Fit in learning path:** Use the General Education Edition for absolute beginners, then the main edition for deeper study. The Cases and Practice book provides hands-on code.

---

#### "Deep Learning" Chinese Translation (深度学习)

| Field | Details |
|-------|---------|
| **Original** | Goodfellow, Bengio & Courville (2016) |
| **Publisher (Chinese)** | People's Posts and Telecommunications Press (人民邮电出版社) |
| **Translators** | Zhao Shenjia (赵申嘉) et al. |

**Why it's good:** Official Chinese translation of the "Flower Book." Essential reference for Chinese-speaking learners who want the authoritative text.

---

#### "Deep Learning with Python" Chinese Translation (Python深度学习)

| Field | Details |
|-------|---------|
| **Original** | Chollet, "Deep Learning with Python" (2nd ed.) |
| **Publisher (Chinese)** | People's Posts and Telecommunications Press |

**Why it's good:** Chinese translation of Chollet's practical guide. Good for Chinese-speaking learners who prefer Keras/TensorFlow.

---

### 5.2 Online Courses

#### Hung-yi Lee (李宏毅) -- Machine Learning / Deep Learning

| Field | Details |
|-------|---------|
| **Platform** | YouTube / Bilibili (B站) |
| **Institution** | National Taiwan University |
| **Language** | Mandarin Chinese |
| **Cost** | Free |
| **Search on B站** | "李宏毅 机器学习" or "李宏毅 深度学习 2024" |

**Topics covered:**
- Machine learning fundamentals
- Deep learning (CNNs, RNNs, transformers)
- GANs, reinforcement learning
- Large language models (2024+ editions)
- Assignments and code materials included

**Why it's good:**
- Full Chinese-language instruction
- Engaging and humorous teaching style
- Updated annually (2024 edition includes LLM content)
- Comprehensive assignments and code

**Fit in learning path:** **Best Chinese-language course.** Can serve as the primary lecture resource for Chinese-speaking learners, supplemented by D2L Chinese edition.

---

#### Andrew Ng's Courses (Chinese Subtitles)

| Field | Details |
|-------|---------|
| **Search on B站** | "吴恩达 深度学习 中文字幕" or "吴恩达 机器学习 中文" |
| **Platform** | Coursera / Bilibili |

**Why it's good:** Andrew Ng's courses have community-contributed Chinese subtitles on Bilibili. The structured approach works well for Chinese-speaking beginners.

---

#### Li Mu (李沐) -- "Dive into Deep Learning" Video Lectures

| Field | Details |
|-------|---------|
| **Search on B站** | "李沐 动手学深度学习" |
| **Companion book** | https://zh.d2l.ai/ |
| **Language** | Mandarin Chinese |

**Why it's good:** Li Mu (one of the D2L authors) provides Chinese-language video lectures that walk through the D2L textbook chapter by chapter. The combination of video + interactive book is powerful.

---

### 5.3 Other Chinese Resources

| Resource | URL/Platform | Notes |
|----------|-------------|-------|
| **CS231n Chinese subtitles** | B站: search "CS231n 中文字幕" | Stanford course with Chinese subs |
| **Lin Xuantian ML** | B站: "林轩田 机器学习基石" | Taiwan Univ., classic ML intro |
| **PaddlePaddle tutorials** | https://www.paddlepaddle.org.cn/ | Baidu's DL framework, Chinese docs |
| **Zhihu (知乎) DL columns** | zhihu.com | Many high-quality Chinese DL articles |
| **Chinese version of PyTorch docs** | https://pytorch.apachecn.org/ | Community translation |

---

## 6. Supplementary Resources

### 6.1 Visualization & Interactive Tools

| Resource | URL | What It Provides |
|----------|-----|-----------------|
| **TensorFlow Playground** | https://playground.tensorflow.org/ | Interactive neural network visualization in browser |
| **CNN Explainer** | https://poloclub.github.io/cnn-explainer/ | Interactive visualization of CNN layers |
| **ConvNetJS** | https://cs.stanford.edu/people/karpathy/convnetjs/ | CNN in the browser |
| **Distill.pub** | https://distill.pub/ | Interactive ML research articles (archived but excellent) |
| **Weights & Biases** | https://wandb.ai/ | Experiment tracking, visualization, free for academics |

### 6.2 Practice Platforms

| Platform | URL | Notes |
|----------|-----|-------|
| **Kaggle** | https://www.kaggle.com/ | Competitions, datasets, notebooks, free GPUs |
| **Google Colab** | https://colab.research.google.com/ | Free Jupyter with GPU/TPU |
| **Papers With Code** | https://paperswithcode.com/ | SOTA results + code implementations |

### 6.3 Community & Forums

| Community | URL | Notes |
|-----------|-----|-------|
| **fast.ai Forums** | https://forums.fast.ai/ | Very beginner-friendly |
| **r/learnmachinelearning** | reddit.com/r/learnmachinelearning | Active Q&A community |
| **r/deeplearning** | reddit.com/r/deeplearning | More technical discussions |
| **Stack Overflow** | stackoverflow.com | Tag: [deep-learning] |

---

## 7. Recommended Learning Path

Below is a structured 10-week learning path from zero to U-Net, incorporating the resources above.

### Phase 0: Orientation (Day 1)

| Activity | Resource | Time |
|----------|----------|------|
| Watch 3Blue1Brown neural network series | YouTube | 1 hour |
| Play with TensorFlow Playground | playground.tensorflow.org | 30 min |

**Goal:** Build intuition about what neural networks are and how they learn.

---

### Phase 1: Math Foundations (Week 1-2)

| Activity | Resource | Time |
|----------|----------|------|
| Linear algebra essentials | 3Blue1Brown "Essence of Linear Algebra" | 3 hours |
| Calculus essentials | 3Blue1Brown "Essence of Calculus" | 3 hours |
| Probability basics | MML Book Ch. 6 | 2 hours |
| Python & NumPy basics | D2L Ch. 1-2 or any Python tutorial | 3 hours |

**Goal:** Comfortable with vectors, matrices, derivatives, and Python/NumPy.

**Chinese alternative:** 李宏毅课程前两节 + 李沐D2L视频前几节

---

### Phase 2: Neural Network Basics (Week 3-4)

| Activity | Resource | Time |
|----------|----------|------|
| Read Nielsen's book, Ch. 1-3 | neuralnetworksanddeeplearning.com | 6 hours |
| Andrew Ng DL Specialization, Course 1 | Coursera | 10 hours |
| Implement a simple NN from scratch | D2L Ch. 3-4 or Nielsen exercises | 4 hours |

**Goal:** Understand perceptrons, activation functions, loss functions, gradient descent, and backpropagation.

**Key concepts to master:**
- Forward pass and backward pass
- How weights are updated
- The role of learning rate
- Overfitting vs. underfitting

---

### Phase 3: Deep Networks & Training Techniques (Week 5-6)

| Activity | Resource | Time |
|----------|----------|------|
| Deep networks & regularization | Andrew Ng Course 2 OR D2L Ch. 7-8 | 8 hours |
| Read Prince's UDL book, Ch. 4-9 | udlbook.github.io | 6 hours |
| Implement MLP in PyTorch | D2L PyTorch notebooks | 4 hours |

**Goal:** Understand multi-layer networks, regularization, optimization (Adam), batch normalization, dropout.

---

### Phase 4: Convolutional Neural Networks (Week 7-8)

| Activity | Resource | Time |
|----------|----------|------|
| CNN fundamentals | Andrew Ng Course 4 (first half) OR D2L Ch. 9 | 6 hours |
| Classic architectures | D2L Ch. 10-12 (LeNet, AlexNet, VGG, ResNet) | 6 hours |
| Read AlexNet paper | Krizhevsky et al. (2012) | 1 hour |
| Read ResNet paper | He et al. (2015) | 1 hour |
| Implement a CNN in PyTorch | D2L notebooks or CS231n Assignment 2 | 4 hours |

**Goal:** Deep understanding of convolution, pooling, feature extraction, and modern architectures.

---

### Phase 5: Segmentation & U-Net (Week 9-10)

| Activity | Resource | Time |
|----------|----------|------|
| Semantic segmentation concepts | CS231n segmentation lecture | 2 hours |
| Read FCN paper | Long et al. (2015) | 1 hour |
| Read U-Net paper | Ronneberger et al. (2015) | 1 hour |
| Encoder-decoder architecture | fast.ai Course, Lesson 8 | 2 hours |
| Implement U-Net in PyTorch | Course practical sessions | 8 hours |

**Goal:** Full understanding of U-Net architecture, its encoder-decoder structure, skip connections, and application to biomedical image segmentation.

---

### Resource Priority Summary

For a **complete beginner with zero knowledge**, here is the priority ranking:

1. **[MUST]** 3Blue1Brown Neural Networks series (watch first)
2. **[MUST]** Andrew Ng Deep Learning Specialization (Courses 1 & 4)
3. **[MUST]** Dive into Deep Learning (d2l.ai) -- primary textbook
4. **[RECOMMENDED]** Michael Nielsen's book -- for deeper intuition
5. **[RECOMMENDED]** Prince's Understanding Deep Learning -- for modern coverage
6. **[RECOMMENDED]** Implement everything in PyTorch as you learn
7. **[REFERENCE]** Goodfellow's Deep Learning book -- for theoretical depth
8. **[CHINESE]** 李宏毅课程 + 李沐D2L视频 + 邱锡鹏《神经网络与深度学习》

---

## Appendix: Quick Reference Links

| Resource | URL |
|----------|-----|
| Deep Learning book (Goodfellow) | https://www.deeplearningbook.org/ |
| Dive into Deep Learning | https://d2l.ai/ |
| D2L Chinese | https://zh.d2l.ai/ |
| Neural Networks and Deep Learning (Nielsen) | http://neuralnetworksanddeeplearning.com/ |
| Understanding Deep Learning (Prince) | https://udlbook.github.io/ |
| Mathematics for ML | http://mml-book.github.io/ |
| Andrew Ng DL Specialization | https://www.coursera.org/specializations/deep-learning |
| fast.ai course | https://course.fast.ai/ |
| CS231n | https://cs231n.stanford.edu/ |
| MIT 6.S191 | https://introtodeeplearning.com/ |
| 3Blue1Brown | https://www.3blue1brown.com/ |
| 邱锡鹏 NNDL | https://nndl.ai/ |
| TensorFlow Playground | https://playground.tensorflow.org/ |
| Papers With Code | https://paperswithcode.com/ |
| Kaggle | https://www.kaggle.com/ |
| U-Net paper | arXiv:1505.04597 |
