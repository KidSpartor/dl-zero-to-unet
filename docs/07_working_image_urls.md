# Working Image URLs for Deep Learning Educational Diagrams

All URLs verified to return HTTP 200 with image content type on 2026-05-28.

| # | Image | Working URL | Source |
|---|-------|------------|--------|
| 1 | AI/ML/DL hierarchy diagram (nested circles) | https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/AI-ML-DL.svg/500px-AI-ML-DL.svg.png | Wikimedia Commons (Wikipedia: Deep learning) |
| 2 | Artificial neuron / perceptron model diagram | https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/ArtificialNeuronModel_english.png/960px-ArtificialNeuronModel_english.png | Wikimedia Commons (Wikipedia: Backpropagation, Perceptron) |
| 3 | XOR problem / perceptron network diagram | https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/XOR_perceptron_net.png/500px-XOR_perceptron_net.png | Wikimedia Commons (Wikipedia: Feedforward neural network) |
| 4 | MLP / multilayer perceptron architecture | https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/500px-Colored_neural_network.svg.png | Wikimedia Commons (Wikipedia: Artificial neural network) |
| 5 | ReLU activation function plot (with GELU) | https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/ReLU_and_GELU.svg/1280px-ReLU_and_GELU.svg.png | Wikimedia Commons (Wikipedia: Rectifier) |
| 6 | 3D loss landscape visualization (ResNet-56) | https://raw.githubusercontent.com/tomgoldstein/loss-landscape/master/doc/images/resnet56_small.jpg | GitHub: tomgoldstein/loss-landscape (Li et al., NeurIPS 2018) |
| 7 | Backpropagation / computation graph (data flow) | https://cs231n.github.io/assets/dataflow.jpeg | CS231n Course Notes (Stanford) |
| 8 | LeNet-5 CNN architecture diagram | https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/LeNet-5_architecture.svg/960px-LeNet-5_architecture.svg.png | Wikimedia Commons (Wikipedia: LeNet) |
| 9 | CNN feature hierarchy / learned filters | https://cs231n.github.io/assets/cnnvis/filt1.jpeg | CS231n Course Notes (Stanford) |
| 10 | Encoder-decoder architecture diagram (Seq2Seq) | https://upload.wikimedia.org/wikipedia/commons/thumb/2/24/Seq2seq-training.svg/500px-Seq2seq-training.svg.png | Wikimedia Commons (Wikipedia: Seq2seq) |
| 11 | U-Net architecture diagram (original) | https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png | Ronneberger et al. (University of Freiburg) |
| 12 | IoU / Intersection over Union diagram | https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Intersection_over_Union_-_poor%2C_good_and_excellent_score.png/500px-Intersection_over_Union_-_poor%2C_good_and_excellent_score.png | Wikimedia Commons (Wikipedia: Intersection over union) |

## Notes

- **Wikimedia Commons** images use the `/thumb/` path for reasonable file sizes. Remove `/thumb/.../500px-` from the URL to get the original full-resolution version.
- **CS231n** images are from Stanford's Convolutional Neural Networks for Visual Recognition course (cs231n.github.io).
- **Loss landscape** image is from the official repository accompanying "Visualizing the Loss Landscape of Neural Nets" (Li et al., NeurIPS 2018).
- **U-Net** image is from the original paper authors' page at the University of Freiburg.
- Some Wikimedia full-resolution URLs may return 429 (rate limited) under heavy load. The `/thumb/` versions are more reliable.
