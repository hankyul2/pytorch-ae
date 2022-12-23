# Pytorch AutoEncoder

This repository contains pytorch auto encoder examples.

Most code are copy & pasted version from [pytorch-generative](https://github.com/EugenHotaj/pytorch-generative).



---

## Table of contents

1. [Tutorial](#tutorial)
2. [Experiment result](#experiment-result)
2. [Generated Image](#generated-image)
3. [Reference](#reference)



## :seedling:Tutorial

1. git clone this repo

   ```bash
   git clone https://github.com/hankyul2/pytorch-ae.git
   ```

2. train your model

   ```bash
   python3 train.py -m nade
   ```

   

## :four_leaf_clover:Experiment Result

Negative Log Likelihood (NLL) loss on Binarized MNIST dataset.

| Method              | Command                    | Binarized MNIST | Pretrained model                                   |
| ------------------- | -------------------------- | --------------- | -------------------------------------------------- |
| NADE[^1]            | `python3 train.py -m nade` | 86.1            | [[code](pae/model/nade.py)] [[weight]()] [[log]()] |
| PixelRNN[^2]        |                            |                 |                                                    |
| PixelCNN[^3]        |                            |                 |                                                    |
| PixelSnail[^4]      |                            |                 |                                                    |
| AE[^5]              |                            |                 |                                                    |
| VAE[^6]             |                            |                 |                                                    |
| Categorical-VAE[^7] |                            |                 |                                                    |
| VQ-VAE[^8]          |                            |                 |                                                    |
| VQ-VAE-v2[^9]       |                            |                 |                                                    |
| dVAE[^10]           |                            |                 |                                                    |
| CDM[^11]            |                            |                 |                                                    |



## :framed_picture:Generated Image

| Method              | 25 Epoch                                                     | 50 Epoch                                                     |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| NADE[^1]            | ![val_25](https://user-images.githubusercontent.com/31476895/209291712-430abc71-3b6e-4963-81b2-df7b062ebfa0.jpg) | ![hi](https://user-images.githubusercontent.com/31476895/209291150-95a55130-4624-4004-8ddc-6d5b57ed7e92.jpg) |
| PixelRNN[^2]        |                                                              |                                                              |
| PixelCNN[^3]        |                                                              |                                                              |
| PixelSnail[^4]      |                                                              |                                                              |
| AE[^5]              |                                                              |                                                              |
| VAE[^6]             |                                                              |                                                              |
| Categorical-VAE[^7] |                                                              |                                                              |
| VQ-VAE[^8]          |                                                              |                                                              |
| VQ-VAE-v2[^9]       |                                                              |                                                              |
| dVAE[^10]           |                                                              |                                                              |
| CDM[^11]            |                                                              |                                                              |



### :maple_leaf:Reference

[^1]: **NADE:** "Neural Autoregressive Distribution Estimation", JMLR, 2016 [[paper](https://www.jmlr.org/papers/volume17/16-272/16-272.pdf)] [[code](pae/model/nade.py)]
[^2]: **PixelRNN**: "pixel recurrent neural networks", PMLR, 2016 [[paper](http://proceedings.mlr.press/v48/oord16.pdf)] [not available now]
[^3]: **PixelCNN**: "Conditional Image Generation with PixelCNN Decoders", NIPS, 2016 [[paper](https://proceedings.neurips.cc/paper/2016/file/b1301141feffabac455e1f90a7de2054-Paper.pdf)] [not available now]
[^4]: **PixelSnail**: "Cascaded Diffusion Models for High Fidelity Image Generation", PMLR, 2018 [[paper](http://proceedings.mlr.press/v80/chen18h/chen18h.pdf)] [not available now]
[^5]: **AE**: "Autoencoders, Unsupervised Learning, and Deep Architectures", JMLR, 2012 [[paper](https://proceedings.mlr.press/v27/baldi12a/baldi12a.pdf)] [not available now]
[^6]: **VAE**: "Auto-Encoding Variational Bayes", ArXiv, 2013 [[paper](https://arxiv.org/pdf/1312.6114.pdf)] [not available now]
[^7]: **Categorical-VAE**: "Categorical Reparameterization with Gumbel-Softmax", ICLR, 2017 [[paper](https://arxiv.org/pdf/1611.01144.pdf)] [not available now]
[^8]: **VQ-VAE**: "Neural Discrete Representation Learning", NIPS, 2017 [[paper](https://proceedings.neurips.cc/paper/2017/file/7a98af17e63a0ac09ce2e96d03992fbc-Paper.pdf)] [not available now]
[^9]: **VQ-VAE-v2**: "Generating Diverse High-Fidelity Images with VQ-VAE-2", NIPS, 2019 [[paper](https://proceedings.neurips.cc/paper/2019/file/5f8e2fa1718d1bbcadf1cd9c7a54fb8c-Paper.pdf)] [not available now]
[^10]: **dVAE**: "Zero-Shot Text-to-Image Generation", PMLR, 2021 [[paper](http://proceedings.mlr.press/v139/ramesh21a/ramesh21a.pdf)] [not available now]
[^11]: **CDM**: "Cascaded Diffusion Models for High Fidelity Image Generation", JMLR, 2022 [[paper](https://www.jmlr.org/papers/volume23/21-0635/21-0635.pdf)] [not available now]

