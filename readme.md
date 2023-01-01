# Pytorch AutoEncoder

This repository contains pytorch auto encoder examples.

Most code are copy & pasted version from [pytorch-generative](https://github.com/EugenHotaj/pytorch-generative).



---

## Table of contents

1. [Tutorial](#seedlingtutorial)
2. [Experiment result](#four_leaf_cloverexperiment-result)
2. [Generated Image](#framed_picturegenerated-image)
3. [Reference](#maple_leafreference)



## :seedling:Tutorial

1. Clone this repo.

   ```bash
   git clone https://github.com/hankyul2/pytorch-ae.git
   ```

2. Train your model. 

   ```bash
   python3 train.py -m nade
   ```

3. Use trained model in your way. We provide code snippet for how to sample from model.

   ```python
   import torch
   from torchvision.utils import save_image
   from pae.model import NADE
   
   model = NADE()
   model.load_state_dict(torch.load('your_checkpont.pth'))
   generated_img = model.sample(16, 'cpu').reshape(16, 1, 28, 28)
   save_image(generated_img, 'generated_by_NADE.jpg')
   ```

   

## :four_leaf_clover:Experiment Result

Negative Log Likelihood (NLL) loss on Binarized MNIST dataset.

| Method                   | Command                        | NLL  | Pretrained model                                             |
| ------------------------ | ------------------------------ | ---- | ------------------------------------------------------------ |
| [NADE](docs/nade.md)[^1] | `python3 train.py -m NADE`     | 84.0 | [[code](pae/model/nade.py)] [[weight](https://github.com/hankyul2/pytorch-ae/releases/download/v0.0.1/NADE_weight.pth)] [[log](https://github.com/hankyul2/pytorch-ae/releases/download/v0.0.1/NADE_log.txt)] |
| MADE[^2]                 | `python3 train.py -m MADE`     | 83.8 | [[code](pae/model/made.py)] [[weight](https://github.com/hankyul2/pytorch-ae/releases/download/v0.0.1/MADE_weight.pth)] [[log](https://github.com/hankyul2/pytorch-ae/releases/download/v0.0.1/MADE_log.txt)] |
| PixelCNN[^3]             | `python3 train.py -m PixelCNN` | 81.7 | [[code](pae/model/pixel_cnn.py)] [[weight](https://github.com/hankyul2/pytorch-ae/releases/download/v0.0.1/PixelCNN_weight.pth)] [[log](https://github.com/hankyul2/pytorch-ae/releases/download/v0.0.1/PixelCNN_log.txt)] |
| Gated PixelCNN[^4]       |                                |      |                                                              |
| PixelCNN++[^5]           |                                |      |                                                              |
| PixelSnail[^6]           |                                |      |                                                              |
| AE[^7]                   |                                |      |                                                              |
| VAE[^8]                  |                                |      |                                                              |
| Categorical-VAE[^9]      |                                |      |                                                              |
| VQ-VAE[^10]              |                                |      |                                                              |
| VQ-VAE-v2[^11]           |                                |      |                                                              |
| dVAE[^12]                |                                |      |                                                              |
| CDM[^13]                 |                                |      |                                                              |
|                          |                                |      |                                                              |

Issues

1. MADE: we could not utilize any agnostic training tricks (order, connectivity) proposed in the original paper.
2. PixelCNN: we could not understand and implement the PixelRNN model which is mainly discussed in the original paper.



## :framed_picture:Generated Image

| Method              | Reconstructed Image                                          | Randomly Sampled Image                                       |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| NADE[^1]            | ![val_49](https://user-images.githubusercontent.com/31476895/209418791-214369c5-c793-46ee-867b-d7e95fc70202.jpg) | ![sample_49](https://user-images.githubusercontent.com/31476895/209418789-2f5a7ecd-f6f4-4caa-adf1-7d9dd8795705.jpg) |
| MADE[^2]            | ![val](https://user-images.githubusercontent.com/31476895/209778705-d0cf8120-491a-438c-9eff-d6483fc8d54e.jpg) | ![sample](https://user-images.githubusercontent.com/31476895/209778702-cbbf1da1-216e-4810-ad3f-823128880728.jpg) |
| PixelCNN[^3]        | ![val](https://user-images.githubusercontent.com/31476895/209778793-e441aded-0385-41dd-9d5a-43c5aab0f4e1.jpg) | ![sample](https://user-images.githubusercontent.com/31476895/209778786-43bf7dff-2b91-4d28-a976-b312961a17fe.jpg) |
| Gated PixelCNN[^4]  |                                                              |                                                              |
| PixelCNN++[^5]      |                                                              |                                                              |
| PixelSnail[^6]      |                                                              |                                                              |
| AE[^7]              |                                                              |                                                              |
| VAE[^8]             |                                                              |                                                              |
| Categorical-VAE[^9] |                                                              |                                                              |
| VQ-VAE[^10]         |                                                              |                                                              |
| VQ-VAE-v2[^11]      |                                                              |                                                              |
| dVAE[^12]           |                                                              |                                                              |
| CDM[^13]            |                                                              |                                                              |



### :maple_leaf:Reference

[^1]: **NADE:** "Neural Autoregressive Distribution Estimation", JMLR, 2016 [[paper](https://www.jmlr.org/papers/volume17/16-272/16-272.pdf)] 
[^2]: **MADE**: "Masked Autoencoder for Distribution Estimation", PMLR, 2015 [[paper](http://proceedings.mlr.press/v37/germain15.pdf)]
[^3]: **PixelCNN**: "pixel recurrent neural networks", PMLR, 2016 [[paper](http://proceedings.mlr.press/v48/oord16.pdf)] 
[^4]: **Gated PixelCNN**: "Conditional Image Generation with PixelCNN Decoders", NIPS, 2016 [[paper](https://proceedings.neurips.cc/paper/2016/file/b1301141feffabac455e1f90a7de2054-Paper.pdf)]
[^5]: **PixelCNN++**: "Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications", ICLR, 2017 [[paper](https://proceedings.neurips.cc/paper/2016/file/b1301141feffabac455e1f90a7de2054-Paper.pdf)]
[^6]: **PixelSnail**: "Cascaded Diffusion Models for High Fidelity Image Generation", PMLR, 2018 [[paper](http://proceedings.mlr.press/v80/chen18h/chen18h.pdf)][not available now]
[^7]: **AE**: "Autoencoders, Unsupervised Learning, and Deep Architectures", JMLR, 2012 [[paper](https://proceedings.mlr.press/v27/baldi12a/baldi12a.pdf)] 
[^8]: **VAE**: "Auto-Encoding Variational Bayes", ArXiv, 2013 [[paper](https://arxiv.org/pdf/1312.6114.pdf)]
[^9]: **Categorical-VAE**: "Categorical Reparameterization with Gumbel-Softmax", ICLR, 2017 [[paper](https://arxiv.org/pdf/1611.01144.pdf)]
[^10]: **VQ-VAE**: "Neural Discrete Representation Learning", NIPS, 2017 [[paper](https://proceedings.neurips.cc/paper/2017/file/7a98af17e63a0ac09ce2e96d03992fbc-Paper.pdf)]
[^11]: **VQ-VAE-v2**: "Generating Diverse High-Fidelity Images with VQ-VAE-2", NIPS, 2019 [[paper](https://proceedings.neurips.cc/paper/2019/file/5f8e2fa1718d1bbcadf1cd9c7a54fb8c-Paper.pdf)] 
[^12]: **dVAE**: "Zero-Shot Text-to-Image Generation", PMLR, 2021 [[paper](http://proceedings.mlr.press/v139/ramesh21a/ramesh21a.pdf)] 
[^13]: **CDM**: "Cascaded Diffusion Models for High Fidelity Image Generation", JMLR, 2022 [[paper](https://www.jmlr.org/papers/volume23/21-0635/21-0635.pdf)]

