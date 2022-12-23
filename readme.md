# Pytorch AutoEncoder

This repository contains pytorch auto encoder examples.

Most code are copy & pasted version from [pytorch-generative](https://github.com/EugenHotaj/pytorch-generative).



---

## Table of contents

1. [Tutorial](#tutorial)
2. [Experiment result](#experiment-result)
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

|          | Binarized MNIST | Pretrained model                         |
| -------- | --------------- | ---------------------------------------- |
| NADE[^1] |                 | [[code](pae/model/nade.py)] [[weight]()] |
|          |                 |                                          |
|          |                 |                                          |



### :maple_leaf:Reference

[^1]: **NADE:** "Neural Autoregressive Distribution Estimation", MLR, 2016 [[paper](https://www.jmlr.org/papers/volume17/16-272/16-272.pdf)] [[code](pae/model/nade.py)]

