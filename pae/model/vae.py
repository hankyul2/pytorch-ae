import math

import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical


def reparam_trick(mean, std):
    return mean + (0.5 * std).exp() * torch.randn_like(std)


def compute_kl_loss(mean, std):
    return -0.5 * (1 + std - std.exp() - mean ** 2).sum(dim=[1, 2, 3]).mean()


class BasicBlock(nn.Module):

    def __init__(self, hidden, kernel=3):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden, hidden, kernel, padding=1, bias=True)
        # if you put bn, loss won't go lower. But sampling image looks quite good.
        # self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=1, bias=True)
        # self.bn2 = nn.BatchNorm2d(hidden)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out + x)

        return out


class VAE(nn.Module):
    def __init__(self, ch=1, category=1, layer=[3, 3, 3], hidden=64, latent=16, img_size=28):
        super().__init__()
        down_hidden = [ch] + [hidden] * len(layer)
        up_hidden = [hidden] + [hidden] * len(layer)

        size_list = list()
        for i in range(len(layer)):
            size_list.append(img_size)
            img_size = math.ceil(img_size / 2)
        size_list.reverse()

        down_layer, up_layer = list(), list()
        for stage in range(len(layer)):
            op = p = int(size_list[stage] % 2 == 1)
            down_layer.append(nn.Sequential(nn.Conv2d(down_hidden[stage], down_hidden[stage+1], 3, 2, 1), nn.ReLU()))
            down_layer.extend([BasicBlock(down_hidden[stage+1]) for _ in range(layer[stage])])
            up_layer.extend([BasicBlock(up_hidden[stage]) for _ in range(layer[stage])])
            up_layer.append(nn.Sequential(nn.ConvTranspose2d(up_hidden[stage], up_hidden[stage+1], 2, 2, p, op), nn.ReLU()))
        down_layer.append(nn.Conv2d(hidden, latent * 2, 1))
        up_layer.insert(0, nn.Conv2d(latent, hidden, 1))
        up_layer.append(nn.Conv2d(hidden, ch * category, 1))

        self.enc = nn.Sequential(*down_layer)
        self.dec = nn.Sequential(*up_layer)
        self.latent_dim = latent
        self.latent_size = math.ceil(size_list[0] / 2)
        self.c = ch
        self.category = category

    def forward(self, x):
        mean, std = self.enc(x).tensor_split(2, dim=1)
        z = reparam_trick(mean, std)
        kl_loss = compute_kl_loss(mean, std)
        l = self.dec(z)

        b, c, h, w = l.shape
        l = l.reshape(b, self.c, self.category, h, w)
        out = l.permute(0, 2, 1, 3, 4).squeeze(2) # remove category for binary nnl loss
        p = l.permute(0, 1, 3, 4, 2)

        if self.category == 1:
            p = torch.sigmoid(p)
            # sample = Bernoulli(probs=p).sample().squeeze(-1)
            sample = (p > 0.5).float().squeeze(-1)
        else:
            p = torch.softmax(p, dim=-1)
            sample = Categorical(probs=p).sample().squeeze(-1) / 255.0

        return out, sample, kl_loss

    def sample(self, shape, device, *args, **kwargs):
        shape = (shape[0], self.latent_dim, self.latent_size, self.latent_size)
        z = torch.randn(shape).to(device)
        l = self.dec(z)

        b, c, h, w = l.shape
        l = l.reshape(b, self.c, self.category, h, w)
        p = l.permute(0, 1, 3, 4, 2)

        if self.category == 1:
            p = torch.sigmoid(p)
            # sample = Bernoulli(probs=p).sample().squeeze(-1)
            sample = (p > 0.5).float().squeeze(-1)
        else:
            p = torch.softmax(p, dim=-1)
            sample = Categorical(probs=p).sample().squeeze(-1) / 255.0

        return sample


if __name__ == '__main__':
    x = torch.rand(2, 1, 28, 28)
    f = VAE()
    y, sample, loss = f(x)
    random = f.sample([2, 1, 28, 28], 'cpu')
    print(random.shape)
    print(y.shape)