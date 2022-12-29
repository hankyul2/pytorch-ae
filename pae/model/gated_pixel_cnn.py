import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical


class MaskConv2d(nn.Conv2d):
    def __init__(self, *args, spatial_mask:str=None, channel_split:int=1, **kwargs):
        """
        :arg
            spatial_mask: type of spatial mask (A: exclude center, B: include center)
            channel_split: disable to use G, B information in predicting R value.
        """
        super().__init__(*args, **kwargs)
        i, o, h, w = self.weight.shape
        mask = torch.ones((i, o, h, w))

        # spatial mask
        if spatial_mask:
            mask[:, :, h//2+1:, :] = 0
            mask[:, :, h//2:, w//2 + (0 if spatial_mask == 'A' else 1):] = 0

        # channel mask
        for s in range(1, channel_split):
            mask[i//channel_split * (s+1):, :o//channel_split * (s+1)] = 0

        self.register_buffer("mask", mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class GatedActivation(nn.Module):
    def __init__(self, channel_split=1):
        super().__init__()
        self.channel_split = channel_split

    def split_channel(self, dim):
        channels = list([0])
        for i in range(self.channel_split):
            split = (dim // self.channel_split) * (i+1)
            split += dim % self.channel_split if i == self.channel_split - 1 else 0
            channels.append(split)
        return channels

    def forward(self, x):
        result = list()
        split = self.split_channel(x.size(1))
        for i in range(self.channel_split):
            start = split[i]
            end = split[i+1]
            size = end - start
            y1 = torch.tanh(x[:, start:start+size//2])
            y2 = torch.sigmoid(x[:, start+size//2:end])
            out = y1 * y2
            result.append(out)

        return torch.cat(result, dim=1)


class GatedConvBlock(nn.Module):
    def __init__(self, c, in_ch, out_ch, k=3, mask_center=False):
        super().__init__()
        self.mask_center = int(mask_center)
        self.v = MaskConv2d(in_ch, out_ch * 2, kernel_size=(k//2+1, k), padding=(k//2, k//2), channel_split=c)
        self.l = MaskConv2d(out_ch * 2, out_ch * 2, kernel_size=1, padding=0, channel_split=c)
        self.h = MaskConv2d(in_ch, out_ch * 2, kernel_size=(1, k//2+1),
                            padding=(0, k//2 + self.mask_center), channel_split=c)
        self.out = MaskConv2d(out_ch, out_ch, kernel_size=1, padding=0, channel_split=c)
        self.act = GatedActivation(channel_split=c)

    def forward(self, x_v, x_h):
        B, C, H, W = x_v.shape
        v = self.v(x_v)[:, :, :H, :]
        h = self.h(x_h)[:, :, :, :W] + self.l(F.pad(v, (0, 0, 1, 0))[:, :, :H, :])
        v = self.act(v)
        h = self.out(self.act(h))

        if not self.mask_center:
            h = h + x_h

        return v, h


class GatedPixelCNN(nn.Module):
    def __init__(self, ch=1, category=1, hidden=144, dataset='MNIST', layer=10):
        super().__init__()
        self.c = ch
        self.category = category
        self.h = hidden
        self.l = layer
        self.head_dim = 32 if dataset == 'MNIST' else 1024
        self.stem = GatedConvBlock(self.c, self.c, self.h, 7, mask_center=True)
        self.layers = nn.ModuleList([GatedConvBlock(self.c, self.h, self.h) for _ in range(self.l)])
        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            MaskConv2d(in_channels=self.h, out_channels=self.head_dim, kernel_size=1, channel_split=self.c),
            nn.ReLU(inplace=True),
            MaskConv2d(in_channels=self.head_dim, out_channels=self.c * category, kernel_size=1, channel_split=self.c),
        )

    def forward(self, x):
        v, h = self.stem(x, x)
        for layer in self.layers:
            v, h = layer(v, h)
        l = self.head(h)

        b, c, h, w = l.shape
        l = l.reshape(b, self.c, self.category, h, w)
        out = l.permute(0, 2, 1, 3, 4).squeeze(2) # remove category for binary nnl loss
        p = l.permute(0, 1, 3, 4, 2)

        if self.category == 1:
            p = torch.sigmoid(p)
            sample = Bernoulli(probs=p).sample().squeeze(-1)
        else:
            p = torch.softmax(p, dim=-1)
            sample = Categorical(probs=p).sample().squeeze(-1) / 255.0

        return out, sample

    @torch.no_grad()
    def sample(self, shape, device, mean, std):
        B, C, H, W = shape
        x = torch.full(shape, -1).to(torch.float).to(device)
        mean = torch.tensor(mean).to(device)
        std = torch.tensor(std).to(device)

        for h in range(H):
            for w in range(W):
                _, sample = self.forward(x)
                x[:, :, h, w] = (sample[:, :, h, w] - mean) / std

        return x * std.reshape(1, C, 1, 1) + mean.reshape(1, C, 1, 1)


if __name__ == '__main__':
    x = torch.rand(2, 3, 28, 28)
    f = GatedPixelCNN(ch=3, category=256, layer=1)
    logit, sample = f(x)
    img = f.sample((2, 3, 28, 28), 'cpu', (0.5, 0.5, 0.5), (0.1, 0.1, 0.1))

    assert list(logit.shape) == [2, 1, 28, 28]
    print("[TEST] logit shape test success")
    assert list(sample.shape) == [2, 1, 28, 28]
    print("[TEST] sample shape test success")
    assert list(img.shape) == [2, 1, 28, 28]
    print("[TEST] sample shape test2 success")