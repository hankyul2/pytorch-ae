import torch
from torch import nn
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


class ResidualBlock(nn.Module):
    def __init__(self, c, h):
        super().__init__()
        self.conv1 = MaskConv2d(in_channels=h, out_channels=h//2, kernel_size=1, channel_split=c)
        self.conv2 = MaskConv2d(
            in_channels=h//2, out_channels=h//2, kernel_size=3, padding=1,
            spatial_mask='B', channel_split=c
        )
        self.conv3 = MaskConv2d(in_channels=h//2, out_channels=h, kernel_size=1, channel_split=c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        org = x

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        return self.relu(x + org)


class PixelCNN(nn.Module):
    """
    changes: relu(f(x) + x)
    note: don't use init (e.g. he, xavier)
    """
    def __init__(self, ch=1, category=1, hidden=128, dataset='MNIST', layer=15):
        super().__init__()
        self.c = ch
        self.category = category
        self.h = hidden * 2
        self.l = layer
        self.head_dim = 32 if dataset == 'MNIST' else 1024
        self.stem = nn.Sequential(
            MaskConv2d(
                in_channels=self.c, out_channels=self.h, kernel_size=7, padding=3,
                spatial_mask='A', channel_split=self.c
            ),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.ModuleList([ResidualBlock(self.c, self.h) for _ in range(self.l)])
        self.head = nn.Sequential(
            MaskConv2d(in_channels=self.h, out_channels=self.head_dim, kernel_size=1, channel_split=self.c),
            nn.ReLU(inplace=True),
            MaskConv2d(in_channels=self.head_dim, out_channels=self.c * category, kernel_size=1, channel_split=self.c),
        )

    def forward(self, x):
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        l = self.head(x)

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
    x = torch.rand(2, 1, 28, 28)
    f = PixelCNN(ch=1, category=1)
    logit, sample = f(x)
    img = f.sample((2, 1, 28, 28), 'cpu')

    assert list(logit.shape) == [2, 1, 28, 28]
    print("[TEST] logit shape test success")
    assert list(sample.shape) == [2, 1, 28, 28]
    print("[TEST] sample shape test success")
    assert list(img.shape) == [2, 1, 28, 28]
    print("[TEST] sample shape test2 success")
