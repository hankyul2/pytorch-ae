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
    """
    def __init__(self, ch=1, category=1, hidden=16, layer=15):
        super().__init__()
        self.c = ch
        self.category = category
        self.h = hidden * ch * 2
        self.l = layer
        self.stem = MaskConv2d(
            in_channels=self.c, out_channels=self.h, kernel_size=7, padding=3,
            spatial_mask='A', channel_split=self.c
        )
        self.layers = nn.ModuleList([ResidualBlock(self.c, self.h) for _ in range(self.l)])
        self.head = nn.Sequential(
            MaskConv2d(in_channels=self.h, out_channels=self.h, kernel_size=1, channel_split=self.c),
            nn.ReLU(inplace=True),
            MaskConv2d(in_channels=self.h, out_channels=self.c * category, kernel_size=1, channel_split=self.c),
        )
        self.init_weights()

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        x = self.head(x)
        x = torch.sigmoid(x)

        b, c, h, w = x.shape
        p = x.reshape(b, self.c, self.category, h, w).permute(0, 1, 3, 4, 2)

        if self.c == 1:
            sample = Bernoulli(probs=p).sample().squeeze(-1)
        else:
            sample = Categorical(probs=p).sample().squeeze(-1)

        return x, sample

    @torch.no_grad()
    def sample(self, shape, device):
        B, C, H, W = shape
        x = torch.full(shape, -1).to(torch.float).to(device)

        for h in range(H):
            for w in range(W):
                _, sample = self.forward(x)
                x[:, :, h, w] = sample[:, :, h, w]

        return x


# if __name__ == '__main__':
#     x = torch.rand(2, 3, 28, 28)
#     f = PixelCNN(ch=3, category=256)
#     y, sample = f(x)
#     print(y.shape)
#     print(sample.shape)
#     img = f.sample((2, 3, 28, 28), 'cpu')
#     print(img.shape)
