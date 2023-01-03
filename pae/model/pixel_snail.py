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

        self.register_parameter("mask", nn.Parameter(mask, requires_grad=False))

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class GatedActivation(nn.Module):
    """ Gated Activation

    Gated activation is originally proposed in Gated PixelCNN [1].
    Gated activation is modified in PixelCNN++ [2].
    We follow PixelCNN++ [2] version.

    References
    [1]: https://arxiv.org/pdf/1606.05328.pdf
    [2]: https://github.com/pclucas14/pixel-cnn-pp/blob/master/model.py
    """
    def __init__(self):
        super().__init__()
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = x.tensor_split(2, dim=1)
        gate = self.activation(x2)
        return x1 * gate


class BasicBlock(nn.Module):
    def __init__(self, hidden, kernel=2):
        super().__init__()
        self.elu = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(hidden, hidden, kernel, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden * 2, kernel, padding=1)
        self.gated_act = GatedActivation()

    def forward(self, x):
        out = self.elu(x)
        out = self.conv1(out)[:, :, :x.size(2), :x.size(3)]
        out = self.elu(out)
        out = self.conv2(out)[:, :, :x.size(2), :x.size(3)]
        out = self.gated_act(out)

        return x + out


def image_positional_encoding(shape):
    """Generates positional encodings for 2d images.
    The positional encoding is a Tensor of shape (N, 2, H, W) of (x, y) pixel
    coordinates scaled to be between -.5 and .5.
    Args:
        shape: NCHW shape of image for which to generate positional encodings.
    Returns:
        The positional encodings.
    """
    n, c, h, w = shape
    zeros = torch.zeros(n, 1, h, w)
    return torch.cat((
        (torch.arange(-0.5, 0.5, 1 / h)[None, None, :, None] + zeros),
        (torch.arange(-0.5, 0.5, 1 / w)[None, None, None, :] + zeros),
    ), dim=1)


def get_causal_mask(size):
    return torch.tril(torch.ones((size, size)), diagonal=0)


class MaskAttention(nn.Module):
    def __init__(self, in_ch, out_ch, head, qk_dim, v_dim):
        super().__init__()
        self.h = head
        self.div = (qk_dim // head) ** -0.5

        self.dims = [qk_dim, qk_dim, v_dim]
        self.qkv = nn.Conv2d(in_ch, qk_dim * 2 + v_dim, 1)
        self.proj = nn.Conv2d(v_dim, out_ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = [s.reshape(B, self.h, s.size(1)//self.h, H * W).permute(0, 1, 3, 2)
                   for s in self.qkv(x).split(self.dims, dim=1)]

        mask = get_causal_mask(H * W).reshape(1, 1, H*W, H*W).expand(B, self.h, -1, -1).to(x.device)
        attn = (q @ k.transpose(-1, -2) * self.div).masked_fill(mask == 0, -torch.inf)
        attn = F.softmax(attn, dim=-1).masked_fill(mask==0, 0)

        out = (attn @ v).permute(0, 1, 3, 2).reshape(B, -1, H, W)
        out = self.proj(out)

        return out


class SnailBlock(nn.Module):
    def __init__(self, hidden, layer, head=1, qk_dim=16, v_dim=128):
        super().__init__()
        self.resblk = nn.Sequential(*[BasicBlock(hidden) for _ in range(layer)])
        self.attn = MaskAttention(hidden + 2, hidden, head, qk_dim, v_dim)

        self.act = nn.ELU(inplace=True)
        self.out_conv = nn.Conv2d(hidden, hidden, 1)
        self.out_attn = nn.Conv2d(hidden, hidden, 1)
        self.out_proj = nn.Conv2d(hidden, hidden, 1)


    def forward(self, x, pos):
        conv = self.resblk(x)

        attn = torch.cat([conv, pos], dim=1)
        attn = self.attn(attn)

        conv = self.act(self.out_conv(self.act(conv)))
        attn = self.act(self.out_attn(self.act(attn)))
        out = self.act(self.out_proj(self.act(conv + attn)))

        return out


class PixelSnail(nn.Module):
    def __init__(self, ch=1, category=1, hidden=64, dataset='MNIST', layer=2, stage=8, head=4, qk_dim=16, v_dim=32):
        super().__init__()
        self.c = ch
        self.category = category
        self.stem = MaskConv2d(in_channels=ch, out_channels=hidden, kernel_size=3, padding=1, spatial_mask='A')
        self.layers = nn.ModuleList(SnailBlock(hidden, layer, head, qk_dim, v_dim) for _ in range(stage))
        self.fc = nn.Sequential(nn.ELU(inplace=True), nn.Conv2d(hidden, ch, 1))

    def forward(self, x):
        pos = image_positional_encoding(x.shape).to(x.device)
        out = self.stem(x)

        for layer in self.layers:
            out = layer(out, pos) + out

        l = self.fc(out)

        b, c, h, w = l.shape
        l = l.reshape(b, self.c, self.category, h, w)
        out = l.permute(0, 2, 1, 3, 4).squeeze(2)  # remove category for binary nnl loss
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
                for c in range(C):
                    _, sample = self.forward(x)
                    x[:, c, h, w] = (sample[:, c, h, w] - mean[c]) / std[c]

        return x * std.reshape(1, C, 1, 1) + mean.reshape(1, C, 1, 1)


if __name__ == '__main__':
    x = torch.rand(2, 1, 28, 28)
    f = PixelSnail()
    y, sample = f(x)
    print(y.shape)