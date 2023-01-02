import torch
from torch import nn
from torch.distributions import Bernoulli
from torch.nn.utils import weight_norm as WN


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


class CELU(nn.Module):
    """ Concatenated ELU

    References
    [1]: Concatenated Rectified Linear Units, ICML, 2016, http://arxiv.org/abs/1603.05201
    [2]: https://github.com/pclucas14/pixel-cnn-pp/blob/7cb4436f062fda9b63ecc9e3b75d2c2dcb379931/utils.py#L10
    """
    def __init__(self):
        super().__init__()
        self.activation = nn.ELU()

    def forward(self, x):
        return self.activation(torch.cat([x, -x], dim=1))


class ResConvBlock(nn.Module):
    def __init__(self, in_ch=160, out_ch=160, k=3, stride=1, conv_type='vertical', skip_connection=False, dropout=0.5):
        super().__init__()
        self.stride = stride

        if conv_type == 'vertical':
            kernel = (k // 2 + 1, k)
            padding = (k // 2, k // 2, k // 2, 0)
        else:
            kernel = (k // 2 + 1, k // 2 + 1)
            padding = (k // 2, 0, k // 2, 0)

        if stride == 2:
            if skip_connection:
                resize_op = WN(nn.ConvTranspose2d(in_ch, in_ch, 2, stride=stride))
            else:
                resize_op = nn.Sequential(nn.ZeroPad2d(padding), WN(nn.Conv2d(in_ch, in_ch, kernel, stride=stride)))
        else:
            resize_op = nn.Identity()

        self.conv1 = nn.Sequential(resize_op, CELU(), nn.ZeroPad2d(padding=padding),
                                   WN(nn.Conv2d(in_ch * 2, out_ch, kernel, stride=1)))
        self.conv2 = nn.Sequential(CELU(), nn.Dropout(dropout), nn.ZeroPad2d(padding=padding),
                                   WN(nn.Conv2d(out_ch * 2, out_ch * 2, kernel, stride=1)), GatedActivation())

        if skip_connection:
            if conv_type == 'vertical':
                self.link = nn.Sequential(CELU(), WN(nn.Conv2d(in_ch * 2, out_ch, 1)))
            else:
                self.link = nn.Sequential(CELU(), WN(nn.Conv2d(in_ch * 4, out_ch, 1)))
        else:
            if conv_type == 'vertical':
                self.link = None
            else:
                self.link = nn.Sequential(CELU(), WN(nn.Conv2d(in_ch * 2, out_ch, 1)))

    def forward(self, x, skip_connection=None):
        out = self.conv1(x)

        if skip_connection is not None:
            out = out + self.link(skip_connection)

        out = self.conv2(out)

        if self.stride == 1:
            out = out + x

        return out


class DoubleResConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, skip_connection=False, dropout=0.5):
        super().__init__()
        self.v = ResConvBlock(in_ch, out_ch, k, stride=stride, conv_type='vertical',
                              skip_connection=skip_connection, dropout=dropout)
        self.h = ResConvBlock(in_ch, out_ch, k, stride=stride, conv_type='horizontal',
                              skip_connection=skip_connection, dropout=dropout)

    def forward(self, x_v, x_h, x_v_skip=None, x_h_skip=None):
        x_v = self.v(x_v, x_v_skip)
        x_h = self.h(x_h, x_v if x_h_skip is None else torch.cat([x_v, x_h_skip], dim=1))

        return x_v, x_h


class PixelCNNPP(nn.Module):
    n_stage = 3
    def __init__(self, ch=3, category=1, n_layer_per_stage=5, hidden=160, num_mixture=10, dropout=0.5):
        super().__init__()
        # 1. stem layer
        self.v_stem = nn.Sequential(nn.ZeroPad2d((3, 3, 4, 0)), WN(nn.Conv2d(ch, hidden, (4, 7))))
        self.h_stem = nn.Sequential(nn.ZeroPad2d((4, 0, 3, 0)), WN(nn.Conv2d(ch, hidden, (4, 4))))
        self.n_layer = self.n_stage * n_layer_per_stage
        self.category = category

        # 2. body layer
        down_layers = list()
        up_layers = list()

        for i in range(self.n_layer):
            down_layers.append(DoubleResConvBlock(hidden, hidden, dropout=dropout))
            up_layers.append(DoubleResConvBlock(hidden, hidden, skip_connection=True, dropout=dropout))

            if i and (i+1) != self.n_layer and (i+1) % n_layer_per_stage == 0:
                down_layers.append((DoubleResConvBlock(hidden, hidden, stride=2, dropout=dropout)))
                up_layers.append((DoubleResConvBlock(hidden, hidden, stride=2, skip_connection=True, dropout=dropout)))

        self.down_layers = nn.ModuleList(down_layers)
        self.up_layers = nn.ModuleList(up_layers)

        # 3. head layer
        if self.category > 1:
            mixture_component = 1 + ch * (2 if ch == 1 else 3)
            self.classifier = nn.Sequential(nn.ELU(), nn.Conv2d(hidden, mixture_component * num_mixture, 1))
        else:
            self.classifier = nn.Conv2d(hidden, 1, 1)

    def forward(self, x):
        # 1. stem
        skip_connection = [(self.v_stem(x)[:, :, :-1, :], self.h_stem(x)[:, :, :, :-1])]

        # 2. down
        for i, layer in enumerate(self.down_layers):
            skip_connection.append(layer(*skip_connection[-1]))

        # 3. up
        x_v, x_h = skip_connection.pop()
        for i, layer in enumerate(self.up_layers):
            x_v, x_h = layer(x_v, x_h, *skip_connection.pop())

        # 4. classifier
        if self.category > 1:
            out = self.classifier(x_h)
            sample = sample_from_mixture_logit(out) * 0.5 + 0.5

            return out, sample
        else:
            l = self.classifier(x_h)
            b, c, h, w = l.shape
            l = l.reshape(b, 1, 1, h, w)
            out = l.permute(0, 2, 1, 3, 4).squeeze(2) # remove category for binary nnl loss
            p = l.permute(0, 1, 3, 4, 2)

            p = torch.sigmoid(p)
            sample = Bernoulli(probs=p).sample().squeeze(-1)

            return out, sample


    @torch.no_grad()
    def sample(self, shape, device, *args, **kwargs):
        B, C, H, W = shape
        x = torch.full(shape, -1).to(torch.float).to(device)

        for h in range(H):
            for w in range(W):
                _, sample = self.forward(x)
                if self.category > 1:
                    x[:, :, h, w] = (sample[:, :, h, w] - 0.5) / 0.5
                else:
                    x[:, :, h, w] = sample[:, :, h, w]

        return x


def sample_from_mixture_logit(x):
    B, L, H, W = x.shape
    C = L // 10 // 3
    x = x.reshape(B, L//10, 10, H, W)

    # 1. choose mixture
    logit_prob = x[:, 0]
    noise = torch.zeros_like(logit_prob).uniform_(1e-5, 1 - 1e-5)
    noise = -torch.log(-torch.log(noise))
    mixture = (logit_prob + noise).argmax(dim=1)
    mixture = mixture[:, None, None, :, :]

    # 2. compute pixel value
    mean_and_scales = x[:, 1:].tensor_split(C, dim=1)
    if C == 1:
        mean, log_scale = mean_and_scales[0].tensor_split(2, dim=1)
        log_scale = torch.clamp(log_scale, min=-7.0)

        mean = torch.gather(mean, 2, mixture).squeeze(2)
        log_scale = torch.gather(log_scale, 2, mixture).squeeze(2)

        noise = torch.zeros_like(mean).uniform_(1e-5, 1 - 1e-5)
        sample = mean + torch.exp(log_scale) * (torch.log(noise) - torch.log(1-noise))
        sample = torch.clamp(sample, min=-1.0, max=1.0)

    else:
        (m0, m1, m2), (c0, c1, c2), log_scale = [channel.tensor_split(3, dim=1) for channel in mean_and_scales]
        log_scale = torch.clamp(torch.cat(log_scale, dim=1), min=-7.0)
        log_scale = torch.gather(log_scale, 2, mixture).squeeze(2)

        noise = torch.zeros_like(log_scale).uniform_(1e-5, 1 - 1e-5)
        noise = torch.exp(log_scale) * (torch.log(noise) - torch.log(1-noise))
        m0, m1, m2 = [torch.gather(m + noise, 2, mixture).squeeze(2) for m in (m0, m1, m2)]
        c0, c1, c2 = [torch.gather(torch.tanh(c), 2, mixture).squeeze(2) for c in (c0, c1, c2)]

        m0 = torch.clamp(m0, min=-1.0, max=1.0)
        m1 = torch.clamp(m1 + m0 * c0, min=-1.0, max=1.0)
        m2 = torch.clamp(m2 + m0 * c1 + m1 * c2, min=-1.0, max=1.0)
        sample = torch.cat([m0, m1, m2], dim=1)

    return sample


if __name__ == '__main__':
    x = torch.rand([2, 3, 28, 28])
    f = PixelCNNPP(ch=3)
    y, sample = f(x)
    img = f.sample((2, 3, 28, 28), 'cpu', (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    print(y.shape)