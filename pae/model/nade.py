import torch
from torch import nn
from torch.distributions import Bernoulli
from torch.nn import functional as F


class NADE(nn.Module):
    """Neural Autoregressive Density Estimation (NADE)

    Limits: this model only support binary image (single channel & binary (0, 1) pixel)
    """
    def __init__(self, d=784, h=500):
        super().__init__()
        self.d = d

        self.W = nn.Parameter(torch.rand(h, d))
        self.c = nn.Parameter(torch.rand(h))
        self.V = nn.Parameter(torch.rand(d, h))
        self.b = nn.Parameter(torch.rand(d))

        # it is really important to initialize weight by normal distribution.
        nn.init.kaiming_normal_(self.W)
        nn.init.kaiming_normal_(self.V)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(shape[0], -1)
        l = list()
        x_sample = list()

        a_d = self.c.expand(shape[0], -1) # B x H
        for d in range(self.d):
            h_d = F.relu(a_d, inplace=False) # B x H
            l_d = h_d @ self.V[d:d + 1].t() + self.b[d]
            p_d = torch.sigmoid(l_d) # B x 1

            x_org = x[:, d:d+1]
            x_new = Bernoulli(probs=p_d).sample().to(x_org.dtype)
            need_to_sample = x_org == -1
            x_d = torch.where(need_to_sample, x_new, x_org)
            a_d = x_d @ self.W[:, d:d+1].t() + a_d

            l.append(l_d)
            x_sample.append(x_new)

        l = torch.concat(l, dim=-1).reshape(*shape)
        x_sample = torch.concat(x_sample, dim=-1).reshape(*shape)

        return l, x_sample

    @torch.no_grad()
    def sample(self, shape, device, *args, **kwargs):
        x = torch.full(shape, -1).to(torch.float).to(device)
        _, x_sample = self.forward(x)
        return x_sample


if __name__ == '__main__':
    x = torch.rand(2, 1, 28, 28)
    f = NADE(784, 500)
    logit, sample = f(x)
    img = f.sample((2, 1, 28, 28), 'cpu')

    assert list(logit.shape) == [2, 1, 28, 28]
    print("[TEST] logit shape test success")
    assert list(sample.shape) == [2, 1, 28, 28]
    print("[TEST] sample shape test success")
    assert list(img.shape) == [2, 1, 28, 28]
    print("[TEST] sample shape test2 success")
