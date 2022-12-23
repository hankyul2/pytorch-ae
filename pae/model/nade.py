import torch
from torch import nn
from torch.distributions import Bernoulli
from torch.nn import functional as F


class NADE(nn.Module):
    def __init__(self, D=784, H=500):
        super().__init__()
        self.D = D
        self.W = nn.Parameter(torch.rand(H, D))
        self.c = nn.Parameter(torch.rand(H))
        self.V = nn.Parameter(torch.rand(D, H))
        self.b = nn.Parameter(torch.rand(D))

    def forward(self, x):
        shape = x.shape
        x = x.reshape(shape[0], -1)
        p = list()
        x_sample = list()

        a_d = self.c.expand(shape[0], -1) # B x H
        for d in range(self.D):
            h_d = F.relu(a_d, inplace=False) # B x H
            p_d = torch.sigmoid(h_d @ self.V[d:d+1].t() + self.b[d]) # B x 1

            x_org = x[:, d:d+1]
            x_new = Bernoulli(probs=p_d).sample()
            need_to_sample = x_org == -1
            x_d = torch.where(need_to_sample, x_new, x_org)
            a_d = x_d @ self.W[:, d:d+1].t() + a_d

            p.append(p_d)
            x_sample.append(x_new)

        p = torch.concat(p, dim=-1).reshape(*shape)
        x_sample = torch.concat(x_sample, dim=-1).reshape(*shape)
        return p, x_sample