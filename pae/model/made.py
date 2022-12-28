import random

import torch
from torch import nn
from torch.distributions import Bernoulli


class MaskLinear(nn.Linear):
    def __init__(self, *args, mask=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', mask)
        self.mask_index = 0

    def apply(self, idx):
        self.mask_index = idx

    def forward(self, x):
        self.weight.data *= self.mask[self.mask_index]
        return super().forward(x)


class MADE(nn.Module):
    SEED = 1000
    def __init__(self, d=784, h=8000, l=2, n_mask=1):
        super().__init__()
        assert n_mask == 1, f"The number of mask should be 1, not {n_mask}."
        self.d = d
        self.h = h
        self.l = l + 1
        self.n_mask = n_mask
        self.dims = [d] + [h] * l + [d]
        self.masks, self.orders = self.generate_mask()

        self.layers = nn.ModuleList([
           MaskLinear(self.dims[i], self.dims[i+1], mask=self.masks[i]) for i in range(len(self.dims)-1)
        ])
        self.act = nn.ReLU(inplace=True)

    def generate_mask(self):
        orders = list()
        masks = [torch.zeros(self.n_mask, self.dims[i+1], self.dims[i]) for i in range(self.l)]

        for mask_idx in range(self.n_mask):
            # 1. fix seed for reproducibility
            g = torch.Generator().manual_seed(self.SEED + mask_idx)

            # 2. generate random connection
            connections = [torch.randperm(self.d, generator=g)]
            for i, dim in enumerate(self.dims[1:-1]):
                low = min(connections[i]) if i > 0 else 0
                connections.append(torch.randint(low, self.d-1, (dim,), generator=g))
            connections.append(connections[0])

            # 3. generate mask
            for layer_idx in range(self.l-1):
                masks[layer_idx][mask_idx] = connections[layer_idx][None, :] <= connections[layer_idx+1][:, None]
            masks[-1][mask_idx] = connections[-2][None, :] < connections[-1][:, None]

            # 4. append order
            orders.append(connections[0].argsort())

        return masks, orders

    def choose_mask(self, idx):
        for layer in self.layers:
            layer.apply(idx)

    def forward_once(self, x):
        shape = x.shape
        x = x.reshape(shape[0], -1)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != self.l - 1:
                x = self.act(x)

        x = x.reshape(shape)
        p = torch.sigmoid(x)
        sample = Bernoulli(probs=p).sample()

        return x, sample

    def forward(self, x):
        if self.train():
            mask_idx = random.randint(0, self.n_mask - 1)
            self.choose_mask(mask_idx)
            return self.forward_once(x)
        else:
            logits, samples = list(), list()
            for mask_idx in range(self.n_mask):
                self.choose_mask(mask_idx)
                logit, sample = self.forward_once(x)
                logits.append(logit)
                samples.append(sample)
            logit = sum(logits) / len(logits)
            sample = sum(samples) / len(samples)
            return logit, sample

    @torch.no_grad()
    def sample(self, shape, device, *args, **kwargs):
        B, C, H, W = shape

        output = list()
        for mask_idx in range(self.n_mask):
            x = torch.full(shape, -1).to(torch.float).to(device)
            self.choose_mask(mask_idx)
            order = self.orders[mask_idx]
            i = 0
            for h in range(H):
                for w in range(W):
                    _, sample = self.forward_once(x)
                    x[:, :, order[i] // W,  order[i] % W] = sample.reshape(B, -1)[:, order[i]:order[i]+1]
                    i += 1
            output.append(x)
        output = sum(output) / len(output)

        return output


if __name__ == '__main__':
    x = torch.rand(3, 1, 28, 28)
    f = MADE()
    logit, sample = f(x)
    generated = f.sample((3, 1, 28, 28), 'cpu')
    print(logit.shape)
    print(generated.shape)
