import torch
from torch import nn


def reparam_trick(mean, std):
    return mean + (0.5 * std).exp() * torch.randn_like(std)


def compute_kl_loss(mean, std):
    return -0.5 * (1 + std - std.exp() - mean ** 2).sum(dim=1).mean()


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0):
        super().__init__()
        self.residual = in_dim == out_dim
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        if self.residual:
            out = out + x

        return out


class DFT_VAE(nn.Module):
    def __init__(self, shape=(1, 28, 28), layer=6, hidden_dim=256, crop_dim=9, latent_dim=8):
        super().__init__()
        C, H, W = shape
        in_dim = crop_dim * crop_dim * 2 if crop_dim else C * H * (int(W) // 2 + 1) * 2
        enc_dims = [in_dim] + [hidden_dim] * layer + [latent_dim * 2]
        dec_dims = [latent_dim] + [hidden_dim] * layer + [in_dim]
        self.enc = nn.Sequential(*[MLP(enc_dims[i], enc_dims[i+1]) for i in range(len(enc_dims)-1)])
        self.dec = nn.Sequential(*[MLP(dec_dims[i], dec_dims[i+1]) for i in range(len(dec_dims)-1)])
        self.crop_dim = crop_dim
        self.latent_dim = latent_dim

    def forward(self, x):
        B, C, H, W = x.shape
        signal = torch.fft.rfft2(x, norm='forward')
        signal = signal[:, :, :self.crop_dim, :self.crop_dim]
        signal = torch.view_as_real(signal).reshape(B, -1)

        m, std = self.enc(signal).tensor_split(2, dim=1)
        z = reparam_trick(m, std)
        kl_loss = compute_kl_loss(m, std)
        signal = self.dec(z)

        sample = torch.view_as_complex(torch.zeros(B, C, H, int(W) // 2 + 1, 2)).to(x.device)
        signal = torch.view_as_complex(signal.float().reshape(B, C, self.crop_dim, self.crop_dim, 2))
        sample[:, :, :self.crop_dim, :self.crop_dim] = signal
        sample = torch.fft.irfft2(sample, norm='forward').float()
        return signal, sample, kl_loss

    def sample(self, shape, device, *args, **kwargs):
        B, C, H, W = shape
        shape = (B, self.latent_dim)
        z = torch.randn(shape).to(device)
        l = self.dec(z)
        sample = torch.view_as_complex(torch.zeros(B, C, H, int(W) // 2 + 1, 2)).to(device)
        signal = torch.view_as_complex(l.float().reshape(B, C, self.crop_dim, self.crop_dim, 2))
        sample[:, :, :self.crop_dim, :self.crop_dim] = signal
        sample = (torch.fft.irfft2(sample, norm='forward') > 0.5).float()

        return sample


class Linear_VAE(nn.Module):
    def __init__(self, ch=1, category=1, shape=(28, 28), hidden_dim=256, layer=6, latent_dim=16):
        super().__init__()
        in_dim = ch * shape[0] * shape[1]
        enc_dims = [in_dim] + [hidden_dim] * layer + [latent_dim * 2]
        dec_dims = [latent_dim] + [hidden_dim] * layer + [in_dim]
        self.enc = nn.Sequential(*[MLP(enc_dims[i], enc_dims[i+1]) for i in range(len(enc_dims)-1)])
        self.dec = nn.Sequential(*[MLP(dec_dims[i], dec_dims[i+1]) for i in range(len(dec_dims)-1)])
        self.latent_dim = latent_dim

    def forward(self, x):
        B, C, H, W = x.shape
        m, std = self.enc(x.reshape(B,-1)).tensor_split(2, dim=1)
        z = reparam_trick(m, std)
        kl_loss = compute_kl_loss(m, std)
        out = self.dec(z).reshape(B, C, H, W)
        sample = (torch.sigmoid(out) > 0.5).float()

        return out, sample, kl_loss

    def sample(self, shape, device, *args, **kwargs):
        B, C, H, W = shape
        shape = (B, self.latent_dim)
        z = torch.randn(shape).to(device)
        l = (torch.sigmoid(self.dec(z)) > 0.5).reshape(B, C, H, W).float()

        return l


if __name__ == '__main__':
    x = torch.rand(2, 1, 28, 28)
    f = Linear_VAE()
    signal, sample, kl_loss = f(x)
    random = f.sample([2, 1, 28, 28], 'cpu')
    print(signal.shape)
    print(sample.shape)
    print(random.shape)