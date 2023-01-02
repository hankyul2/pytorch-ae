import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss


class NativeScalerWithGradAccum:
    def __init__(self):
        """NativeScalerWithGradAccum (timm)
        Native(pytorch) f16 scaler
        """
        self._scaler = GradScaler()

    def __call__(self, loss, optimizer, model_param, scheduler=None, grad_norm=None, update=True):
        self._scaler.scale(loss).backward()
        if update:
            if grad_norm:
                self._scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_param, grad_norm)
            self._scaler.step(optimizer)
            self._scaler.update()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


class BNLLLoss(BCEWithLogitsLoss):
    def __init__(self):
        """Binary Negative Log Likelihood (BNLL)"""
        super().__init__(reduction='sum')
    def forward(self, x, y):
        output = super().forward(x, y)
        return output / len(x), output / len(x)


class CNLLLoss(CrossEntropyLoss):
    def __init__(self):
        """Negative Log Likelihood (BNLL)"""
        super().__init__(reduction='mean')
        self.register_buffer('log128', torch.log(torch.tensor(128)))
        self.register_buffer('log2', torch.log(torch.tensor(2)))

    def forward(self, x, y):
        output = super().forward(x, y)
        return output, -(output-self.log128) / self.log2


class MultiLogitLoss(nn.Module):
    def __init__(self, mixture=10):
        super().__init__()
        self.register_buffer('log127', torch.log(torch.tensor(127.5)))
        self.register_buffer('log2', torch.log(torch.tensor(2)))
        self.mixture = mixture

    def forward(self, x, y):
        B, C, H, W = y.shape
        L = x.size(1)
        y = y[:, :, None, :, :].expand(-1, -1, self.mixture, -1, -1)
        x = x.reshape(B, L//self.mixture, self.mixture, H, W)

        # 1. compute mean & scale
        logit_prob = x[:, 0]
        mean_and_scales = x[:, 1:].tensor_split(C, dim=1)

        if C == 1:
            mean, log_scale = mean_and_scales[0].tensor_split(2, dim=1)
            log_scale = torch.clamp(log_scale, min=-7.0)
        else:
            (m0, m1, m2), (c0, c1, c2), log_scale = [channel.tensor_split(3, dim=1) for channel in mean_and_scales]
            y0, y1, y2 = y.tensor_split(3, dim=1)
            mean = torch.cat([m0, m1 + y0 * torch.tanh(c0), m2 + y0 * torch.tanh(c1) + y1 * torch.tanh(c2)], dim=1)
            log_scale = torch.clamp(torch.cat(log_scale, dim=1), min=-7.0)

        # 2. compute logit prob
        std = torch.exp(-log_scale)

        in_plus = std * (y - mean + 1/255.0)
        in_zero = std * (y - mean)
        in_minus = std * (y - mean - 1/255.0)

        cdf_plus = torch.sigmoid(in_plus)
        cdf_minus = torch.sigmoid(in_minus)

        # 3. compute target objective
        cdf_delta = cdf_plus - cdf_minus
        log_cdf_plus = in_plus - F.softplus(in_plus)
        log_pdf = in_zero - log_scale - 2 * F.softplus(in_zero) - self.log127
        log_cdf_minus = -F.softplus(in_minus)

        # 4. compute loss
        multi_logit_loss = torch.where(
            y < -0.999, log_cdf_plus, torch.where(
            y > 0.999, log_cdf_minus, torch.where(
            cdf_delta > 1e-5, torch.log(torch.clamp(cdf_delta, min=1e-12)), log_pdf
        ))).sum(dim=1)
        idk_loss = torch.log_softmax(logit_prob, 1)

        if C == 1:
            loss = -(multi_logit_loss + idk_loss).logsumexp(1).sum() / (B * self.mixture) # NAT
        else:
            loss = -(multi_logit_loss + idk_loss).logsumexp(1).mean() / self.log2 # Bit Per Dim (BPD)

        return loss, loss


def get_criterion_scaler(args):
    """Get Criterion(Loss) function and scaler
    Criterion functions are divided depending on usage of mixup
    - w/ mixup - you don't need to add smoothing loss, because mixup will add smoothing loss.
    - w/o mixup - you should need to add smoothing loss
    """
    if args.num_classes == 1:
        criterion = BNLLLoss()
    else:
        criterion = CNLLLoss().to(args.device)

        if args.model_name in ['PixelCNN++']:
            criterion = MultiLogitLoss().to(args.device)

    if args.amp:
        scaler = NativeScalerWithGradAccum()
    else:
        scaler = None

    return criterion, scaler


if __name__ == '__main__':
    x = torch.rand([2, 100, 28, 28]).fill_(1.0)
    y = torch.rand([2, 3, 28, 28])
    criterion = MultiLogitLoss()
    loss = criterion.forward(x, y)
    loss2 = criterion.forward2(x, y)
    print(f"loss: {loss[0]} loss2: {loss2}")






