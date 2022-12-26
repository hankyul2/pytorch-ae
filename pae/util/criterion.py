import torch
from torch import nn
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

    if args.amp:
        scaler = NativeScalerWithGradAccum()
    else:
        scaler = None

    return criterion, scaler








