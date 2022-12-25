import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss


class BNLLLoss(BCEWithLogitsLoss):
    def __init__(self):
        """Binary Negative Log Likelihood (BNLL)"""
        super().__init__(reduction='sum')
    def forward(self, x, y):
        output = super().forward(x, y)
        return output / len(x)


class CNLLLoss(CrossEntropyLoss):
    def __init__(self):
        """Negative Log Likelihood (BNLL)"""
        super().__init__(reduction='sum')
    def forward(self, x, y):
        output = super().forward(x, y)
        return output / len(x)


class Metric:
    def __init__(self, header='', fmt='{val:.4f} ({avg:.4f})'):
        """Base Metric Class supporting ddp setup
        :arg
            reduce_ever_n_step(int): call all_reduce every n step in ddp mode
            reduce_on_compute(bool): call all_reduce in compute() method
            fmt(str): format representing metric in string
        """
        self.val = 0
        self.sum = 0
        self.n = 0
        self.avg = 0
        self.header = header
        self.fmt = fmt

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.detach().clone()

        self.val = val
        self.sum += val * n
        self.n += n
        self.avg = self.sum / self.n

    def compute(self):
        return self.avg

    def __str__(self):
        return self.header + ' ' + self.fmt.format(**self.__dict__)
