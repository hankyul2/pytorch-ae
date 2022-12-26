import torch
from torch import distributed as dist


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


def all_reduce_mean(val, world_size):
    """Collect value to each gpu
    :arg
        val(tensor): target
        world_size(int): the number of process in each group
    """
    val = val.clone()
    dist.all_reduce(val, dist.ReduceOp.SUM)
    val = val / world_size
    return val


def all_reduce_sum(val):
    """Collect value to each gpu
    :arg
        val(tensor): target
        world_size(int): the number of process in each group
    """
    val = val.clone()
    dist.all_reduce(val, dist.ReduceOp.SUM)
    return val


def reduce_mean(val, world_size):
    """Collect value to local zero gpu
    :arg
        val(tensor): target
        world_size(int): the number of process in each group
    """
    val = val.clone()
    dist.reduce(val, 0, dist.ReduceOp.SUM)
    val = val / world_size
    return val


def all_gather(x):
    """Collect value to local rank zero gpu
    :arg
        x(tensor): target
    """
    if dist.is_initialized():
        dest = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(dest, x)
        return torch.cat(dest, dim=0)
    else:
        return x


def all_gather_with_different_size(x):
    """all gather operation with different sized tensor
    :arg
        x(tensor): target
    (reference) https://stackoverflow.com/a/71433508/17670380
    """
    if dist.is_initialized():
        local_size = torch.tensor([x.size(0)], device=x.device)
        all_sizes = all_gather(local_size)
        max_size = max(all_sizes)

        size_diff = max_size.item() - local_size.item()
        if size_diff:
            padding = torch.zeros(size_diff, device=x.device, dtype=x.dtype)
            x = torch.cat((x, padding))

        all_gathered_with_pad = all_gather(x)
        all_gathered = []
        ws = dist.get_world_size()
        for vector, size in zip(all_gathered_with_pad.chunk(ws), all_sizes.chunk(ws)):
            all_gathered.append(vector[:size])

        return torch.cat(all_gathered, dim=0)
    else:
        return x
