import gc
import glob
import logging
import os
import random
from pathlib import Path
from functools import partial
from datetime import datetime

import numpy
import torch
import wandb
import torch.distributed as dist


def allow_print_to_master(is_master):
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)

        if force or is_master:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def check_need_init():
    if os.environ.get('INITIALIZED', None):
        return False
    else:
        return True


def init_distributed_mode(args):
    os.environ['INITIALIZED'] = 'TRUE'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    print(f'{datetime.now().strftime("[%Y/%m/%d %H:%M]")} ', end='')

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.distributed = True
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_backend = 'nccl'
        args.dist_url = 'env://'

        print(f'| distributed init (rank {args.rank}): {args.dist_url}')
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                             world_size=args.world_size, rank=args.rank)
    else:
        print('| Not using distributed mode')
        args.distributed = False
        args.gpu = 0

    args.is_rank_zero = args.gpu == 0
    allow_print_to_master(args.is_rank_zero)
    torch.cuda.set_device(args.gpu)
    args.device = torch.device(f'cuda:{args.gpu}')


def make_logger(log_file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", "[%Y/%m/%d %H:%M]")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(filename=log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def log(msg, metric=False, logger=None):
    if logger:
        if metric:
            wandb.log(msg)
        else:
            logger.info(msg)


def init_logger(args):
    args.exp_name = '_'.join(str(getattr(args, target)) for target in args.exp_target)
    args.version_id = len(list(glob.glob(os.path.join(args.output_dir, f'{args.exp_name}_v*'))))
    args.exp_name = f'{args.exp_name}_v{args.version_id}'

    args.start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    args.log_dir = os.path.join(args.output_dir, args.exp_name)
    args.text_log_path = os.path.join(args.log_dir, 'log.txt')
    args.best_weight_path = os.path.join(args.log_dir, 'best_weight.pth')

    if args.distributed:
        dist.barrier()  # to ensure have save version id (must be same for knn classifier)

    if args.is_rank_zero:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        args.logger = make_logger(args.text_log_path)
        if args.use_wandb:
            wandb.init(project=args.project_name, name=args.exp_name, entity=args.who, config=args, reinit=True)
    else:
        args.logger = None

    args.log = partial(log, logger=args.logger)


def clear(args):
    # 1. clear gpu memory
    torch.cuda.empty_cache()
    # 2. clear cpu memory
    gc.collect()
    # 3. close logger
    if args.is_rank_zero:
        handlers = args.logger.handlers[:]
        for handler in handlers:
            args.logger.removeHandler(handler)
            handler.close()
        if args.use_wandb:
            wandb.finish(quiet=True)


def setup(args):
    if check_need_init():
        init_distributed_mode(args)
    init_logger(args)

    if args.seed is not None:
        numpy.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = True