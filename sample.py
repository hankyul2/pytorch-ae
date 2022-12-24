import os
from argparse import ArgumentParser
from pathlib import Path

from torchvision.utils import save_image
from tqdm import tqdm

import torch
import torchvision.transforms as TVT
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from pae.dataset import _dynamically_binarize
from pae.model import NADE
from pae.util import Metric, BNLLLoss


def get_arg_parser():
    # 1. setting
    parser = ArgumentParser(description='pytorch-auto-encoder')
    parser.add_argument('--data-dir', type=str, default=os.path.join('data', 'mnist'), help='root path of dataset')
    parser.add_argument('--log-dir', type=str, default='sample_log', help='root log dir')
    parser.add_argument('--cuda', type=str, default='0,', help="cuda devices")

    # 2. model
    parser.add_argument('-m', '--model-name', type=str, default='NADE', help='the name of model')
    parser.add_argument('-cp', '--checkpoint', type=str, default='', help='saved state dict path')

    return parser


def setup(args):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    args.device = 'cuda:0'

    log_root_path = os.path.join(args.log_dir, args.model_name)
    Path(log_root_path).mkdir(exist_ok=True, parents=True)

    args.run_id = f"v{len(os.listdir(log_root_path))}"
    args.log_dir = os.path.join(log_root_path, args.run_id)
    Path(args.log_dir).mkdir(exist_ok=True, parents=True)


def run(args):
    setup(args)

    if args.model_name == 'NADE':
        f = NADE().to(args.device)
    else:
        AssertionError(f"{args.model_name} is not supported yet!")
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    f.load_state_dict(state_dict)

    sampled_img = f.sample(16, args.device).reshape(16, 1, 28, 28)
    save_image(sampled_img, os.path.join(args.log_dir, f'{args.model_name}_sampled_img.jpg'))


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    run(args)