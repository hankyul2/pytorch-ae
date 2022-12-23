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
    parser.add_argument('--log-dir', type=str, default='log', help='root log dir')
    parser.add_argument('--project-name', type=str, default="pytorch-ae", help="project name used for logger")
    parser.add_argument('--cuda', type=str, default='0,', help="cuda devices")

    # 2. model
    parser.add_argument('-m', '--model-name', type=str, default='NADE', help='the name of model')

    # 3. optimizer & learning rate
    parser.add_argument('--batch-size', type=int, default=512, help='the number of batch per step')
    parser.add_argument('--epoch', type=int, default=50, help='the number of training epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')

    return parser


@torch.no_grad()
def validate(dataloader, f, critic, device, epoch, log_dir):
    loss_m = Metric(header='Loss:')
    prog_bar = tqdm(dataloader, leave=True)
    for x, y in prog_bar:
        x = x.to(device)

        y_hat, x_hat = f(x)
        loss = critic(y_hat, x)

        loss_m.update(loss, len(x))
        prog_bar.set_description(f"Val {loss_m}")

    save_image(x_hat[:16], os.path.join(log_dir, f"val_{epoch}.jpg"))

    return loss_m.compute()


def train(dataloader, f, critic, optim, device):
    loss_m = Metric(header='Loss:')
    prog_bar = tqdm(dataloader, leave=True)
    for x, y in prog_bar:
        x = x.to(device)

        y_hat, x_hat = f(x)
        loss = critic(y_hat, x)
        loss.backward()
        optim.step()
        optim.zero_grad()

        loss_m.update(loss, len(x))
        prog_bar.set_description(f"Train {loss_m}")

    return loss_m.compute()


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

    transform = TVT.Compose([TVT.ToTensor(), _dynamically_binarize])
    train_dataset = MNIST(args.data_dir, train=True, download=True, transform=transform)
    val_dataset = MNIST(args.data_dir, train=False, download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.model_name == 'NADE':
        f = NADE().to(args.device)
    else:
        AssertionError(f"{args.model_name} is not supported yet!")

    optim = AdamW(f.parameters(), lr=args.lr, weight_decay=args.wd)
    critic = BNLLLoss()

    for epoch in range(args.epoch):
        train_loss = train(train_dataloader, f, critic, optim, args.device)
        val_loss = validate(val_dataloader, f, critic, args.device, epoch, args.log_dir)
        # print(f"{epoch}/{args.epoch} epoch {train_loss:0.4f} train loss {val_loss:0.4f} val loss")


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    run(args)