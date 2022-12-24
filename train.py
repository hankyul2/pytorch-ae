import os
from argparse import ArgumentParser

import torch
import torchvision.transforms as TVT
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from pae.dataset import _dynamically_binarize
from pae.model import NADE, PixelCNN
from pae.util import Metric, BNLLLoss, setup, clear


def get_arg_parser():
    # 1. setting
    parser = ArgumentParser(description='pytorch-auto-encoder')
    parser.add_argument('--data-dir', type=str, default=os.path.join('data', 'mnist'), help='root path of dataset')
    parser.add_argument('--output-dir', type=str, default='log', help='root log dir')
    parser.add_argument('--who', type=str, default="hankyul2", help="entity name used for logger")
    parser.add_argument('--project-name', type=str, default="pytorch-ae", help="project name used for logger")
    parser.add_argument('--exp-target', type=str, default=['model_name'], help="arguments for experiment name")
    parser.add_argument('--cuda', type=str, default='0,', help="cuda devices")
    parser.add_argument('--print-freq', type=int, default=20, help='print log frequency')
    parser.add_argument('--use-wandb', action='store_true', help="use wandb to log metric")
    parser.add_argument('--seed', type=int, default=42, help='fix randomness for better reproducibility')

    # 2. model
    parser.add_argument('-m', '--model-name', type=str, default='NADE', help='the name of model')

    # 3. optimizer & learning rate
    parser.add_argument('--batch-size', type=int, default=512, help='the number of batch per step')
    parser.add_argument('--epoch', type=int, default=50, help='the number of training epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')

    return parser


@torch.no_grad()
def validate(dataloader, f, critic, args, epoch):
    loss_m = Metric(header='Loss:')
    total_iter = len(dataloader)
    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(args.device)

        y_hat, x_hat = f(x)
        loss = critic(y_hat, x)

        loss_m.update(loss, len(x))

        if batch_idx and args.print_freq and batch_idx % args.print_freq == 0:
            num_digits = len(str(total_iter))
            args.log(f"VALID({epoch:03}): [{batch_idx:>{num_digits}}/{total_iter}] {loss_m}")

    save_image(x_hat[:16], os.path.join(args.log_dir, f"val_{epoch}.jpg"))

    return loss_m.compute()


def train(dataloader, f, critic, optim, args, epoch):
    loss_m = Metric(header='Loss:')
    total_iter = len(dataloader)
    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(args.device)

        y_hat, x_hat = f(x)
        loss = critic(y_hat, x)
        loss.backward()
        optim.step()
        optim.zero_grad()

        loss_m.update(loss, len(x))

        if batch_idx and args.print_freq and batch_idx % args.print_freq == 0:
            num_digits = len(str(total_iter))
            args.log(f"TRAIN({epoch:03}): [{batch_idx:>{num_digits}}/{total_iter}] {loss_m}")

    return loss_m.compute()


def sample(f, args, epoch):
    sampled_img = f.sample((16, 1, 28, 28), args.device)
    save_image(sampled_img, os.path.join(args.log_dir, f'sample_{epoch}.jpg'))


def run(args):
    setup(args)

    transform = TVT.Compose([TVT.ToTensor(), _dynamically_binarize])
    train_dataset = MNIST(args.data_dir, train=True, download=True, transform=transform)
    val_dataset = MNIST(args.data_dir, train=False, download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.model_name == 'NADE':
        f = NADE().to(args.device)
    elif args.model_name == 'PixelCNN':
        f = PixelCNN().to(args.device)
    else:
        AssertionError(f"{args.model_name} is not supported yet!")

    optim = AdamW(f.parameters(), lr=args.lr, weight_decay=args.wd)
    critic = BNLLLoss()

    best_loss = 1000.0
    num_digit = len(str(args.epoch))
    for epoch in range(args.epoch):
        train_loss = train(train_dataloader, f, critic, optim, args, epoch)
        val_loss = validate(val_dataloader, f, critic, args, epoch)
        sample(f, args, epoch)

        args.log(f"EPOCH({epoch:>{num_digit}}/{args.epoch}): Train Loss: {train_loss:.04f} Val Loss: {val_loss:.04f}")
        if args.use_wandb:
            args.log({
                'train_loss':train_loss, 'val_loss':val_loss,
                'val_img': wandb.Image(os.path.join(args.log_dir, f'val_{epoch}.jpg')),
                'sample_img': wandb.Image(os.path.join(args.log_dir, f'sample_{epoch}.jpg'))
            }, metric=True)

        if best_loss > val_loss:
            best_loss = val_loss
            state_dict = {k: v.cpu() for k, v in f.state_dict().items()},
            torch.save(state_dict, os.path.join(args.log_dir, f'{args.model_name}.pth'))
            args.log(f"Saved model (val loss: {best_loss:0.4f}) in to {args.log_dir}")
    clear(args)


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    run(args)