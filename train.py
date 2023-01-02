import json
import os
from argparse import ArgumentParser

import torch
import wandb
from torchvision.utils import save_image

from pae.dataset import get_dataset, get_dataloader
from pae.model import get_model
from pae.util import Metric, setup, clear, get_optimizer_and_scheduler, get_criterion_scaler, reduce_mean


def get_arg_parser():
    # 1. setting
    parser = ArgumentParser(description='pytorch-auto-encoder')
    parser.add_argument('--config', type=str, default='config/train.json', help='train configuration json file path')
    parser.add_argument('--data-dir', type=str, default=os.path.join('data', 'mnist'), help='root path of dataset')
    parser.add_argument('--output-dir', type=str, default='log', help='root log dir')
    parser.add_argument('--who', type=str, default="hankyul2", help="entity name used for logger")
    parser.add_argument('--project-name', type=str, default="pytorch-ae", help="project name used for logger")
    parser.add_argument('--exp-target', type=str, default=['dataset_type', 'model_name'], help="arguments for experiment name")
    parser.add_argument('-c', '--cuda', type=str, default='0,', help="cuda devices")
    parser.add_argument('--print-freq', type=int, default=20, help='print log frequency')
    parser.add_argument('--use-wandb', action='store_true', help="use wandb to log metric")
    parser.add_argument('--seed', type=int, default=42, help='fix randomness for better reproducibility')
    parser.add_argument('--amp', action='store_true', default=False, help='enable native amp(fp16) training')
    parser.add_argument('--channels-last', action='store_true', help='change memory format to channels last')

    # 2. augmentation & dataset & dataloader
    parser.add_argument('-d', '--dataset-type', type=str, default='MNIST',
                        choices=['MNIST', 'CIFAR10', 'ImageNet64',], help='dataset')
    parser.add_argument('--train-size', type=int, default=(224, 224), nargs='+', help='train image size')
    parser.add_argument('--train-resize-mode', type=str, default='RandomResizedCrop', help='train image resize mode')
    parser.add_argument('--random-crop-pad', type=int, default=0, help='pad size for ResizeRandomCrop')
    parser.add_argument('--random-crop-scale', type=float, default=(0.08, 1.0), nargs='+',
                        help='train image resized scale for RandomResizedCrop')
    parser.add_argument('--random-crop-ratio', type=float, default=(3 / 4, 4 / 3), nargs='+',
                        help='train image resized ratio for RandomResizedCrop')
    parser.add_argument('-hf', '--hflip', type=float, default=0.5, help='random horizontal flip')
    parser.add_argument('-aa', '--auto-aug', action='store_true', default=False, help='enable timm rand augmentation')
    parser.add_argument('--cutmix', type=float, default=None, help='cutmix probability')
    parser.add_argument('--mixup', type=float, default=None, help='mix probability')
    parser.add_argument('-re', '--remode', type=float, default=None, help='random erasing probability')
    parser.add_argument('--gaussian-noise', action='store_true', help='add gaussian noise')
    parser.add_argument('--test-size', type=int, default=(224, 224), nargs='+', help='test image size')
    parser.add_argument('--test-resize-mode', type=str, default='resize_shorter', choices=['resize_shorter', 'resize'],
                         help='test resize mode')
    parser.add_argument('--center-crop-ptr', type=float, default=0.875, help='test image crop percent')
    parser.add_argument('--interpolation', type=str, default='bicubic', help='image interpolation mode')
    parser.add_argument('--mean', type=float, default=(0.485, 0.456, 0.406), nargs='+', help='image mean')
    parser.add_argument('--std', type=float, default=(0.229, 0.224, 0.225), nargs='+', help='image std')
    parser.add_argument('--aug-repeat', type=int, default=None, help='repeat augmentation')
    parser.add_argument('--drop-last', default=False, action='store_true', help='enable drop_last in train dataloader')
    parser.add_argument('-j', '--num-workers', type=int, default=4, help='number of workers')
    parser.add_argument('--pin-memory', action='store_true', default=False, help='pin memory in dataloader')

    # 2. model
    parser.add_argument('-m', '--model-name', type=str, default='NADE', help='the name of model')

    # 3. optimizer & learning rate
    parser.add_argument('-b', '--batch-size', type=int, default=512, help='the number of batch per step')
    parser.add_argument('--epoch', type=int, default=50, help='the number of training epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer name')
    parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
    parser.add_argument('--nesterov', action='store_true', default=False, help='use nesterov momentum')
    parser.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.999], help='adam optimizer beta parameter')
    parser.add_argument('--eps', type=float, default=1e-6, help='optimizer eps')
    parser.add_argument('--scheduler', type=str, default='cosine', help='lr scheduler')
    parser.add_argument('--step-size', type=int, default=2, help='lr decay step size')
    parser.add_argument('--decay-rate', type=float, default=0.1, help='lr decay rate')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='lowest lr used for cosine scheduler')
    parser.add_argument('--restart-epoch', type=int, default=20, help='warmup restart epoch period')
    parser.add_argument('--milestones', type=int, nargs='+', default=[150, 225], help='multistep lr decay step')
    parser.add_argument('--warmup-scheduler', type=str, default='linear', help='warmup lr scheduler type')
    parser.add_argument('--warmup-lr', type=float, default=1e-4, help='warmup start lr')
    parser.add_argument('--warmup-epoch', type=int, default=5, help='warmup epoch')
    parser.add_argument('--grad-norm', type=float, default=None, help='gradient clipping threshold')
    parser.add_argument('--grad-accum', type=int, default=1, help='gradient accumulation')
    parser.add_argument('--early-stop-epoch', type=int, default=50, help='early stop epoch')

    return parser


@torch.no_grad()
def validate(dataloader, model, critic, args, epoch):
    loss_m = Metric(header='Loss:')
    total_iter = len(dataloader)

    model.eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    for batch_idx, ((x_in, x_out), y) in enumerate(dataloader):
        x_in = x_in.to(args.device)
        x_out = x_out.to(args.device)

        if args.channels_last:
            x_in = x_in.to(memory_format=torch.channels_last)

        with torch.cuda.amp.autocast(args.amp):
            logit, x_recon = model(x_in)

            if args.model_name in ['PixelCNN++']:
                loss, nll_loss = critic(logit, x_in)
            else:
                loss, nll_loss = critic(logit, x_out)

        if args.distributed:
            loss = reduce_mean(nll_loss, args.world_size)

        loss_m.update(loss, len(x_in))

        if batch_idx and args.print_freq and batch_idx % args.print_freq == 0:
            num_digits = len(str(total_iter))
            args.log(f"VALID({epoch:03}): [{batch_idx:>{num_digits}}/{total_iter}] {loss_m}")

    save_image(x_recon[:16], os.path.join(args.log_dir, f"val_{epoch}.jpg"))

    return loss_m.compute()


def train(dataloader, model, critic, optimizer, scheduler, scaler, args, epoch):
    loss_m = Metric(header='Loss:')
    total_iter = len(dataloader)

    model.train()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    for batch_idx, ((x_in, x_out), y) in enumerate(dataloader):
        x_in = x_in.to(args.device)
        x_out = x_out.to(args.device)

        if args.channels_last:
            x_in = x_in.to(memory_format=torch.channels_last)

        with torch.cuda.amp.autocast(args.amp):
            logit, x_recon = model(x_in)

            if args.model_name in ['PixelCNN++'] and args.num_classes > 1:
                loss, nll_loss = critic(logit, x_in)
            else:
                loss, nll_loss = critic(logit, x_out)

        if args.amp:
            scaler(loss, optimizer, model.parameters(), scheduler, args.grad_norm, batch_idx % args.grad_accum == 0)
        else:
            loss.backward()
            if args.grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            if batch_idx % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()

        if args.distributed:
            loss = reduce_mean(nll_loss, args.world_size)

        loss_m.update(loss, len(x_in))

        if batch_idx and args.print_freq and batch_idx % args.print_freq == 0:
            num_digits = len(str(total_iter))
            args.log(f"TRAIN({epoch:03}): [{batch_idx:>{num_digits}}/{total_iter}] {loss_m}")

    return loss_m.compute()


@torch.no_grad()
def sample(f, args, epoch):
    shape = [16, args.num_channels, *args.train_size]

    with torch.cuda.amp.autocast(args.amp):
        sampled_img = f.sample(shape, args.device, args.mean, args.std)

    if args.is_rank_zero:
        save_image(sampled_img, os.path.join(args.log_dir, f'sample_{epoch}.jpg'))


def run(args):
    setup(args)

    # 1. define transform & load dataset
    train_dataset, val_dataset = get_dataset(args)
    train_dataloader, val_dataloader = get_dataloader(train_dataset, val_dataset, args)

    # 2. load model
    model, ddp_model = get_model(args)

    # 3. load optimizer, scheduler, criterion
    optimizer, scheduler = get_optimizer_and_scheduler(model, args)
    criterion, scaler = get_criterion_scaler(args)

    best_epoch = 0
    best_loss = 1000.0
    num_digit = len(str(args.epoch))
    for epoch in range(args.epoch):
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)

        train_loss = train(train_dataloader, ddp_model if args.distributed else model,
                           criterion, optimizer, scheduler, scaler, args, epoch)
        val_loss = validate(val_dataloader, model, criterion, args, epoch)
        sample(model, args, epoch)

        args.log(f"EPOCH({epoch:>{num_digit}}/{args.epoch}): Train Loss: {train_loss:.04f} Val Loss: {val_loss:.04f}")
        if args.use_wandb and args.is_rank_zero:
            args.log({
                'train_loss':train_loss, 'val_loss':val_loss,
                'val_img': wandb.Image(os.path.join(args.log_dir, f'val_{epoch}.jpg')),
                'sample_img': wandb.Image(os.path.join(args.log_dir, f'sample_{epoch}.jpg'))
            }, metric=True)

        if best_loss > val_loss:
            best_epoch = epoch
            best_loss = val_loss
            if args.is_rank_zero:
                torch.save(model.state_dict(), os.path.join(args.log_dir, f'{args.model_name}.pth'))
                args.log(f"Saved model (val loss: {best_loss:0.4f}) in to {args.log_dir}")

        if args.early_stop_epoch and best_epoch + args.early_stop_epoch < epoch:
            args.log(f"Early stop at {epoch}")
            break

    clear(args)


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    with open(args.config, 'rt') as f:
        default = json.load(f)[args.dataset_type]
    parser.set_defaults(**default)
    args = parser.parse_args()

    run(args)