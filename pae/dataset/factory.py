from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler, default_collate
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import RandomChoice

from .cutmix import MixUP, CutMix
from .repeated_aug_sampler import RepeatAugSampler
from .transform import TrainTransform, ValTransform


_dataset_dict = {
    "MNIST": dict(dataset_class=MNIST, num_channels=1, num_classes=1),
    "CIFAR10": dict(dataset_class=CIFAR10, num_channels=3, num_classes=256),
}

def get_dataset(args):
    assert _dataset_dict.get(args.dataset_type, None), f"{args.dataset_type} is not found."

    dataset_dict = _dataset_dict[args.dataset_type]
    dataset_class = dataset_dict['dataset_class']
    is_binary = dataset_dict['num_classes'] == 1
    args.num_channels = dataset_dict['num_channels']
    args.num_classes = dataset_dict['num_classes']

    train_transform = TrainTransform(
        args.train_size, args.train_resize_mode, args.random_crop_pad, args.random_crop_scale, args.random_crop_ratio,
        args.hflip, args.auto_aug, args.remode, args.gaussian_noise, args.interpolation, args.mean, args.std, is_binary,
    )
    val_transform = ValTransform(
        args.test_size, args.test_resize_mode, args.center_crop_ptr, args.interpolation, args.mean, args.std, is_binary,
    )
    train_dataset = dataset_class(args.data_dir, train=True, download=True, transform=train_transform)
    val_dataset = dataset_class(args.data_dir, train=False, download=True, transform=val_transform)

    return train_dataset, val_dataset


def get_dataloader(train_dataset, val_dataset, args):
    # 1. create sampler
    if args.distributed:
        if args.aug_repeat:
            train_sampler = RepeatAugSampler(train_dataset, num_repeats=args.aug_repeat)
        else:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)

    # 2. create collate_fn
    mix_collate = []
    if args.mixup:
        mix_collate.append(MixUP(alpha=args.mixup, nclass=args.num_classes))
    if args.cutmix:
        mix_collate.append(CutMix(alpha=args.mixup, nclass=args.num_classes))

    if mix_collate:
        mix_collate = RandomChoice(mix_collate)
        collate_fn = lambda batch: mix_collate(*default_collate(batch))
    else:
        collate_fn = None

    # 3. create dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler,
                                  num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=args.pin_memory,
                                  drop_last=args.drop_last)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
                                  num_workers=args.num_workers, collate_fn=None, pin_memory=False)

    args.iter_per_epoch = len(train_dataloader)

    return train_dataloader, val_dataloader
