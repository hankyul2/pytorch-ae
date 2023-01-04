import torch
from torch.nn.parallel import DistributedDataParallel

from pae.model import NADE, MADE, PixelCNN, GatedPixelCNN, PixelCNNPP, PixelSnail, PixelSnailPP


def get_model(args):
    if args.model_name == 'NADE':
        model = NADE().to(args.device)
    elif args.model_name == 'MADE':
        model = MADE().to(args.device)
    elif args.model_name == 'PixelCNN':
        model = PixelCNN(ch=args.num_channels, category=args.num_classes, dataset=args.dataset_type).to(args.device)
    elif args.model_name == 'GatedPixelCNN':
        model = GatedPixelCNN(ch=args.num_channels, category=args.num_classes, dataset=args.dataset_type).to(args.device)
    elif args.model_name == 'PixelCNN++':
        model = PixelCNNPP(ch=args.num_channels, category=args.num_classes).to(args.device)
    elif args.model_name == 'PixelSnail':
        model = PixelSnail(ch=args.num_channels, category=args.num_classes).to(args.device)
    elif args.model_name == 'PixelSnail++':
        model = PixelSnailPP(ch=args.num_channels, category=args.num_classes).to(args.device)
    else:
        AssertionError(f"{args.model_name} is not supported yet!")

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # if args.sync_bn:
    #     model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.distributed:
        ddp_model = DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        ddp_model = None

    return model, ddp_model
