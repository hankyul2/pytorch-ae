import os
from argparse import ArgumentParser


def get_arg_parser():
    # 1. setting
    parser = ArgumentParser(description='pytorch-auto-encoder')
    parser.add_argument('--data-dir', type=str, default=os.path.join('data', 'mnist'), help='root path of dataset')
    parser.add_argument('--project-name', type=str, default="pytorch-ae", help="project name used for logger")

    # 2. optimizer & learning rate
    parser.add_argument('--epoch', type=int, default=10, help='the number of training epoch')

    return parser


def run(args):
    print(f'this is {args.project_name} project. good luck with your experiment.')


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    run(args)