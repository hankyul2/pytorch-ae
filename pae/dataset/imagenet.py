import os
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm


class ImageNet64:
    train_split_paths = ['Imagenet64_train_part1_npz', 'Imagenet64_train_part2_npz']
    val_split_paths = ['Imagenet64_val_npz']
    def __init__(self, root, train, transform=None, **kwargs):
        self.root = root
        self.train = train
        self.transform = transform
        if self.train:
            data = [np.load(x, allow_pickle=True) for folder in self.train_split_paths
                    for x in glob(os.path.join(root, folder, '*.npz'))]
        else:
            data = [np.load(x, allow_pickle=True) for folder in self.val_split_paths
                         for x in glob(os.path.join(root, folder, '*.npz'))]

        is_single = os.environ.get('LOCAL_RANK', None) is None
        is_master = is_single or int(os.environ['LOCAL_RANK']) == 0

        self.x = [item['data'] for item in tqdm(data, desc='load image', disable=not is_master)]
        self.y = [item['labels'] for item in tqdm(data, desc='load label', disable=not is_master)]

        self.size_of_data = [len(item) for item in self.x]
        self.cum_size_of_data = [sum(self.size_of_data[:i+1]) for i in range(len(self.size_of_data))]

    def __len__(self):
        return sum(self.size_of_data)

    def get_index(self, item):
        for i in range(len(self.cum_size_of_data)):
            if item < self.cum_size_of_data[i]:
                return i, item - (0 if i==0 else self.cum_size_of_data[i-1])

    def __getitem__(self, item):
        group_idx, item_idx = self.get_index(item)
        img = Image.fromarray(self.x[group_idx][item_idx].reshape(64, 64, 3))
        label = self.y[group_idx][item_idx]

        if self.transform:
            img = self.transform(img)

        return img, label


if __name__ == '__main__':
    dataset = ImageNet64('../../data/imagenet64/', True)
    print(dataset[0][0])
