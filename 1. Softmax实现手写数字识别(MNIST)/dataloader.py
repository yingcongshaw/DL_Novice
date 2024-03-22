import os
import struct
import numpy as np


class Dataset(object):

    def __init__(self, data_root, mode='train', num_classes=10):
        # 初始化数据集
        assert mode in ['train', 'val', 'test']

        # 加载图片和标签
        kind = {'train': 'train', 'val': 'train', 'test': 't10k'}[mode]
        labels_path = os.path.join(data_root, '{}-labels.idx1-ubyte'.format(kind))
        images_path = os.path.join(data_root, '{}-images.idx3-ubyte'.format(kind))

        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)

        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

        if mode == 'train':
            # 训练集及其标签
            self.images = images[:55000]  # shape: (55000, 784)
            self.labels = labels[:55000]  # shape: (55000,)

        elif mode == 'val':
            # 验证集及其标签
            self.images = images[55000:]  # shape: (5000, 784)
            self.labels = labels[55000:]  # shape: (5000, )

        else:
            # 测试集及其标签
            self.images = images  # shape: (10000, 784)
            self.labels = labels  # shape: (10000, )

        self.num_classes = num_classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # 将图像的像素值归一化
        image = image / 255.0
        image = image - np.mean(image)

        return image, label


# 定义一个迭代batch采样器类
class IterationBatchSampler(object):

    # 初始化函数，参数为数据集，最大epoch，batch_size，shuffle
    def __init__(self, dataset, max_epoch, batch_size=2, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    # 准备epoch索引函数
    def prepare_epoch_indices(self):
        # 计算迭代次数
        num_iteration = len(self.dataset) // self.batch_size
        # 生成索引
        indices = np.arange(num_iteration*self.batch_size)
        # 如果shuffle为True，则打乱索引
        if self.shuffle:
            np.random.shuffle(indices)
        # 将索引按照迭代次数分割
        self.batch_indices = np.split(indices, num_iteration)
        
    # 返回迭代器
    def __iter__(self):
        return iter(self.batch_indices)

    # 返回长度
    def __len__(self):
        return len(self.batch_indices)


# 定义Dataloader类
class Dataloader(object):

    # 初始化函数
    def __init__(self, dataset, sampler):
        self.dataset = dataset
        self.sampler = sampler

    # 迭代函数
    def __iter__(self):
        self.sampler.prepare_epoch_indices()

        for batch_indices in self.sampler:
            batch_images = []
            batch_labels = []
            for idx in batch_indices:
                img, label = self.dataset[idx]
                batch_images.append(img)
                batch_labels.append(label)

            batch_images = np.stack(batch_images)
            batch_labels = np.stack(batch_labels)

            yield batch_images, batch_labels

    # 返回数据长度
    def __len__(self):
        return len(self.sampler)


# 构建数据加载器
def build_dataloader(data_root, max_epoch, batch_size, shuffle=False, mode='train'):
    # 构建数据加载器
    dataset = Dataset(data_root, mode)
    sampler = IterationBatchSampler(dataset, max_epoch, batch_size, shuffle)
    data_lodaer = Dataloader(dataset, sampler)
    return data_lodaer