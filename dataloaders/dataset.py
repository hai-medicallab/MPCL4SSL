import os
from random import random

import torch
import numpy as np
from glob import glob

from scipy import ndimage
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
from torchvision.transforms import Compose, transforms
import random


# 数据增强手段
def color_jitter(image):
    # 检查输入类型
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # 处理输入维度 (20, 160, 160)
    if image.dim() == 3:  # 形状为 (D, H, W)
        # 对每个切片进行颜色抖动
        jittered_images = []
        jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)  # 设置抖动强度

        for i in range(image.size(0)):  # 遍历每个切片
            jittered_image = jitter(image[i].unsqueeze(0))  # 添加通道维度，变为 (1, H, W)
            jittered_images.append(jittered_image.squeeze(0))  # 去掉通道维度

        # 将结果堆叠为 (D, H, W) 形状
        jittered_images = torch.stack(jittered_images)

        return jittered_images  # 返回抖动后的图像


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def cutout_gray(img, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1 / 0.3, value_min=0, value_max=1,
                pixel_level=True):
    if random.random() < p:
        img = np.array(img)
        mask = np.array(mask)
        img_h, img_w = img.shape

        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_w)
            y = np.random.randint(0, img_h)
            if x + erase_w <= img_w and y + erase_h <= img_h:
                break

        if pixel_level:
            value = np.random.randint(value_min, value_max + 1, (erase_h, erase_w))
        else:
            value = np.random.randint(value_min, value_max + 1)

        img[y:y + erase_h, x:x + erase_w] = value
        mask[y:y + erase_h, x:x + erase_w] = 0

    return img, mask





class X(Dataset):
    def __init__(self, base_dir=None, split='train', slice_strategy=2, num=None, transform=None):
        # self._base_dir = base_dir
        # self.transform = transform
        # self.sample_list = []
        # self.slice_random = np.random.randint(0, slice_strategy, 40)
        # self.labelidx = [i for i in list(range(0, 32, slice_strategy))] # CT use 32
        # self.split = split
        # print(self.labelidx)
        def generate_labelidx(slice_strategy):
            total_slices = 32
            # 生成每个块的随机偏移量
            block_offsets = np.random.randint(0, slice_strategy, size=total_slices // slice_strategy)

            # 生成带随机偏移的标签索引
            labelidx = []
            for block_idx, offset in enumerate(block_offsets):
                base = block_idx * slice_strategy
                # 确保不越界
                final_idx = min(base + offset, total_slices - 1)
                labelidx.append(final_idx)

            # 去重并排序
            return sorted(list(set(labelidx)))

            # 使用示例

        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        # self.slice_random = np.random.randint(0, slice_strategy, 40)
        # print(self.slice_random)
        all_slices = list(range(0, 32))  # 全部的索引列表
        # self.slice1 = [i for i in list(range(0, 20, slice_strategy))]
        # self.slice2 = [i for i in list(range(0, 160, slice_strategy * 8))]
        # 根据 slice_strategy 选择存入 slice1 的索引
        # 设置随机偏移实验
        # self.labelidx = [i for i in all_slices if i % slice_strategy == 0]
        self.labelidx = generate_labelidx(slice_strategy)
        print(self.labelidx)
        # 加载训练或测试样本列表
        if 'train' in split:
            with open(self._base_dir + 'train.txt', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir + 'test.txt', 'r') as f:
                self.image_list = f.readlines()

        # 移除换行符并提取每个图像的名称
        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]

        if 'train' in split:
            print("train total {} samples".format(len(self.image_list)))
        elif split == 'test':
            print("test total {} samples".format(len(self.image_list)))



    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/{}.h5".format(image_name), 'r')
        image = h5f['image'][:]  # 64*192*192
        labeltmp = h5f['label'][:]
        label = np.zeros_like(labeltmp)

        # 根据 slice_strategy 和 slice_random 确定标记的切片数
        slice1 = self.labelidx

        for i in slice1:
            label[:, :, i] = labeltmp[:, :, i]  # 只赋值标记的切片部分

        # 对所有数据执行强增强 双重随机增强输入扰动
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        # 逐片进行强增强处理
        image_strong_slices = []
        label_strong_slices = []

        for i in range(image.shape[-1]):
            image_slice = image[:, :, i]
            label_slice = label[:, :, i]

            image_strong, label_strong = cutout_gray(image_slice, label_slice, p=0.5)

            image_strong_slices.append(image_strong)
            label_strong_slices.append(label_strong)

        image_strong = np.stack(image_strong_slices, axis=-1)
        label_strong = np.stack(label_strong_slices, axis=-1)
        image_strong = color_jitter(image_strong).type("torch.FloatTensor")
        img_tensor = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(label)
        mask_strong = torch.from_numpy(label_strong)

        # 将类别权重包含在返回的 sample 中
        sample = {
            'image': img_tensor,
            'label': mask_tensor,
            'image_strong': image_strong,
            'label_strong': mask_strong,
            'slice1': slice1,
            # 'class_weights': self.class_weights  # 添加类别权重到 sample
        }

        return sample


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        if 'image' in sample and 'label' in sample:
            image, label = sample['image'], sample['label']
            # pad the sample if necessary
            if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                    self.output_size[2]:
                pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
                ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
                pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
                image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

            (w, h, d) = image.shape
            # if np.random.uniform() > 0.33:
            #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
            #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
            # else:
            w1 = np.random.randint(0, w - self.output_size[0])
            h1 = np.random.randint(0, h - self.output_size[1])
            d1 = np.random.randint(0, d - self.output_size[2])

            label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label}
        elif 'image_strong' in sample and 'label_strong' in sample:
            image_strong, label_strong = sample['image_strong'], sample['label_strong']
            # pad the sample if necessary
            if label_strong.shape[0] <= self.output_size[0] or label_strong.shape[1] <= self.output_size[1] or label_strong.shape[2] <= \
                    self.output_size[2]:
                pw = max((self.output_size[0] - label_strong.shape[0]) // 2 + 3, 0)
                ph = max((self.output_size[1] - label_strong.shape[1]) // 2 + 3, 0)
                pd = max((self.output_size[2] - label_strong.shape[2]) // 2 + 3, 0)
                image_strong = np.pad(image_strong, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                label_strong = np.pad(label_strong, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

            (w, h, d) = image_strong.shape
            # if np.random.uniform() > 0.33:
            #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
            #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
            # else:
            w1 = np.random.randint(0, w - self.output_size[0])
            h1 = np.random.randint(0, h - self.output_size[1])
            d1 = np.random.randint(0, d - self.output_size[2])

            label_strong = label_strong[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            image_strong = image_strong[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image_strong': image_strong, 'label_strong':  label_strong}

class MMRandomCrop_Sparse(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, slice1, slice2 = sample['image'], sample['label'], sample['slice1']
        #image, label, slice1 = sample['image'], sample['label'], sample['slice1']
        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])
        weight = np.zeros_like(image)
        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        #weight[:]=1

        for i in slice1:
            weight[:, :, i + 3] = 1
        weight = weight[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label, 'weight': weight}


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}




class MMRandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        if 'weight' in sample:
            image, label, weight = sample['image'], sample['label'], sample['weight']
            k = np.random.randint(0, 4)
            image = np.rot90(image, k)
            label = np.rot90(label, k)
            weight = np.rot90(weight, k)
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
            weight = np.flip(weight, axis=axis).copy()
            return {'image': image, 'label': label, 'weight': weight}
        else:
            image, label = sample['image'], sample['label']
            k = np.random.randint(0, 4)
            image = np.rot90(image, k)
            label = np.rot90(label, k)
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()

            return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2 * self.sigma,
                        2 * self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label, 'onehot_label': onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        elif 'weight' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'weight': torch.from_numpy(sample['weight'])}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


class CenterCrop_2(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def _get_transform(self, label):
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)

            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = label.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        def do_transform(x):
            if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[
                2]:
                x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            x = x[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return x

        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]


class ToTensor_2(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample[0]
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        sample = [image] + [*sample[1:]]
        return [torch.from_numpy(s.astype(np.float32)) for s in sample]


class make_data_3d(Dataset):
    def __init__(self, imgs, plabs, masks, labs, patch_size):
        self.img = [img.squeeze() for img in imgs]
        self.plab = [np.squeeze(lab) for lab in plabs]
        self.mask = [np.squeeze(mask) for mask in masks]
        self.lab = [np.squeeze(lab) for lab in labs]
        self.num = len(self.img)
        self.tr_transform = Compose([
            CenterCrop_2(patch_size),
            # RandomNoise(),
            ToTensor_2()
        ])

    def __getitem__(self, idx):
        samples = self.img[idx], self.plab[idx], self.mask[idx], self.lab[idx]
        # pdb.set_trace()
        samples = self.tr_transform(samples)
        imgs, plabs, masks, labs = samples
        return imgs, plabs.long(), masks.float(), labs.long()

    def __len__(self):
        return self.num
