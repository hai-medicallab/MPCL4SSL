# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math
import os
import pickle
import random

import numpy as np
import torch
from einops import rearrange
from scipy.stats import wasserstein_distance
from torch import nn
from torch.utils.data.sampler import Sampler

import networks


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def load_model(path):
    """Loads model and return it without DataParallel table."""
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)

        # size of the top layer
        N = checkpoint['state_dict']['top_layer.bias'].size()

        # build skeleton of the model
        sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()
        model = models.__dict__[checkpoint['arch']](sobel=sob, out=int(N[0]))

        # deal with a dataparallel table
        def rename_key(key):
            if not 'module' in key:
                return key
            return ''.join(key.split('.module'))

        checkpoint['state_dict'] = {rename_key(key): val
                                    for key, val
                                    in checkpoint['state_dict'].items()}

        # load weights
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        size_per_pseudolabel = int(self.N / len(self.images_lists)) + 1
        res = np.zeros(size_per_pseudolabel * len(self.images_lists))

        for i in range(len(self.images_lists)):
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res[i * size_per_pseudolabel: (i + 1) * size_per_pseudolabel] = indexes

        np.random.shuffle(res)
        return res[:self.N].astype('int')

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.N


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


class Logger():
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)





def js_divergence(p, q):
    """
    计算 Jensen-Shannon Divergence
    :param p: 第一个分布
    :param q: 第二个分布
    :return: JS 散度
    """
    p = p + 1e-10  # 避免对数计算中的零值
    q = q + 1e-10
    m = 0.5 * (p + q)
    return 0.5 * (torch.nn.functional.kl_div(p.log(), m, reduction='sum') +
                  torch.nn.functional.kl_div(q.log(), m, reduction='sum'))



def LSUS(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, label_batch, label_batch_strong, args,
              label_idx):
    # 自适应双向置换补丁 (有监督数据)
    patches_supervised_1 = rearrange(outputs1_max[label_idx], 'b (h p1) (w p2) -> b (h w) (p1 p2)',
                                     p1=args.patch_size, p2=args.patch_size)
    patches_supervised_2 = rearrange(outputs2_max[label_idx], 'b (h p1) (w p2) -> b (h w) (p1 p2)',
                                     p1=args.patch_size, p2=args.patch_size)

    image_patch_supervised_1 = rearrange(volume_batch.squeeze(1)[label_idx], 'b (h p1) (w p2) -> b (h w)(p1 p2)',
                                         p1=args.patch_size, p2=args.patch_size)
    image_patch_supervised_2 = rearrange(volume_batch_strong.squeeze(1)[label_idx], 'b (h p1) (w p2) -> b (h w)(p1 p2)',
                                         p1=args.patch_size, p2=args.patch_size)

    label_patch_supervised_1 = rearrange(label_batch[label_idx], 'b (h p1) (w p2) -> b (h w)(p1 p2)',
                                         p1=args.patch_size, p2=args.patch_size)
    label_patch_supervised_2 = rearrange(label_batch_strong[label_idx], 'b (h p1) (w p2) -> b (h w)(p1 p2)',
                                         p1=args.patch_size, p2=args.patch_size)

    # 分别计算两个集合中的有效补丁位置
    valid_indices_1 = [(torch.mean(label_patch_supervised_1[i].detach(), dim=1) != 0).nonzero(as_tuple=True)[0]
                       for i in range(len(label_idx))]
    valid_indices_2 = [(torch.mean(label_patch_supervised_2[i].detach(), dim=1) != 0).nonzero(as_tuple=True)[0]
                       for i in range(len(label_idx))]

    # 计算补丁均值并分别排序
    patches_mean_1 = [torch.mean(patches_supervised_1[i][valid_indices_1[i]], dim=1) for i in range(len(label_idx))]
    patches_mean_2 = [torch.mean(patches_supervised_2[i][valid_indices_2[i]], dim=1) for i in range(len(label_idx))]

    sorted_indices_1 = [valid_indices_1[i][torch.argsort(patches_mean_1[i])] for i in range(len(label_idx))]
    sorted_indices_2 = [valid_indices_2[i][torch.argsort(patches_mean_2[i])] for i in range(len(label_idx))]

    # 双向复制内容

    for i in range(len(label_idx)):
        if 2 <= len(sorted_indices_1[i]) == len(sorted_indices_2[i]) >= 2:
            half_len_1 = len(sorted_indices_1[i]) // 2
            half_len_2 = len(sorted_indices_2[i]) // 2
            # 将 sorted_indices_1[i] 的后一半补丁覆盖到 sorted_indices_2[i] 的前一半位置
            for j in range(half_len_1):
                image_patch_supervised_2[i][sorted_indices_2[i][j]] = \
                    image_patch_supervised_1[i][sorted_indices_1[i][-j - 1]].clone()
                label_patch_supervised_2[i][sorted_indices_2[i][j]] = \
                    label_patch_supervised_1[i][sorted_indices_1[i][-j - 1]].clone()

            # 将 sorted_indices_2[i] 的后一半补丁覆盖到 sorted_indices_1[i] 的前一半位置
            for j in range(half_len_2):
                image_patch_supervised_1[i][sorted_indices_1[i][j]] = \
                    image_patch_supervised_2[i][sorted_indices_2[i][-j - 1]].clone()
                label_patch_supervised_1[i][sorted_indices_1[i][j]] = \
                    label_patch_supervised_2[i][sorted_indices_2[i][-j - 1]].clone()
        else:
            pass

        # 还原补丁为图像和标签形式
        image_patch_supervised = torch.cat([image_patch_supervised_1, image_patch_supervised_2], dim=0)
        label_patch_supervised = torch.cat([label_patch_supervised_1, label_patch_supervised_2], dim=0)

    # 还原为完整图像和标签形式
    image_last = rearrange(image_patch_supervised, 'b (h w) (p1 p2) -> b (h p1) (w p2)',
                           h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)
    label_last = rearrange(label_patch_supervised, 'b (h w) (p1 p2) -> b (h p1) (w p2)',
                           h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)

    return image_last, label_last


def USUS(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, args, unlabel_idx):
    # 自适应双向置换补丁 (无监督数据)
    patches_1 = rearrange(outputs1_max[unlabel_idx], 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=args.patch_size,
                          p2=args.patch_size)
    patches_2 = rearrange(outputs2_max[unlabel_idx], 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=args.patch_size,
                          p2=args.patch_size)
    image_patch_1 = rearrange(volume_batch.squeeze(1)[unlabel_idx], 'b (h p1) (w p2) -> b (h w) (p1 p2)',
                              p1=args.patch_size, p2=args.patch_size)
    image_patch_2 = rearrange(volume_batch_strong.squeeze(1)[unlabel_idx], 'b (h p1) (w p2) -> b (h w) (p1 p2)',
                              p1=args.patch_size, p2=args.patch_size)

    # 计算补丁均值
    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    # 获取自适应阈值，选取重要补丁
    top_k = int(math.sqrt(patches_mean_1.shape[1]))


    patches_mean_1_topk_values, patches_mean_1_topk_indices = patches_mean_1.topk(top_k, dim=1)
    patches_mean_2_topk_values, patches_mean_2_topk_indices = patches_mean_2.topk(top_k, dim=1)

    # 获取后topk个补丁
    patches_mean_1_bottomk_values, patches_mean_1_bottomk_indices = patches_mean_1.topk(top_k, dim=1, largest=False)
    patches_mean_2_bottomk_values, patches_mean_2_bottomk_indices = patches_mean_2.topk(top_k, dim=1, largest=False)

    for i in range(len(unlabel_idx)):
        # 使用JS散度确定image_patch_1的替换
        used_indices_1 = set()  # 记录已使用的索引
        for j in range(top_k):
            p1 = patches_mean_1[i, patches_mean_1_topk_indices[i, j]].softmax(dim=-1)
            best_js = float('inf')
            for k in range(top_k):
                if k not in used_indices_1:
                    q1 = patches_mean_2[i, patches_mean_2_bottomk_indices[i, k]].softmax(dim=-1)
                    js_value = js_divergence(p1, q1)  # 计算JS散度
                    if js_value < best_js:  # 更小的JS散度
                        best_js = js_value
                        best_index_1 = patches_mean_2_bottomk_indices[i, k]

            image_patch_1[i][best_index_1] = image_patch_2[i][patches_mean_1_topk_indices[i, j]].clone()
            used_indices_1.add(best_index_1)

        # 使用JS散度确定image_patch_2的替换
        used_indices_2 = set()  # 记录已使用的索引
        for j in range(top_k):
            p2 = patches_mean_2[i, patches_mean_2_topk_indices[i, j]].softmax(dim=-1)
            best_js = float('inf')
            for k in range(top_k):
                if k not in used_indices_2:
                    q2 = patches_mean_1[i, patches_mean_1_bottomk_indices[i, k]].softmax(dim=-1)
                    js_value = js_divergence(p2, q2)  # 计算JS散度
                    if js_value < best_js:  # 更小的JS散度
                        best_js = js_value
                        best_index_2 = patches_mean_1_bottomk_indices[i, k]

            image_patch_2[i][best_index_2] = image_patch_1[i][patches_mean_2_topk_indices[i, j]].clone()
            used_indices_2.add(best_index_2)

    # 还原补丁为图像形式
    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    image_patch_last = rearrange(image_patch, 'b (h w)(p1 p2) -> b (h p1) (w p2)', h=args.h_size, w=args.w_size,
                                 p1=args.patch_size, p2=args.patch_size)
    return image_patch_last
