import os

import numpy as np
import torch
import random

from torch.backends import cudnn
from torch.nn import GroupNorm

from networks.unet import UNet
from networks.vnet import VNet
from utils.test_util import test_all_case
from networks.config import test_para_set, train_para_set
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
FLAGS = test_para_set()
args = train_para_set()
parent_dir = FLAGS.parent_dir
snapshot_path = FLAGS.snapshot_path
test_save_path = FLAGS.test_save_path

num_classes = FLAGS.num_classes
with open(parent_dir + FLAGS.split + '_mmwhs.txt', 'r') as f:
    image_list = f.readlines()
image_list = [parent_dir + item.replace('\n', '').split(",")[0] + '.h5' for item in image_list]


def create_3dmodel(ema=False):
    # Network definition
    net = VNet(n_channels=1, n_classes=num_classes, normalization='groupnorm', has_dropout=True)
    # net = UNet3D(1, num_classes)
    # net = ResUNet(1, num_classes)
    # net = UNETR(1, num_classes, [196,196,92])
    model = net.cuda()

    return model


def create_2dmodel(ema=False):
    # Network definition
    net = UNet(1,num_classes)
    model = net.cuda()

    return model





def test_calculate_metric(epoch_num):
    net1 = create_2dmodel()
    # net1 = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False).cuda()
    # net1 = ResUNet(1, num_classes).cuda()
    checkpoint = torch.load(snapshot_path + 'model2_iter_1000.pth',
                            map_location=torch.device("cuda"))  # 先反序列化模型
    net1.load_state_dict(checkpoint)

    net1.eval()

    avg_dice, avg_iou, avg_hd, avg_asd, dice_list = test_all_case(net1, image_list, num_classes=num_classes,
                                                                  patch_size=(224, 224, 32), stride_xy=16, stride_z=4,
                                                                  save_result=True, test_save_path=test_save_path
                                                                  )

    return avg_dice, avg_iou, avg_hd, avg_asd, dice_list


if __name__ == '__main__':
    if FLAGS.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.sliceseed)
    np.random.seed(args.seed)

    maxmetric = 0
    maxi = -1
    path = snapshot_path + 'test_12slice.txt'
    for i in range(FLAGS.min_iteration, FLAGS.max_iteration + 1, FLAGS.iteration_step):
        avg_dice, avg_iou, avg_hd, avg_asd, dice_list = test_calculate_metric(i)
        strmetric = 'net' + str(FLAGS.modeleffe) + ": iter" + str(i) + ":\n" + str(avg_dice) + '\n' + str(
            avg_iou) + '\n' + str(avg_hd) + '\n' + str(avg_asd) + '\n'
        with open(path, "a") as f:
            f.writelines(strmetric)
        if avg_dice[-1] > maxmetric:
            maxi = i
            maxmetric = avg_dice[-1]
    print(maxmetric, "||", maxi)
    with open(path, "a") as f:
        f.writelines(str(maxi) + '\n')
    with open(parent_dir + (FLAGS.split).replace('valid', 'test') + '_mmwhs.txt', 'r') as f:
        image_list = f.readlines()
    image_list = [parent_dir + item.replace('\n', '').split(",")[0] + '.h5' for item in image_list]
    avg_dice, avg_iou, avg_hd, avg_asd, dice_list = test_calculate_metric(maxi)
    strmetric = 'net' + str(FLAGS.modeleffe) + ": iter" + str(maxi) + ":\n" + str(avg_dice) + '\n' + str(
        avg_iou) + '\n' + str(avg_hd) + '\n' + str(avg_asd) + '\n'
    with open(path, "a") as f:
        f.writelines(strmetric)
# 1PZ 2TZ 3Urethra 4Anterior
