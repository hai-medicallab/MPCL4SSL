import math
import os
import sys

import h5py
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from networks.vnet import VNet
# from networks.unet import UNet
from utils import ramps, losses
from utils.util import  LSUS,USUS
from dataloaders.dataset import X
from networks.config import train_para_set

from networks.unet import UNet, UNet_2d
from utils.test_util import test_all_case

args = train_para_set()
parent_dir = args.parent_dir
train_data_path = parent_dir
snapshot_path = args.snapshot_path
batch_size = args.batch_size
max_iterations = args.max_iterations
base_lr = args.base_lr

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


if __name__ == "__main__":
    # make logger file
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.sliceseed)
    np.random.seed(args.seed)
    # 加载数据集 仅进行稀疏注释操作
    num_classes = args.num_classes
    # db_train1 = ACDC(base_dir=train_data_path,
    #                       split=args.split,
    #                       num=args.num,
    #                       slice_strategy=args.slice_strategy
    #                       )

    db_train1 = X(base_dir=train_data_path,
                                  split=args.split,
                                  num=args.num,
                                  slice_strategy=args.slice_strategy
                                  )


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    trainloader1 = DataLoader(db_train1, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              worker_init_fn=worker_init_fn)


    def create_3dmodel(ema=False):
        net = VNet(n_channels=1, n_classes=num_classes, normalization='groupnorm', has_dropout=True)
        model = net.cuda()
        return model


    def create_2dmodel(ema=False):
        # Network definition
        net = UNet(1, num_classes)
        model = net.cuda()
        return model


    # model1 = create_3dmodel()
    model2 = create_2dmodel()
    model3 = create_3dmodel()
    # model1.train()
    model2.train()
    model3.train()


    # optimizer1 = optim.Adam(model1.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer3 = optim.SGD(model3.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader1)))
    # 继续训练加载器
    continue_train = args.continue_train
    continue_train_iter = args.continue_train_iter
    epoch_num = args.epoch_num
    iter_num = 0
    if continue_train == 1:
        # checkpoint1 = torch.load(snapshot_path + '/model1_iter_{}.pth'.format(continue_train_iter),
        #                          map_location=torch.device("cuda"))  # 先反序列化模型
        checkpoint2 = torch.load(snapshot_path + '/model2_iter_{}.pth'.format(continue_train_iter),
                                 map_location=torch.device("cuda"))  # 先反序列化模型
        checkpoint3 = torch.load(snapshot_path + '/model3_iter_{}.pth'.format(continue_train_iter),
                                 map_location=torch.device("cuda"))  # 先反序列化模型
        # model1.load_state_dict(checkpoint1)
        model2.load_state_dict(checkpoint2)
        model3.load_state_dict(checkpoint3)
        iter_num = continue_train_iter
    if continue_train == 0:
        iter_num = 0

    max_epoch = 520
    lr_ = base_lr
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    best_performance2 = 0.0
    best_performance3 = 0.0
    for epoch_num in tqdm(range(max_epoch), ncols=70):  #  迭代并显示进度条
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader1):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))

            volume_batch1, label_batch1, volume_batch_strong, label_batch_strong, sclie_idx = (
                sampled_batch['image'], sampled_batch['label'],
                sampled_batch['image_strong'], sampled_batch['label_strong'],
                sampled_batch['slice1'])
            # 所有输入cuda化除了标签索引
            volume_batch1 = volume_batch1.transpose(1, 4)
            volume_batch1 = volume_batch1.squeeze(4)
            volume_batch1 = volume_batch1.transpose(0, 1)

            label_batch1 = label_batch1.unsqueeze(0)
            label_batch1 = label_batch1.transpose(1, 4)
            label_batch1 = label_batch1.squeeze(4)
            label_batch1 = label_batch1.transpose(0, 1)
            label_batch1 = label_batch1.squeeze(1)

            label_batch_strong = label_batch_strong.unsqueeze(0)
            label_batch_strong = label_batch_strong.transpose(1, 4)
            label_batch_strong = label_batch_strong.squeeze(4)
            label_batch_strong = label_batch_strong.transpose(0, 1)
            label_batch_strong = label_batch_strong.squeeze(1)

            volume_batch_strong = volume_batch_strong.transpose(0, 1)
            volume_batch1, label_batch1 = volume_batch1.float().cuda(), label_batch1.float().cuda()
            volume_batch_strong, label_batch_strong = volume_batch_strong.float().cuda(), label_batch_strong.float().cuda()
            sclie_idx_np = [mask.item() for mask in sclie_idx]  # 使用 item() 提取单个元素

            # 现在 sclie_idx_np 是一个包含整数的 NumPy 数组
            label_idx = np.array(sclie_idx_np)  # 将列表转换为 NumPy 数组
            # sclie_idx 表示标签索引  unlabel_idx表示非标签索引
            unlabel_idx = [i for i in range(volume_batch1.shape[0]) if i not in sclie_idx_np]

            # 双差异模型Unet+ VNet
            # 示例张量
            outputs2 = model2(volume_batch1)  # D C H W
            # # 抽取没GT标签的预测
            outputs2_unlabel = outputs2[unlabel_idx]

            outputs_soft2 = torch.softmax(outputs2, dim=1)
            outputs2_max = torch.max(outputs_soft2.detach(), dim=1)[0]
            pseudo_outputs2 = torch.argmax(outputs_soft2[unlabel_idx].detach(), dim=1, keepdim=False)

            v_all = volume_batch_strong.unsqueeze(4)
            v_all = v_all.transpose(0, 4)
            v_all_padded = F.pad(v_all, (0, 0))
            outputs3 = model3(v_all_padded)  # B C H W D

            # 修改 BCHWD为 DCHW
            outputs3 = outputs3.transpose(0, 4).squeeze(4)
            outputs3_unlabel = outputs3[unlabel_idx]
            outputs_soft3 = torch.softmax(outputs3, dim=1)
            outputs3_max = torch.max(outputs_soft3.detach(), dim=1)[0]
            pseudo_outputs3 = torch.argmax(outputs_soft3[unlabel_idx].detach(), dim=1, keepdim=False)

            # New Training Sample
            image_patch_supervised_last, label_patch_supervised_last = LSUS(outputs2_max, outputs3_max, volume_batch1,
                                                                             volume_batch_strong, label_batch1,
                                                                             label_batch_strong, args, label_idx)

            image_output_supervised_1 = model2(image_patch_supervised_last.unsqueeze(1))
            image_output_soft_supervised_1 = torch.softmax(image_output_supervised_1, dim=1)

            # 修改适配Vnet

            image_patch_supervised_last = image_patch_supervised_last.unsqueeze(1).unsqueeze(-1).transpose(0, 4)
            c = image_patch_supervised_last.size(4)
            image_patch_supervised_last_pad = F.pad(image_patch_supervised_last, (0, 32-c))
            image_output_supervised_2 = model3(image_patch_supervised_last_pad)
            image_output_supervised_2 = image_output_supervised_2[:, :, :, :, :c]
            image_output_supervised_2 = image_output_supervised_2.transpose(0, 4).squeeze(-1)

            image_output_soft_supervised_2 = torch.softmax(image_output_supervised_2, dim=1)

            # New Training Sample
            image_patch_last = USUS(outputs2_max, outputs3_max, volume_batch1, volume_batch_strong, args, unlabel_idx)

            # 双副网络输入利用无标签

            image_output_1 = model2(image_patch_last.unsqueeze(1))
            image_output_1 = image_output_1[:32, :, :, :]
            image_output_soft_1 = torch.softmax(image_output_1, dim=1)
            pseudo_image_output_1 = torch.argmax(image_output_soft_1.detach(), dim=1, keepdim=False)

            image_patch_last = image_patch_last.unsqueeze(1).unsqueeze(-1).transpose(0, 4)
            image_patch_last_pad = F.pad(image_patch_last, (0, 32-image_patch_last.size(4)))
            image_output_2 = model3(image_patch_last_pad)
            image_output_2 = image_output_2[:, :, :, :, :image_patch_last.size(4)]
            image_output_2 = image_output_2.transpose(0, 4).squeeze(-1)

            image_output_soft_2 = torch.softmax(image_output_2, dim=1)
            pseudo_image_output_2 = torch.argmax(image_output_soft_2.detach(), dim=1, keepdim=False)

            # First Step Loss

            # 双网络真实标签loss
            Loss2 = 0.5 * (ce_loss(outputs2[label_idx], label_batch1[label_idx].long()) + dice_loss(
                outputs_soft2[label_idx], label_batch1[label_idx].unsqueeze(1)))
            Loss3 = 0.5 * (ce_loss(outputs3[label_idx], label_batch_strong[label_idx].long()) + dice_loss(
                outputs_soft3[label_idx], label_batch_strong[label_idx].unsqueeze(1)))

            pseudo_supervision2 = dice_loss(outputs_soft2[unlabel_idx], pseudo_outputs3.unsqueeze(1))
            pseudo_supervision3 = dice_loss(outputs_soft3[unlabel_idx], pseudo_outputs2.unsqueeze(1))

            # Second Step Loss
            if iter_num > 20000:
                loss5 = 0
                loss6 = 0
            else:
                loss5 = 0.5 * (ce_loss(image_output_supervised_1, label_patch_supervised_last.long()) + dice_loss(
                    image_output_soft_supervised_1, label_patch_supervised_last.unsqueeze(1)))
                loss6 = 0.5 * (ce_loss(image_output_supervised_2, label_patch_supervised_last.long()) + dice_loss(
                    image_output_soft_supervised_2, label_patch_supervised_last.unsqueeze(1)))

            pseudo_supervision5 = dice_loss(image_output_soft_1, pseudo_image_output_2.unsqueeze(1))
            pseudo_supervision6 = dice_loss(image_output_soft_2, pseudo_image_output_1.unsqueeze(1))

            # Total Loss
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            model2_loss = Loss2 + 2 * loss5 + consistency_weight * (pseudo_supervision2 + pseudo_supervision5)
            model3_loss = Loss3 + 2 * loss6 + consistency_weight * (pseudo_supervision3 + pseudo_supervision6)
            # 双差异网络Loss
            # loss = model1_loss + model2_loss + model3_loss
            loss = model2_loss + model3_loss
            print('iter {},loss {},  model2_loss {}, model3_loss {}'.format(iter_num, loss, model2_loss, model3_loss))

            optimizer2.zero_grad()
            optimizer3.zero_grad()
            loss.backward()
            optimizer2.step()
            optimizer3.step()

            iter_num = iter_num + 1
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer3.param_groups:
                param_group['lr'] = lr_
            if iter_num % 500 == 0 and max_iterations >= iter_num >= 1000:
                with open(parent_dir + "test" + '.txt', 'r') as f:
                    image_list = f.readlines()
                image_list = [parent_dir + item.replace('\n', '').split(",")[0] + '.h5' for item in image_list]
                model2.eval()
                avg_dice, avg_iou, avg_hd, avg_asd, dice_list = test_all_case(model2, image_list,
                                                                              num_classes=num_classes,
                                                                              patch_size=(224, 224, 32), stride_xy=16,
                                                                              stride_z=4,
                                                                              save_result=False,
                                                                              a="Unet"
                                                                              )
                performance2 = np.mean(avg_dice, axis=0)
                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path1 = snapshot_path + '/model2_iter_' + str(iter_num) + '.pth'
                    torch.save(model2.state_dict(), save_mode_path1)
                    logging.info("save Unet to {}".format(save_mode_path1))
                logging.info('iteration %d : Unet_mean_dice : %f' % (iter_num, performance2))
                model2.train()

                model3.eval()
                avg_dice, avg_iou, avg_hd, avg_asd, dice_list = test_all_case(model3, image_list,
                                                                              num_classes=num_classes,
                                                                              patch_size=(224, 224, 32), stride_xy=16,
                                                                              stride_z=4,
                                                                              save_result=True,
                                                                              a="Vnet"
                                                                              )
                performance3 = np.mean(avg_dice, axis=0)
                if performance3 > best_performance3:
                    best_performance3 = performance3
                    save_mode_path2 = snapshot_path + '/model3_iter_' + str(iter_num) + '.pth'
                    torch.save(model3.state_dict(), save_mode_path2)
                    logging.info("save Vnet to {}".format(save_mode_path2))
                logging.info('iteration %d : Vnet_mean_dice : %f' % (iter_num, performance3))
                model3.train()

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break

    # writer.close()
