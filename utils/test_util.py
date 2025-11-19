import os

import cv2
import h5py
import math
import nibabel as nib
import numpy as np
from jinja2.nodes import Slice
from matplotlib import pyplot as plt
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime


import numpy as np
import torch
import torch.nn.functional as F
import math
import SimpleITK as sitk
def save_visualization(prediction, label):
    rand_index = np.random.randint(prediction.shape[2])  # 随机选择一个切片索引

    rand_slice_out = prediction[:, :, rand_index]  # 直接使用 NumPy 数组
    plt.imshow(rand_slice_out, cmap='jet')  # 使用伪彩色映射
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path1 = f"./Infe_result/{timestamp}_pred_image.png"
    plt.savefig(path1)
    plt.close()

    rand_label_slice = label[:, :, rand_index]  # 直接使用 NumPy 数组
    plt.imshow(rand_label_slice, cmap='jet')  # 使用伪彩色映射
    path2 = f"./Infe_result/{timestamp}_label_image.png"
    plt.savefig(path2)
    plt.close()




def overlay_cam_and_label_contour(slice_data, cam, label_data, alpha=0.3, contour_color=(0, 0, 255),
                                  contour_thickness=1, contour_alpha=0.7):
    """
    将Grad-CAM热力图叠加到2D切片上，并在上面绘制带透明度的标签轮廓

    参数:
    slice_data: 2D原始图像 [H,W]，可为float
    cam: 2D热力图 [H,W]，归一化到0-1
    label_data: 2D标签图像 [H,W]，非零值表示标签区域
    alpha: 热力图透明度
    contour_color: 轮廓颜色，BGR格式，默认为红色
    contour_thickness: 轮廓线粗细
    contour_alpha: 轮廓透明度

    返回:
    BGR叠加图，包含热力图和标签轮廓
    """
    # 归一化原始图像
    original = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())

    original = (original * 255).astype(np.uint8)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    # 生成热力图
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # 叠加热力图到原始图像
    superimposed = cv2.addWeighted(original_rgb, 1 - alpha, heatmap, alpha, 0)

    # 提取标签轮廓
    # 将标签数据转换为二值图像
    _, binary_label = cv2.threshold(label_data.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(binary_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个透明图层用于绘制轮廓
    contour_layer = np.zeros_like(superimposed, dtype=np.uint8)
    cv2.drawContours(contour_layer, contours, -1, contour_color, contour_thickness)

    # 将轮廓图层与叠加图像融合，实现轮廓透明度
    result = cv2.addWeighted(superimposed, 1.0, contour_layer, contour_alpha, 0)

    return result
# def overlay_cam_on_slice(slice_data, cam, alpha=0.3):
#     """
#     将Grad-CAM热力图叠加到2D切片上
#     slice_data: 2D原始图像 [H,W]，可为float
#     cam: 2D热力图 [H,W]，归一化到0-1
#     alpha: 热力图透明度
#     return: BGR叠加图
#     """
#     # 归一化原始图像
#     original = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
#     original = (original * 255).astype(np.uint8)
#     original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
#
#     # 生成热力图
#     heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
#
#     # 叠加图像
#     superimposed = cv2.addWeighted(original_rgb, 1 - alpha, heatmap, alpha, 0)
#     return superimposed
#
#
# import numpy as np
# import cv2
#
#
# def overlay_cam_on_label(label_data, cam, alpha=0.3):
#     """
#     将Grad-CAM热力图叠加到标签图像上，标签区域显示为红色，背景为黑色
#
#     参数:
#     label_data: 2D标签图像 [H,W]，非零值表示标签区域
#     cam: 2D热力图 [H,W]，归一化到0-1
#     alpha: 热力图透明度
#
#     返回:
#     BGR叠加图
#     """
#     # 创建红色标签图像（背景黑色，标签红色）
#     red_label = np.zeros((label_data.shape[0], label_data.shape[1], 3), dtype=np.uint8)
#     # 将标签区域设置为红色
#     red_label[label_data > 0] = [0, 0, 255]  # OpenCV使用BGR格式，所以红色是[0, 0, 255]
#
#     # 生成热力图
#     heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
#
#     # 叠加图像
#     superimposed = cv2.addWeighted(red_label, 1 - alpha, heatmap, alpha, 0)
#     return superimposed
def test_all_case(net1, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=16, stride_z=4, save_result=True,
                  test_save_path=None, preproc_fn=None, a="Unet"):
    total_metric = 0.0
    totaldice = 0.0
    totalasd = 0.0
    totalhd = 0.0
    totaliou = 0.0
    dicelist = []
    # delete after
    #slice_random=[5,0,3,11,3,7,9,3 , 5,  8 , 4,  7,  6,  8,  8, 10]
    #img_idx=0
    for image_path in tqdm(image_list):
        print(image_path)
        id = image_path.split('/')[-1]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]

        image = (image - np.mean(image)) / np.std(image)

        if preproc_fn is not None:
            image = preproc_fn(image)
        # prediction, score_map, cam_map = test_single_case_gradcam_3d(net1, image, stride_xy, patch_size=patch_size, stride_z=stride_z)

        for c in range(num_classes):
            prediction, score_map, cam_map = test_single_case_gradcam_2d(
                net1, image, stride_xy, patch_size1=(224, 224), num_classes=num_classes, target_class=c
            )

            # 每个类别单独一个文件夹
            cam_save_dir = os.path.join(test_save_path, f"{id}_cam_overlay_class{c}")
            os.makedirs(cam_save_dir, exist_ok=True)

            for z in range(image.shape[2]):
                slice_img = image[:, :, z]
                slice_label = (label[:, :, z] == c).astype(np.uint8)  # 提取该类别的二值标签
                cam_slice = cam_map[:, :, z]

                overlay_img = overlay_cam_and_label_contour(slice_img, cam_slice, slice_label)
                cv2.imwrite(os.path.join(cam_save_dir, f"slice_image_{z:03d}.png"), overlay_img)

        # cam_itk = sitk.GetImageFromArray(cam_map.astype(np.float32))
        # cam_itk.SetSpacing((1, 1, 10))
        # sitk.WriteImage(cam_itk, os.path.join(test_save_path, f"{id}_cam.nii.gz"))
        # 可视化预测
        # save_visualization(prediction, label)
        classdice = []
        classiou = []
        classasd = []
        classhd = []
        for c in range(1, num_classes):
            if np.count_nonzero(prediction == c) == 0 and np.count_nonzero(label[:] == c) != 0:
                curdice, curiou, curhd, curasd = (0, 0, 50, 50)
            elif np.count_nonzero(prediction == c) == 0 and np.count_nonzero(label[:] == c) == 0:
                curdice, curiou, curhd, curasd = (1, 1, 0, 0)
            else:
                curdice, curiou, curhd, curasd = calculate_metric_percase(prediction == c, label[:] == c)
            classdice.append(curdice)
            classiou.append(curiou)
            classhd.append(curhd)
            classasd.append(curasd)
        classdice.append(np.mean(classdice))
        classiou.append(np.mean(classiou))
        classhd.append(np.mean(classhd))
        classasd.append(np.mean(classasd))
        totaldice += np.asarray(classdice)
        totalhd += np.asarray(classhd)
        totaliou += np.asarray(classiou)
        totalasd += np.asarray(classasd)
        print(classdice)
        print(classiou)
        print(classhd)
        print(classasd)
        dicelist.append(classdice)

        # if save_result:
        #     nib.save(nib.Nifti1Image(image.astype(np.float32), np.eye(4)), test_save_path + id + "_img.nii.gz")
        #     nib.save(nib.Nifti1Image(score_map.astype(np.float32), np.eye(4)), test_save_path + id + "_prob.nii.gz")
        #     nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + id + "_pred.nii.gz")
        #     nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + id + "_gt.nii.gz")
    avg_dice = totaldice / len(image_list)
    avg_iou = totaliou / len(image_list)
    avg_hd = totalhd / len(image_list)
    avg_asd = totalasd / len(image_list)
    print('average metric is {},{},{},{}'.format(avg_dice, avg_iou, avg_hd, avg_asd))
    print(dicelist)
    return avg_dice, avg_iou, avg_hd, avg_asd, dicelist




def test_single_case_gradcam_2d(net1, image, stride_xy, patch_size1=(224, 224), num_classes=5,target_class = 1):
    """
    2D patch-wise inference with Grad-CAM support for 2D UNet.
    image: [W,H,D]
    patch_size: (patch_w, patch_h)
    Returns:
        label_map: [W,H,D] predicted label
        score_map: [W,H,D] max softmax probability
        cam_map:   [W,H,D] Grad-CAM map for target class
    """

    w, h, d = image.shape

    # --- padding if needed ---
    w_pad = max(patch_size1[0] - w, 0)
    h_pad = max(patch_size1[1] - h, 0)
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2

    if w_pad > 0 or h_pad > 0:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (0, 0)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    # --- prepare maps ---
    score_map = np.zeros((num_classes, ww, hh, dd), dtype=np.float32)
    cnt = np.zeros((ww, hh, dd), dtype=np.float32)
    cam_map = np.zeros((ww, hh, dd), dtype=np.float32)
    cam_count = np.zeros((ww, hh, dd), dtype=np.float32)

    # --- register hooks ---
    activations = {}
    gradients = {}
    target_layer = net1.decoder.out_conv if hasattr(net1, 'decoder') and hasattr(net1.decoder, 'out_conv') else \
    list(net1.modules())[-2]

    def forward_hook(module, input, output):
        activations['layer'] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients['layer'] = grad_output[0].detach()

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    net1.eval()

    # --- sliding window over W,H, loop over slices ---
    sx = math.ceil((ww - patch_size1[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size1[1]) / stride_xy) + 1

    for z in range(dd):
        slice_img = image[:, :, z]
        for x in range(sx):
            xs = min(stride_xy * x, ww - patch_size1[0])
            for y in range(sy):
                ys = min(stride_xy * y, hh - patch_size1[1])
                test_patch = slice_img[xs:xs + patch_size1[0], ys:ys + patch_size1[1]]
                test_patch = torch.from_numpy(test_patch[np.newaxis, np.newaxis, :, :]).float().cuda()
                test_patch.requires_grad = True

                # --- forward ---
                outputs = net1(test_patch)  # [B,C,H,W]
                outputs = outputs.squeeze(0)  # remove batch
                y = F.softmax(outputs, dim=0).cpu().detach().numpy()  # [C,H,W]

                # --- accumulate score map ---
                score_map[:, xs:xs + patch_size1[0], ys:ys + patch_size1[1], z] += y
                cnt[xs:xs + patch_size1[0], ys:ys + patch_size1[1], z] += 1

                # --- Grad-CAM ---
                net1.zero_grad()
                  # 假设前景
                outputs[target_class].mean().backward()

                acts = activations['layer'][0].cpu().numpy()  # [C,H,W]
                grads = gradients['layer'][0].cpu().numpy()  # [C,H,W]
                weights = np.mean(grads, axis=(1, 2))  # global avg pool
                cam = np.sum(weights[:, np.newaxis, np.newaxis] * acts, axis=0)
                cam = np.maximum(cam, 0)
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

                cam_map[xs:xs + patch_size1[0], ys:ys + patch_size1[1], z] += cam
                cam_count[xs:xs + patch_size1[0], ys:ys + patch_size1[1], z] += 1

    # --- normalize ---
    score_map = score_map / np.expand_dims(cnt, axis=0)
    cam_map = cam_map / (cam_count + 1e-8)

    label_map = np.argmax(score_map, axis=0)
    score_map_max = np.max(score_map, axis=0)

    # --- remove hooks ---
    handle_forward.remove()
    handle_backward.remove()

    # --- remove padding ---
    if w_pad > 0 or h_pad > 0:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, :]
        score_map_max = score_map_max[wl_pad:wl_pad + w, hl_pad:hl_pad + h, :]
        cam_map = cam_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, :]

    return label_map, score_map_max, cam_map


def test_single_case_gradcam_3d(net1, image, stride_xy, stride_z, patch_size,
                                  num_classes=2, target_class=1):
    """
    3D patch-wise inference with Grad-CAM support for Vnet.
    image: [W,H,D]
    patch_size: (pw, ph, pd)
    Returns:
        label_map: [W,H,D] predicted label
        score_map: [W,H,D] max softmax probability
        cam_map:   [W,H,D] Grad-CAM map for target class
    """
    w, h, d = image.shape

    # ---- padding if needed ----
    add_pad = False
    w_pad = max(patch_size[0] - w, 0)
    h_pad = max(patch_size[1] - h, 0)
    d_pad = max(patch_size[2] - d, 0)
    if w_pad > 0 or h_pad > 0 or d_pad > 0:
        add_pad = True
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)],
                       mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    # ---- prepare maps ----
    score_map = np.zeros((num_classes, ww, hh, dd), dtype=np.float32)
    cnt = np.zeros((ww, hh, dd), dtype=np.float32)
    cam_map = np.zeros((ww, hh, dd), dtype=np.float32)
    cam_count = np.zeros((ww, hh, dd), dtype=np.float32)

    # ---- register hooks ----
    activations, gradients = {}, {}

    target_layer = net1.decoder.out_conv if hasattr(net1, 'decoder') and hasattr(net1.decoder, 'out_conv') else \
                   list(net1.modules())[-2]

    def forward_hook(module, input, output):
        activations['layer'] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients['layer'] = grad_output[0].detach()

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    net1.eval()

    # ---- sliding window ----
    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    for x in range(sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0],
                                   ys:ys + patch_size[1],
                                   zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                test_patch.requires_grad = True

                # ---- forward (Vnet 专用) ----
                test_patch_padded = F.pad(test_patch, (0, 0))  # 在 D 维上 pad
                outputs = net1(test_patch_padded)              # [B,C,W,H,D+pad]
                outputs = outputs[:, :, :, :, :patch_size[2]]  # 裁切到原始大小

                # ---- softmax ----
                y = F.softmax(outputs, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]  # [C,W,H,D]

                # ---- accumulate score map ----
                score_map[:, xs:xs + patch_size[0],
                             ys:ys + patch_size[1],
                             zs:zs + patch_size[2]] += y
                cnt[xs:xs + patch_size[0],
                    ys:ys + patch_size[1],
                    zs:zs + patch_size[2]] += 1

                # ---- Grad-CAM ----
                net1.zero_grad()
                outputs[0, target_class].mean().backward()

                acts = activations['layer'][0].cpu().numpy()  # [C,W,H,D]
                grads = gradients['layer'][0].cpu().numpy()   # [C,W,H,D]
                weights = np.mean(grads, axis=(1, 2, 3))      # GAP over 3D
                cam = np.sum(weights[:, None, None, None] * acts, axis=0)
                cam = np.maximum(cam, 0)
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

                cam_map[xs:xs + patch_size[0],
                        ys:ys + patch_size[1],
                        zs:zs + patch_size[2]] += cam
                cam_count[xs:xs + patch_size[0],
                          ys:ys + patch_size[1],
                          zs:zs + patch_size[2]] += 1

    # ---- normalize ----
    score_map = score_map / (np.expand_dims(cnt, axis=0) + 1e-8)
    cam_map = cam_map / (cam_count + 1e-8)

    label_map = np.argmax(score_map, axis=0)
    score_map_max = np.max(score_map, axis=0)

    # ---- remove hooks ----
    handle_forward.remove()
    handle_backward.remove()

    # ---- remove padding ----
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map_max = score_map_max[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        cam_map = cam_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]

    return label_map, score_map_max, cam_map

def extract_slices(volume_batch):
    B, C, H, W, D = volume_batch.shape
    slices = []

    for start_idx in range(D):
        slices_idx = start_idx
        slice = volume_batch[:, :, :, :, slices_idx].unsqueeze(1)

        slices.append(slice)
    slices = torch.cat(slices, dim=0)

    return slices


def test_single_case(net1, image, stride_xy, stride_z, patch_size, num_classes=1, modelname="Unet"):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant',
                       constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                # UNet
                if modelname == "Unet":
                    zs = min(stride_z * z, dd - patch_size[2])
                    test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                    test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                    test_patch = torch.from_numpy(test_patch).cuda()
                    volume_batch1 = test_patch.transpose(0, 4).squeeze(4)
                    outputs = net1(volume_batch1)
                    outputs = outputs.unsqueeze(4).transpose(0, 4)
                    y = F.softmax(outputs, dim=1)
                    y = y.cpu().data.numpy()
                    y = y[0, :, :, :, :]
                # #CATNet
                # zs = min(stride_z * z, dd - patch_size[8])
                # test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[8]]
                # test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                # test_patch = torch.from_numpy(test_patch).cuda()
                # volume_batch = test_patch.view(32, 1, 128, 128)  # 或者根据需要调整
                # outputs1 = net1(volume_batch)
                # y1 = outputs1.view(1, 5, 128, 128, 32)  # 变换形状
                # y = F.softmax(y1, dim=1)
                # y = y.cpu().data.numpy()
                # y = y[0, :, :, :, :]
                if modelname == "Vnet":
                    zs = min(stride_z * z, dd - patch_size[2])
                    test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                    test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                    test_patch = torch.from_numpy(test_patch).cuda()
                    test_patch_padded = F.pad(test_patch, (0, 0))  # 在最后一个维度 D 上填充12个0
                    # outputs3 = model3(volume_batch_strong)
                    y1 = net1(test_patch_padded)
                    y1 = y1[:, :, :, :, :80]
                    y = F.softmax(y1, dim=1)
                    y = y.cpu().data.numpy()
                    y = y[0, :, :, :, :]
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)
    score_map = np.max(score_map, axis=0)
    # total_params1 = sum(p1.numel() for p1 in net1.parameters())
    # # total_params2 = sum(p2.numel() for p2 in model2.parameters())
    # # total_params3 = sum(p3.numel() for p3 in model3.parameters())
    # # total_params = total_params2+total_params3+total_params1
    # total_params = total_params1
    # bytes_per_param = 4

    # # 计算总字节数
    # total_bytes = total_params * bytes_per_param
    # # 转换为兆字节（MB）和千字节（KB）
    # total_megabytes = total_bytes / (1024 * 1024)
    # print("Total parameters in MB:", total_megabytes)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        #score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num - 1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd
