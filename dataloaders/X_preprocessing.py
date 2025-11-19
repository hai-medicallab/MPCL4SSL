import cv2
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
from skimage import morphology, exposure
import scipy
from scipy import ndimage
import h5py
import torch
from scipy.ndimage import median_filter

v = "v1"
home_dir = r"/".join(os.getcwd().split("\\"))
parent_dir = os.path.dirname(os.path.dirname(home_dir))  # 获取当前目录的上一级目录路径


class CFG:
    do_spacing = False
    target_spacing = [1, 1, 1]

    do_reshape = True
    new_size = [224, 224, 32]


def space_resampling(roiImg, new_spacing, lbl=False):
    # print('old spacing: ', roiImg.GetSpacing())
    new_size = [int(old_sz * old_spc / new_spc) for old_sz, old_spc, new_spc in
                zip(roiImg.GetSize(), roiImg.GetSpacing(), new_spacing)]

    if lbl:
        resampled_sitk = sitk.Resample(roiImg, new_size, sitk.Transform(), sitk.sitkNearestNeighbor, roiImg.GetOrigin(),
                                       new_spacing, roiImg.GetDirection(), 0.0, roiImg.GetPixelIDValue())
    else:
        resampled_sitk = sitk.Resample(roiImg, new_size, sitk.Transform(), sitk.sitkLinear, roiImg.GetOrigin(),
                                       new_spacing, roiImg.GetDirection(), 0.0, roiImg.GetPixelIDValue())

    return resampled_sitk


def resampling(roiImg, new_size, lbl=False):
    new_spacing = [old_sz * old_spc / new_sz for old_sz, old_spc, new_sz in
                   zip(roiImg.GetSize(), roiImg.GetSpacing(), new_size)]
    if lbl:
        resampled_sitk = sitk.Resample(roiImg, new_size, sitk.Transform(), sitk.sitkNearestNeighbor, roiImg.GetOrigin(),
                                       new_spacing, roiImg.GetDirection(), 0.0, roiImg.GetPixelIDValue())
    else:
        resampled_sitk = sitk.Resample(roiImg, new_size, sitk.Transform(), sitk.sitkLinear, roiImg.GetOrigin(),
                                       new_spacing, roiImg.GetDirection(), 0.0, roiImg.GetPixelIDValue())

    return resampled_sitk


def get_expanded_bounding_box(mask, padding=0):
    # 获取非零值的索引（假设mask是二进制掩膜）
    non_zero_indices = np.argwhere(mask)
    if len(non_zero_indices) == 0:
        # 如果标签全为0，返回一个默认大小的边界框
        return [slice(0, mask.shape[dim]) for dim in range(mask.ndim)]

    min_coords = np.min(non_zero_indices, axis=0)
    max_coords = np.max(non_zero_indices, axis=0)

    # 扩展边界框，加入边距
    bbox = [slice(max(0, min_coord - padding), min(mask.shape[dim], max_coord + 1 + padding))
            for dim, (min_coord, max_coord) in enumerate(zip(min_coords, max_coords))]
    return bbox


def crop_image(image, bbox):
    slices = [slice(None)] * image.ndim
    for dim, bbox_slice in enumerate(bbox):
        slices[dim] = bbox_slice
    return image[tuple(slices)]


class MMRandomCrop_Sparse(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, slice1, slice2 = sample['image'], sample['label'], sample['slice1'], sample['slice2']
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
        for i in slice2:
            weight[:, i, :] = 1

        weight = weight[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label, 'weight': weight}


if __name__ == "__main__":
    basedir = parent_dir + '/MPCL/data/X/'
    for i in range(1, 96):
        path = os.path.join(basedir, 'img00' + str(i).zfill(2))
        path2 = os.path.join(basedir, 'label00' + str(i).zfill(2))
        ct = sitk.ReadImage(path + '.nii.gz')
        lbl = sitk.ReadImage(path2 + '.nii.gz')
        ct_array = sitk.GetArrayFromImage(ct)
        lbl_array = sitk.GetArrayFromImage(lbl)
        # 进行预处理
        padding = 50  # 定义边距
        # 获取扩展的边界框，即使标签全为0也会返回默认边界框
        bbox = get_expanded_bounding_box(lbl_array, padding)
        ct_list = []
        # 裁切CT图像和标签
        cropped_ct_array = crop_image(ct_array, bbox)
        cropped_lbl_array = crop_image(lbl_array, bbox)

        #归一化将值缩放到0到1之间
        min_value = np.min(cropped_ct_array)
        max_value = np.max(cropped_ct_array)
        cropped_normalized_image_array = (cropped_ct_array- min_value) / (max_value - min_value)

        # 将标准化后的图像转换回 SimpleITK 图像
        new_ct = sitk.GetImageFromArray(cropped_normalized_image_array)
        new_lbl = sitk.GetImageFromArray(cropped_lbl_array)

        if CFG.do_spacing:
            new_ct = space_resampling(new_ct, CFG.target_spacing, lbl=False)
            new_lbl = space_resampling(new_lbl, CFG.target_spacing, lbl=True)
        elif CFG.do_reshape:
            new_ct = resampling(new_ct, CFG.new_size, lbl=False)
            new_lbl = resampling(new_lbl, CFG.new_size, lbl=True)
        save_ct_array = sitk.GetArrayFromImage(new_ct)
        save_lbl_array = sitk.GetArrayFromImage(new_lbl)
        save_ct_array = save_ct_array.swapaxes(0, 2)
        save_lbl_array = save_lbl_array.swapaxes(0, 2)
        # 保存文件
        save_file = h5py.File(basedir + '/volume-' + str(i).zfill(2) + ".h5", 'w')
        save_file.create_dataset('image', data=save_ct_array)
        save_file.create_dataset('label', data=save_lbl_array)
        save_file.close()
        sitk.WriteImage(new_ct, basedir + '/volume-' + str(i).zfill(2) + '.nii.gz')
        sitk.WriteImage(new_lbl, basedir + '/label-' + str(i).zfill(2) + '.nii.gz')
    exit(0)
