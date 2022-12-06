import numpy as np
import math
from skimage.metrics import structural_similarity as ssim_
from skimage.metrics import mean_squared_error as mse
import cv2 as cv
import os

def psnr(img, gt):
    k = 8
    peak_signal = 2 ** k - 1
    mean_squared_error = mse(img, gt)
    return 10 * math.log10(peak_signal ** 2 / mean_squared_error)

def psnr_color(img, gt):
    img = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
    gt = cv.cvtColor(gt, cv.COLOR_BGR2YCR_CB)
    return psnr(img[0,:,:], gt[0,:,:])

def ssim(img, gt):
    data_range = 255
    return ssim_(img, gt, data_range=data_range)

def ssim_color(img, gt):
    data_range = 255
    return ssim_(img, gt, data_range=data_range, multichannel=True)

# # testing on identical images
# gt = cv.imread(r"datasets\test_data_loader\gt\im1_kernel1_img.png", 0)
# img = cv.imread(r"datasets\test_data_loader\input\im1_kernel1_img.png", 0)
# print("Identical Image Comparison")
# print("PSNR:", psnr(img, gt))
# print("SSIM:", ssim(img, gt))

# # testing on blurry vs gt image
# gt = cv.imread(r"datasets\levin\im1_kernel1_img.png", 0)
# img = cv.imread(r"datasets\levin\gt\im1.png", 0)
# print("Blurry vs. GT Image Comparison")
# print("PSNR:", psnr(img, gt))
# print("SSIM:", ssim(img, gt))

# # comparing different blur kernels
# gt = cv.imread(r"datasets\levin\im1_kernel1_img.png", 0)
# img = cv.imread(r"datasets\levin\im1_kernel2_img.png", 0)
# print("Blur Kernel 1 Image vs. Blur Kernel 2 Image Comparison")
# print("PSNR:", psnr(img, gt))
# print("SSIM:", ssim(img, gt))

def get_folder_statistics(root: str):
    rgb_dir = 'hard_kernels'
    gt_dir = 'gt'
    psnr_total = 0
    ssim_total = 0
    count = 0
    gt_fnames = sorted(os.listdir(os.path.join(root, gt_dir)))
    for rgb_fname in sorted(os.listdir(os.path.join(root, rgb_dir))):
        for gt_fname in gt_fnames:
            if gt_fname[:-4] in rgb_fname and 'x.png' in rgb_fname:
                # if we have a match, calculate statistics
                rgb = cv.imread(os.path.join(root, rgb_dir, rgb_fname), 0)
                gt = cv.imread(os.path.join(root, gt_dir, gt_fname), 0)
                psnr_total += psnr(rgb, gt)
                ssim_total += ssim(rgb, gt)
                count += 1
    psnr_average = psnr_total / count
    ssim_average = ssim_total / count
    print("Average PSNR:", psnr_average)
    print("Average SSIM:", ssim_average)

# testing folder analysis of results vs. ground truth
get_folder_statistics("test_statistics/")