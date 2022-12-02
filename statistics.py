import numpy as np
import math
from skimage.metrics import structural_similarity as ssim_
from skimage.metrics import mean_squared_error as mse
import cv2 as cv

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

# testing on identical images
gt = cv.imread(r"datasets\test_data_loader\gt\im1_kernel1_img.png", 0)
img = cv.imread(r"datasets\test_data_loader\input\im1_kernel1_img.png", 0)
print("Identical Image Comparison")
print("PSNR:", psnr(img, gt))
print("SSIM:", ssim(img, gt))

# testing on blurry vs gt image
gt = cv.imread(r"datasets\levin\im1_kernel1_img.png", 0)
img = cv.imread(r"datasets\levin\gt\im1.png", 0)
print("Blurry vs. GT Image Comparison")
print("PSNR:", psnr(img, gt))
print("SSIM:", ssim(img, gt))

# comparing different blur kernels
gt = cv.imread(r"datasets\levin\im1_kernel1_img.png", 0)
img = cv.imread(r"datasets\levin\im1_kernel2_img.png", 0)
print("Blur Kernel 1 Image vs. Blur Kernel 2 Image Comparison")
print("PSNR:", psnr(img, gt))
print("SSIM:", ssim(img, gt))