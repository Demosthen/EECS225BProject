
from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from networks.hyper import HyperDip, HyperFCN, HyperNetwork
from networks.skip import skip
from networks.fcn import fcn
import cv2
import torch
import torch.optim
import glob
from skimage.io import imread
from skimage.io import imsave
import warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from utils.common_utils import *
from SSIM import SSIM
import dataloader
import wandb
from statistics import psnr, ssim

def evaluate_hnet(hyper_dip, hyper_fcn, net, net_kernel, img_size, kernel_size, n_k, iterations):
    output_img = 0
    data_path = "datasets/test_data_loader/"
    save_path = "results/levin/hnet_evaluation/"
    INPUT = 'noise'

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        dtype = torch.cuda.FloatTensor
    else:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        dtype = torch.FloatTensor

    dataloader = dataloader.get_dataloader(
        data_path, batch_size=32, shuffle=False)
    print(f"Evaluating HNet")

    for i, (rgb, gt, rgb_path) in enumerate(dataloader):
        # Get our current batch size since it could be less than opt.batch_size
        batch_size = len(rgb)

        y = gt.type(dtype)
        rgb = rgb.type(dtype)

        img_size = rgb.shape
        # ######################################################################
        padh, padw = kernel_size[0]-1, kernel_size[1]-1
        img_size[0], img_size[1] = img_size[2]+padh, img_size[3]+padw

        '''
        x_net:
        '''
        input_depth = 8

        net_input = get_noise(input_depth, INPUT,
                            (img_size[0], img_size[1])).type(dtype)

        '''
        k_net:
        '''
        net_input_kernel = get_noise(n_k, INPUT, (1, 1)).type(dtype)
        net_input_kernel.squeeze_()

        # Losses
        mse = torch.nn.MSELoss().type(dtype)
        ssim = SSIM().type(dtype)

        # optimizer
        optimizer = torch.optim.Adam([{'params': net.parameters()}, {
            'params': net_kernel.parameters(), 'lr': 1e-4},
            {'params': hyper_dip.parameters()}, {'params': hyper_fcn.parameters()}], lr=LR)
        scheduler = MultiStepLR(optimizer, milestones=[
                                2000, 3000, 4000], gamma=0.5)  # learning rates

        # initilization inputs
        net_input_saved = net_input.detach().clone()
        net_input_kernel_saved = net_input_kernel.detach().clone()

        # get the network output
        dip_weights = hyper_dip(rgb)
        fcn_weights = hyper_fcn(rgb)
        
        psnr_total = 0
        mse_total = 0

        for i, img in enumerate(rgb):
            out_x = net(net_input, weights=dip_weights[i])
            out_k = net_kernel(net_input_kernel, fcn_weights[i]) 
            out_k_m = out_k.view(-1, 1, kernel_size[0], kernel_size[1])
            psnr_total += psnr(out_x, y)
            mse_total += mse(out_x, y)
            if i == output_img:
                path_to_image = rgb_path[i]
                imgname = os.path.basename(path_to_image)
                imgname = os.path.splitext(imgname)[0]

                save_path = os.path.join(save_path, '%s_x.png' % imgname)
                out_x_np = torch_to_np(out_x)
                out_x_np = out_x_np.squeeze()
                out_x_np = out_x_np[padh//2:padh//2 +
                                    img_size[2], padw//2:padw//2+img_size[3]]
                imsave(save_path, out_x_np.astype(np.uint8))

                save_path = os.path.join(save_path, '%s_k.png' % imgname)
                out_k_np = torch_to_np(out_k_m)
                out_k_np = out_k_np.squeeze()
                out_k_np /= np.max(out_k_np)
                imsave(save_path, out_k_np.astype(np.uint8))

                torch.save(net, os.path.join(
                    save_path, "%s_xnet.pth" % imgname))
                torch.save(net_kernel, os.path.join(
                    save_path, "%s_knet.pth" % imgname))
                to_log["img"] = wandb.Image(out_x_np, mode="L")
                to_log["kernel"] = wandb.Image(out_k_np, mode="L")
        psnr_average = psnr_total / len(rgb)
        mse_average = mse_total / len(rgb)

        to_log = {
            "psnr average": psnr_average,
            "mse average": mse_average
        }

        wandb.log(to_log)
