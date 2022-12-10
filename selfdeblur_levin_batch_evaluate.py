
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
from dataloader import get_dataloader
import wandb
from statistics import psnr, ssim

def evaluate_hnet(opt, hyper_dip, hyper_fcn, net, net_kernel, n_k, iterations):
    output_img = 0
    validation_data_path = "datasets/test_data_loader/"
    validation_save_path = "results/levin/hnet_evaluation/"
    INPUT = 'noise'
    reg_noise_std = 0.001

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        dtype = torch.cuda.FloatTensor
    else:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        dtype = torch.FloatTensor

    dataloader = get_dataloader(
        validation_data_path, batch_size=32, shuffle=False)
    print(f"Evaluating HNet")

    iterator = iter(dataloader)
    for i, (rgb, gt, rgb_path) in enumerate(iterator):
        # Get our current batch size since it could be less than opt.batch_size
        batch_size = len(rgb)

        y = gt.type(dtype)
        rgb = rgb.type(dtype)

        img_size = rgb.shape
        # ######################################################################
        padh, padw = opt.kernel_size[0]-1, opt.kernel_size[1]-1
        opt.img_size[0], opt.img_size[1] = img_size[2]+padh, img_size[3]+padw

        '''
        x_net:
        '''
        input_depth = 8

        net_input = get_noise(input_depth, INPUT,
                            (opt.img_size[0], opt.img_size[1])).type(dtype)

        '''
        k_net:
        '''
        net_input_kernel = get_noise(n_k, INPUT, (1, 1)).type(dtype)
        net_input_kernel.squeeze_()

        # Losses
        mse = torch.nn.MSELoss().type(dtype)
        ssim_tensor = SSIM().type(dtype)

        # optimizer
        optimizer = torch.optim.Adam([{'params': net.parameters()}, {
                                    'params': net_kernel.parameters(), 'lr': 1e-4}], lr=LR)
        scheduler = MultiStepLR(optimizer, milestones=[
                                2000, 3000, 4000], gamma=0.5)  # learning rates

        # initilization inputs
        net_input_saved = net_input.detach().clone()
        net_input_kernel_saved = net_input_kernel.detach().clone()

        # get the network output
        dip_weights = hyper_dip(rgb)
        fcn_weights = hyper_fcn(rgb)

        # initialize evaluation parameters
        psnr_total = 0
        ssim_total = 0

        # TODO: how frequently do we want to log to wandb?
        to_log = {}

        for j, img in enumerate(rgb):
            ### train SelfDeblur
            for step in tqdm(range(iterations)):

                # input regularization
                net_input = net_input_saved + reg_noise_std * \
                    torch.zeros(net_input_saved.shape).type_as(
                        net_input_saved.data).normal_()

                optimizer.zero_grad()

                # get the network output
                if step == 0:
                    out_x = net(net_input, weights=dip_weights[j])
                    out_k = net_kernel(net_input_kernel, weights=fcn_weights[j])
                else:
                    out_x = net(net_input)
                    out_k = net_kernel(net_input_kernel)

                out_k_m = out_k.view(-1, 1, opt.kernel_size[0], opt.kernel_size[1])
                # print(out_k_m)
                out_y = nn.functional.conv2d(out_x, out_k_m, padding=0, bias=None)

                ref_grayscale = torch.mean(rgb[j], dim=0)[None, None, :, :]

                if step < 1000:
                    total_loss = mse(out_y, ref_grayscale)
                else:
                    total_loss = 1-ssim_tensor(out_y, ref_grayscale)
                # total_loss = 1-ssim_tensor(out_y, ref_grayscale)

                total_loss.backward(retain_graph=True)
                optimizer.step()
        
                # change the learning rate
                scheduler.step()

            # evaluate trained selfdeblur
            out_x = net(net_input)
            out_k = net_kernel(net_input_kernel) 
            out_k_m = out_k.view(-1, 1, opt.kernel_size[0], opt.kernel_size[1])
            out_x_np = torch_to_np(out_x).squeeze()
            out_x_np = out_x_np[padh//2:padh//2 +
                                img_size[2], padw//2:padw//2+img_size[3]]
            out_y_np = torch_to_np(y[j])
            print(out_x_np.shape)
            print(out_y_np.shape)
            psnr_total += psnr(out_x_np, out_y_np)
            ssim_total += ssim(out_x_np, out_y_np)
            if j == output_img:
                path_to_image = rgb_path[j]
                imgname = os.path.basename(path_to_image)
                imgname = os.path.splitext(imgname)[0]

                curr_img_path = os.path.join(validation_save_path, '%s_x.png' % imgname)
                # out_x_np = torch_to_np(out_x)
                # out_x_np = out_x_np.squeeze()
                # out_x_np = out_x_np[padh//2:padh//2 +
                #                     img_size[2], padw//2:padw//2+img_size[3]]
                imsave(curr_img_path, out_x_np.astype(np.uint8))

                curr_kernel_path = os.path.join(validation_save_path, '%s_k.png' % imgname)
                out_k_np = torch_to_np(out_k_m)
                out_k_np = out_k_np.squeeze()
                out_k_np /= np.max(out_k_np)
                imsave(curr_kernel_path, out_k_np.astype(np.uint8))

                torch.save(net, os.path.join(
                    validation_save_path, "%s_xnet.pth" % imgname))
                torch.save(net_kernel, os.path.join(
                    validation_save_path, "%s_knet.pth" % imgname))
                to_log["img"] = wandb.Image(out_x_np, mode="L")
                to_log["kernel"] = wandb.Image(out_k_np, mode="L")
        psnr_average = psnr_total / len(rgb)
        ssim_average = ssim_total / len(rgb)

        to_log = {
            "psnr average": psnr_average,
            "mse average": ssim_average
        }

        wandb.log(to_log)
        #return statistics here




parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=50,
                    help='number of epochs of training')
parser.add_argument('--num_iter', type=int, default=50,
                    help='number of iterations per image')
parser.add_argument('--img_size', type=int,
                    default=[256, 256], help='size of each image dimension')
parser.add_argument('--kernel_size', type=int,
                    default=[27, 27], help='size of blur kernel [height, width]')
parser.add_argument('--data_path', type=str,
                    default="datasets/test_data_loader/", help='path to blurry images')
parser.add_argument('--batch_size', type=int,
                    default=16, help='number of images in batch')
parser.add_argument('--save_path', type=str,
                    default="results/levin/hnet_models/", help='path to save results')
parser.add_argument('--save_frequency', type=int,
                    default=10, help='lfrequency to save results')
parser.add_argument('--l1_coeff', type=float,
                    default=0, help="coefficient on L1 norm of kernel in loss function")
opt = parser.parse_args()

if isinstance(opt.kernel_size, int):
    opt.kernel_size = [opt.kernel_size, opt.kernel_size]

# testing evaluate_hnet
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
else:
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    dtype = torch.FloatTensor

INPUT = 'noise'
pad = 'reflection'
LR = 0.01
KERNEL_LR= 0.01
num_iter = 5000
reg_noise_std = 0.001
input_depth = 8


net = HyperDip(input_depth, 1,
               num_channels_down=[128, 128, 128, 128, 128],
               num_channels_up=[128, 128, 128, 128, 128],
               num_channels_skip=[16, 16, 16, 16, 16],
               upsample_mode='bilinear',
               need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
net = net.type(dtype)

n_k = 200
net_kernel = HyperFCN(n_k, opt.kernel_size[0]*opt.kernel_size[1])
net_kernel = net_kernel.type(dtype)

hyper_dip = HyperNetwork(net)
hyper_dip = hyper_dip.type(dtype)

hyper_fcn = HyperNetwork(net_kernel)
hyper_fcn = hyper_fcn.type(dtype)

evaluate_hnet(opt, hyper_dip, hyper_fcn, net, net_kernel, n_k, 2)