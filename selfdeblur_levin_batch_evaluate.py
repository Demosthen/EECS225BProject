
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
from statistics import psnr, psnr_tensor, ssim


def evaluate_hnet(opt, hyper_dip, hyper_fcn, net, net_kernel, n_k, iterations, validation_save_path, run_original=False, ignore_kernel=True):
    input_depth = 8
    validation_data_path = "datasets/test_data_loader/"
    os.makedirs(validation_save_path, exist_ok=True)
    INPUT = 'noise'
    reg_noise_std = 0.001
    imgs_to_track = [0, 9, 18, 27]

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        dtype = torch.cuda.FloatTensor
    else:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        dtype = torch.FloatTensor

    # run vanilla ver
    if run_original:
        net = HyperDip(input_depth, 1,
                       num_channels_down=[128, 128, 128, 128, 128],
                       num_channels_up=[128, 128, 128, 128, 128],
                       num_channels_skip=[16, 16, 16, 16, 16],
                       upsample_mode='bilinear',
                       need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
        net = net.type(dtype)
        net.train()

    if run_original or ignore_kernel:
        n_k = 200
        net_kernel = HyperFCN(n_k, opt.kernel_size[0]*opt.kernel_size[1])
        net_kernel = net_kernel.type(dtype)
        net_kernel.train()
    # end vanilla ver

    loader_batch_size = 32

    padh, padw = opt.kernel_size[0]-1, opt.kernel_size[1]-1
    dataloader = get_dataloader(
        validation_data_path, batch_size=loader_batch_size, shuffle=False, padh=padh, padw=padw)
    print(f"Evaluating HNet")

    iterator = iter(dataloader)
    for i, (rgb, gt, rgb_path, net_input, net_input_kernel) in enumerate(iterator):
        # Get our current batch size since it could be less than opt.batch_size
        batch_size = len(rgb)

        y = gt.type(dtype)
        y.requires_grad = False
        rgb = rgb.type(dtype)
        rgb.requires_grad = False

        img_size = rgb.shape
        # ######################################################################
        
        opt.img_size[0], opt.img_size[1] = img_size[2]+padh, img_size[3]+padw

        '''
        x_net:
        '''
        input_depth = 8

        # net_input = get_noise(input_depth, INPUT,
        #                     (opt.img_size[0], opt.img_size[1])).type(dtype)

        net_input = net_input.type(dtype)
        net_input.requires_grad = False
        '''
        k_net:
        '''
        # net_input_kernel = get_noise(n_k, INPUT, (1, 1)).type(dtype)
        # net_input_kernel.squeeze_()
        net_input_kernel = net_input_kernel.type(dtype)
        net_input_kernel.requires_grad = False
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
        net_input_saved.requires_grad = False
        net_input_kernel_saved.requires_grad = False

        # get the network output

        # don't run hypernetwork if running vanilla version
        if run_original == False:
            dip_weights = hyper_dip(rgb)
            if not ignore_kernel:
                fcn_weights = hyper_fcn(rgb)

            if loader_batch_size == 1:
                dip_weights = [dip_weights]
                if not ignore_kernel:
                    fcn_weights = [fcn_weights]

        # initialize evaluation parameters
        psnr_total = 0
        ssim_total = 0
        mse_total = 0

        # initialize logging dict
        to_log = {}

        for j, img in enumerate(rgb):
            all_psnr = np.zeros(iterations)
            all_ssim = np.zeros(iterations)
            all_mse = np.zeros(iterations)

            # train SelfDeblur
            for step in tqdm(range(iterations)):

                # input regularization
                net_input = net_input_saved + reg_noise_std * \
                    torch.zeros(net_input_saved.shape).type_as(
                        net_input_saved.data).normal_()
                net_input.requires_grad = False
                # net_input_kernel = net_input_kernel_saved + reg_noise_std*torch.zeros(net_input_kernel_saved.shape).type_as(net_input_kernel_saved.data).normal_()

                optimizer.zero_grad()

                # get the network output
                if step == 0 and run_original == False:
                    out_x = net(net_input[j], weights=[
                                nn.Parameter(w) for w in dip_weights[j]])
                    if ignore_kernel:
                        out_k = net_kernel(net_input_kernel[j])
                    else:
                        out_k = net_kernel(net_input_kernel[j], weights=[
                                    nn.Parameter(w) for w in fcn_weights[j]])
                else:
                    out_x = net(net_input[j])
                    out_k = net_kernel(net_input_kernel[j])

                out_k_m = out_k.view(-1, 1,
                                     opt.kernel_size[0], opt.kernel_size[1])
                # print(out_k_m)
                out_y = nn.functional.conv2d(
                    out_x, out_k_m, padding=0, bias=None)

                ref_grayscale = torch.mean(rgb[j], dim=0)[None, None, :, :]

                curr_mse = mse(out_y, ref_grayscale)
                curr_ssim = ssim_tensor(out_y, ref_grayscale)
                curr_psnr = psnr_tensor(out_y, ref_grayscale)

                if step < 1000:
                    total_loss = curr_mse
                else:
                    total_loss = 1-curr_ssim

                total_loss.backward()
                optimizer.step()

                # adjust the learning rate based on scheduler
                scheduler.step()

                # logging
                all_mse[step] += curr_mse
                all_ssim[step] += curr_ssim
                all_psnr[step] += curr_psnr

                checkpoint_freq = 100
                if (step % checkpoint_freq) == 0:
                    # save image and kernel to disk
                    path_to_image = rgb_path[j]
                    imgname = os.path.basename(path_to_image)
                    imgname = os.path.splitext(imgname)[0]

                    curr_img_path = os.path.join(
                        validation_save_path, imgname + f'_step_{step}.png')
                    out_x_np = torch_to_np(out_x)
                    out_x_np = out_x_np.squeeze()
                    out_x_np = out_x_np[padh//2:padh//2 +
                                        img_size[2], padw//2:padw//2+img_size[3]]
                    imsave(curr_img_path, out_x_np.astype(np.uint8))

                    curr_kernel_path = os.path.join(
                        validation_save_path, f'kernel_from_' + imgname + f'_step_{step}.png')
                    out_k_np = torch_to_np(out_k_m)
                    out_k_np = out_k_np.squeeze()
                    out_k_np /= np.max(out_k_np)
                    imsave(curr_kernel_path, out_k_np.astype(np.uint8))

                    # log specified images to wandb
                    if j in imgs_to_track:
                        to_log[imgname +
                               f'_step_{step}.png'] = wandb.Image(out_x_np, mode='L')
                        to_log[f'kernel_from_' + imgname +
                               f'_step_{step}.png'] = wandb.Image(out_k_np, mode='L')

            # evaluate trained selfdeblur
            out_x = net(net_input[j])
            out_k = net_kernel(net_input_kernel[j]) 
            out_k_m = out_k.view(-1, 1, opt.kernel_size[0], opt.kernel_size[1])

            out_x = out_x[..., padh//2:padh//2 +
                          img_size[2], padw//2:padw//2+img_size[3]]
            out_x_np = torch_to_np(out_x).squeeze()

            out_k_np = torch_to_np(out_k_m).squeeze()
            out_k_np /= np.max(out_k_np)

            out_y = y[j]
            out_y_np = torch_to_np(out_y)
            psnr_total += psnr(out_x_np, out_y_np)
            ssim_total += ssim(out_x_np, out_y_np)
            mse_total += mse(out_x, out_y)
            path_to_image = rgb_path[j]
            imgname = os.path.basename(path_to_image)
            imgname = os.path.splitext(imgname)[0]

            curr_img_path = os.path.join(
                validation_save_path, imgname + '_final.png')

            imsave(curr_img_path, out_x_np.astype(np.uint8))

            curr_kernel_path = os.path.join(
                validation_save_path, 'kernel_from_' + imgname + '_final.png')
            imsave(curr_kernel_path, out_k_np.astype(np.uint8))

            if j in imgs_to_track:
                to_log[imgname + '_final'] = wandb.Image(out_x_np, mode="L")
                to_log['kernel_from_' + imgname +
                       '_final.png'] = wandb.Image(out_k_np, mode="L")


        all_mse /= len(rgb)
        np.putmask(all_mse, all_mse > 10^10, 1)
        all_psnr /= len(rgb)
        np.putmask(all_psnr, all_psnr > 10^10, 1)
        all_ssim /= len(rgb)
        np.putmask(all_ssim, all_ssim > 10^10, 1)

        final_psnr_average = psnr_total / len(rgb)
        final_ssim_average = ssim_total / len(rgb)
        final_mse_average = mse_total / len(rgb)

        to_log['final_psnr_average'] = final_psnr_average
        to_log['final_ssim_average'] = final_ssim_average
        to_log['final_mse_average'] = final_mse_average

        for k in range(0, iterations, 1000):
            to_log[f'psnr_average_{i}'] = all_psnr[k]
            to_log[f'ssim_average_{i}'] = all_ssim[k]
            to_log[f'mse_average_{i}'] = all_mse[k]

        plt.figure()
        plt.yscale('log')
        plt.plot(all_mse)
        plt.title('Average MSE over all images vs. training epoch')
        plt.xlabel('Training step')
        plt.ylabel('MSE loss')
        plt.savefig(os.path.join(validation_save_path, 'avg_mse_finetune.png'))
        plt.figure()
        plt.yscale('log')
        plt.plot(all_psnr)
        plt.title('Average PSNR over all images vs. training epoch')
        plt.xlabel('Training step')
        plt.ylabel('MSE PSNR')
        plt.savefig(os.path.join(
            validation_save_path, 'avg_psnr_finetune.png'))
        plt.figure()
        plt.plot(all_ssim)
        plt.title('Average SSIM over all images vs. training epoch')
        plt.xlabel('Training step')
        plt.ylabel('SSIM loss')
        plt.savefig(os.path.join(
            validation_save_path, 'avg_ssim_finetune.png'))

        to_log['avg_mse_plot'] = wandb.Image(os.path.join(
            validation_save_path, 'avg_mse_finetune.png'))
        to_log['avg_psnr_plot'] = wandb.Image(os.path.join(
            validation_save_path, 'avg_psnr_finetune.png'))
        to_log['avg_ssim_plot'] = wandb.Image(os.path.join(
            validation_save_path, 'avg_ssim_finetune.png'))

        # return statistics
        return to_log

INPUT = 'noise'
pad = 'reflection'
LR = 0.01
KERNEL_LR = 0.01

# parser = argparse.ArgumentParser()
# parser.add_argument('--num_epochs', type=int, default=50,
#                     help='number of epochs of training')
# parser.add_argument('--num_iter', type=int, default=50,
#                     help='number of iterations per image')
# parser.add_argument('--img_size', type=int,
#                     default=[256, 256], help='size of each image dimension')
# parser.add_argument('--kernel_size', type=int,
#                     default=[27, 27], help='size of blur kernel [height, width]')
# parser.add_argument('--data_path', type=str,
#                     default="datasets/test_data_loader/", help='path to blurry images')
# parser.add_argument('--batch_size', type=int,
#                     default=16, help='number of images in batch')
# parser.add_argument('--save_path', type=str,
#                     default="results/levin/hnet_models/", help='path to save results')
# parser.add_argument('--save_frequency', type=int,
#                     default=10, help='lfrequency to save results')
# parser.add_argument('--l1_coeff', type=float,
#                     default=0, help="coefficient on L1 norm of kernel in loss function")
# opt = parser.parse_args()

# if isinstance(opt.kernel_size, int):
#     opt.kernel_size = [opt.kernel_size, opt.kernel_size]
# # testing evaluate_hnet
# if torch.cuda.is_available():
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = True
#     dtype = torch.cuda.FloatTensor
# else:
#     torch.backends.cudnn.enabled = False
#     torch.backends.cudnn.benchmark = False
#     dtype = torch.FloatTensor
# num_iter = 5000
# reg_noise_std = 0.001
# input_depth = 8


# net = HyperDip(input_depth, 1,
#                num_channels_down=[128, 128, 128, 128, 128],
#                num_channels_up=[128, 128, 128, 128, 128],
#                num_channels_skip=[16, 16, 16, 16, 16],
#                upsample_mode='bilinear',
#                need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
# net = net.type(dtype)

# n_k = 200
# net_kernel = HyperFCN(n_k, opt.kernel_size[0]*opt.kernel_size[1])
# net_kernel = net_kernel.type(dtype)

# hyper_dip = HyperNetwork(net, dtype=dtype)
# hyper_dip = hyper_dip.type(dtype)

# hyper_fcn = HyperNetwork(net_kernel, dtype=dtype)
# hyper_fcn = hyper_fcn.type(dtype)

# to_log = evaluate_hnet(opt, hyper_dip, hyper_fcn, net, net_kernel, n_k, 1, "results/levin/hnet_evaluation/test/")
# run = wandb.init(project="EECS225BProject", entity="cs182rlproject")
# wandb.log(to_log)