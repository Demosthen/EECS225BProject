
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
from selfdeblur_levin_batch_evaluate import evaluate_hnet

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
parser.add_argument('--l6_coeff', type=float,
                    default=0, help="coefficient on L6 norm of kernel in loss function")
parser.add_argument('--l1_coeff', type=float,
                    default=0, help="coefficient on L1 norm of kernel in loss function")
parser.add_argument('--lr', type=float,
                    default=0.01, help="coefficient on L1 norm of kernel in loss function")
parser.add_argument('--kernel_lr', type=float,
                    default=0.01, help="coefficient on L1 norm of kernel in loss function")
parser.add_argument('--eval_freq', type=int,
                    default=2, help="How many epochs to train between evaluations")
parser.add_argument('--num_steps_per_epoch', type=int,
                    default=None, help="How many batches to call an epoch. default is going through all the data once")
opt = parser.parse_args()
# print(opt)

if isinstance(opt.kernel_size, int):
    opt.kernel_size = [opt.kernel_size, opt.kernel_size]

run = wandb.init(project="EECS225BProject", entity="cs182rlproject")
wandb.config.update(opt)

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
else:
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    dtype = torch.FloatTensor

warnings.filterwarnings("ignore")

files_source = glob.glob(os.path.join(opt.data_path, '*.png'))
files_source.sort()
save_path = opt.save_path
os.makedirs(save_path, exist_ok=True)

INPUT = 'noise'
pad = 'reflection'
LR = opt.lr
KERNEL_LR = opt.kernel_lr
num_iter = opt.num_iter
reg_noise_std = 0.001

input_depth = 8

net = HyperDip(input_depth, 1,
               num_channels_down=[128, 128, 128, 128, 128],
               num_channels_up=[128, 128, 128, 128, 128],
               num_channels_skip=[16, 16, 16, 16, 16],
               upsample_mode='bilinear',
               need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
net = net.type(dtype)
net.train()

n_k = 200
net_kernel = HyperFCN(n_k, opt.kernel_size[0]*opt.kernel_size[1])
net_kernel = net_kernel.type(dtype)
net_kernel.train()

hyper_dip = HyperNetwork(net, dtype=dtype)
hyper_dip = hyper_dip.type(dtype)
hyper_dip.train()

hyper_fcn = HyperNetwork(net_kernel, dtype=dtype)
hyper_fcn = hyper_fcn.type(dtype)
hyper_fcn.train()

wandb.watch((hyper_fcn, hyper_dip, net, net_kernel), log_freq=20, log="all")

pre_softmax_kernel_activation = None

# Register a hook right before the softmax activation so that
# we can regularize with L1


def hook(module, input, output):
    global pre_softmax_kernel_activation
    pre_softmax_kernel_activation = input[0]


net_kernel.model[-1].register_forward_hook(hook)

# Losses
mse = torch.nn.MSELoss().type(dtype)
ssim = SSIM().type(dtype)

# optimizer
optimizer = torch.optim.Adam([
    {'params': hyper_dip.hnet.internal_params}, {'params': hyper_fcn.hnet.internal_params, "lr":KERNEL_LR}], lr=LR)
scheduler = MultiStepLR(optimizer, milestones=[
                        opt.num_epochs // 5, opt.num_epochs // 4, opt.num_epochs // 2], gamma=0.5)  # learning rates


dataloader = get_dataloader(
    opt.data_path, batch_size=opt.batch_size, shuffle=True, use_gopro_data=("gopro" in opt.data_path))
for epoch in range(opt.num_epochs):
    iterator = iter(dataloader)
    for i, (rgb, gt, rgb_path) in enumerate(iterator):
        if opt.num_steps_per_epoch is not None and i >= opt.num_steps_per_epoch:
            break
        print(f"Processing Epoch:{epoch} Batch: {i+1}")
        # Get our current batch size since it could be less than opt.batch_size
        batch_size = len(rgb)

        y = rgb.type(dtype)
        rgb = rgb.type(dtype)
        rgb.requires_grad = False
        y.requires_grad = False

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
        net_input.requires_grad = False

        '''
        k_net:
        '''
        net_input_kernel = get_noise(n_k, INPUT, (1, 1)).type(dtype)
        net_input_kernel.squeeze_()
        net_input_kernel.requires_grad=False

        # initilization inputs
        net_input_saved = net_input.detach().clone()
        net_input_kernel_saved = net_input_kernel.detach().clone()

        # change the learning rate
        scheduler.step(epoch)

        # start SelfDeblur
        for substep in tqdm(range(num_iter)):

            # input regularization
            net_input = net_input_saved + reg_noise_std * \
                torch.zeros(net_input_saved.shape).type_as(
                    net_input_saved.data).normal_()

            optimizer.zero_grad()

            # get the network output
            dip_weights = hyper_dip(rgb)
            fcn_weights = hyper_fcn(rgb)
            # out_x = net(net_input, weights=dip_weights)
            # out_k = net_kernel(net_input_kernel, weights=fcn_weights)
            out_x = []
            out_k_m = []
            out_y = []
            kernel_l1 = []
            kernel_l6 = []

            for j, img in enumerate(rgb):
                out_x.append(net(net_input, weights=dip_weights[j]))
                out_k = net_kernel(net_input_kernel, fcn_weights[j])
                out_k_m.append(
                    out_k.view(-1, 1, opt.kernel_size[0], opt.kernel_size[1]))
                kernel_l1.append(torch.norm(
                    pre_softmax_kernel_activation.view(-1, opt.kernel_size[0] * opt.kernel_size[1]), 1, -1))
                kernel_l6.append(torch.norm(
                    pre_softmax_kernel_activation.view(-1, opt.kernel_size[0] * opt.kernel_size[1]), 6, -1))
                out_y.append(
                    F.conv2d(out_x[-1], out_k_m[-1], padding=0, bias=None))
            out_x = torch.stack(out_x)
            out_k_m = torch.stack(out_k_m)
            out_y = torch.stack(out_y)
            kernel_l1 = torch.stack(kernel_l1)
            kernel_l1_loss = kernel_l1.mean()
            kernel_l6 = torch.stack(kernel_l6)
            kernel_l6_loss = kernel_l6.mean()

            if epoch < (opt.num_epochs // 5):
                acc_loss = mse(out_y, y)
                is_SSIM = 0
            else:
                acc_loss = 1-ssim(out_y.squeeze(1), y)
                is_SSIM = 1

            total_loss = acc_loss + kernel_l1_loss * opt.l1_coeff + kernel_l6_loss * opt.l6_coeff

            total_loss.backward()
            optimizer.step()

            to_log = {
                "total_loss": total_loss.detach(),
                "Kernel_L1_loss": kernel_l1_loss.detach(),
                "Kernel_L6_loss": kernel_l6_loss.detach(),
                "Accuracy_loss": acc_loss.detach(),
                "Epoch": epoch,
                "Learning rate 0": scheduler.get_lr()[0],
                "Learning rate 1": scheduler.get_lr()[1],
                "Using_SSIM": is_SSIM
            }

            # print the loss
            if i % 10 == 0:
                print("{}: {}".format(substep, kernel_l1_loss.detach().item()))

            if (i+1) % opt.save_frequency == 0:
                #print('Iteration %05d' %(step+1))
                out_x_nps = out_x.detach().cpu().numpy()
                out_x_nps = out_x_nps.squeeze()
                out_x_nps = out_x_nps[:, padh//2:padh//2 +
                                    img_size[2], padw//2:padw//2+img_size[3]]

                out_k_nps = out_k_m.detach().cpu().numpy()
                out_k_nps = out_k_nps.squeeze()
                out_k_nps /= np.max(out_k_nps)

                out_y_nps = out_y.detach().cpu().numpy()
                out_y_nps = out_y_nps.squeeze()
                out_y_nps = out_y_nps[:, padh//2:padh//2 +
                                    img_size[2], padw//2:padw//2+img_size[3]]

                for n in range(batch_size):
                    path_to_image = rgb_path[n]
                    imgname = os.path.basename(path_to_image)
                    imgname = os.path.splitext(imgname)[0]

                    save_path = os.path.join(
                        opt.save_path, '%s_x.png' % imgname)
                    out_x_np = out_x_nps[n]
                    imsave(save_path, out_x_np.astype(np.uint8))

                    save_path = os.path.join(
                        opt.save_path, '%s_k.png' % imgname)

                    out_k_np = out_k_nps[n]
                    imsave(save_path, out_k_np.astype(np.uint8))

                    out_y_np = out_y_nps[n]

                    # torch.save(net, os.path.join(
                    #     opt.save_path, "%s_xnet.pth" % imgname))
                    # torch.save(net_kernel, os.path.join(
                    #     opt.save_path, "%s_knet.pth" % imgname))
                    
                to_log["prior"] = wandb.Image(out_x_np, mode="L")
                to_log["kernel"] = wandb.Image(out_k_np, mode="L")
                to_log["img"] = wandb.Image(out_y_np, mode="L")
                to_log["gt"] = wandb.Image(gt[-1], mode="L")
            wandb.log(to_log)
    if epoch % opt.eval_freq == 0:
        to_log = evaluate_hnet(opt, hyper_dip, hyper_fcn, net, net_kernel, n_k, 1000, "results/levin/hnet_evaluation/")
        wandb.log(to_log)
