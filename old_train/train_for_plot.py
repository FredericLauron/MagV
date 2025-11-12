import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

# from comp.datasets import ImageFolder
# from compressai.datasets import ImageFolder
# from comp.datasets import ImageFolder
from custom_comp.datasets import ImageFolder
# from comp.zoo import models
from custom_comp.zoo import models

from opt import parse_args

from utils import train_one_epoch, test_epoch,compress_one_epoch, RateDistortionLoss, CustomDataParallel, configure_optimizers, save_checkpoint, seed_all, TestKodakDataset, generate_mask_from_unstructured,save_mask, delete_mask,apply_saved_mask
from evaluate import plot_rate_distorsion
import os
import wandb

from lora import get_lora_model, get_vanilla_finetuned_model




def main():

    # #Device definition
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'n gpus: {torch.cuda.device_count()}')


    # #Model instanciation
    net = models["cheng"]()
    net = net.to(device)

    # #Model state dict loading
    # print("Loading")
    # checkpoint = torch.load("/home/ids/flauron-23/MagV/results/mask/adapt_0483_seed_42/013_checkpoint_best_kodak.pth.tar", map_location=device)
    # net.load_state_dict(checkpoint["state_dict"])

    # bpp_list = []
    # psnr_list = []
    # mssim_list = []       

    # print("Make actual compression")
    # net.update(force = True)

    # kodak_dataloader = DataLoader(
    # "/home/ids/flauron-23/kodak", 
    # shuffle=False, 
    # batch_size=1, 
    # pin_memory=(device == "cuda"), 
    # num_workers= 8 
    # )


    # for index in range(6):
    #     # aplly the mask 
    #     apply_saved_mask(net.g_a, all_mask[index])

    #     bpp_ac, psnr_ac, mssim_ac = compress_one_epoch(net, kodak_dataloader, device)
    #     bpp_list.append(bpp_ac)
    #     psnr_list.append(psnr_ac)
    #     mssim_list.append(mssim_ac)

    # # if log_wandb:
    # #     wandb.log({
    # #         f"kodak_compress/bpp_with_ac": bpp_ac,
    # #         f"kodak_compress/psnr_with_ac": psnr_ac,
    # #         f"kodak_compress/mssim_with_ac":mssim_ac
    # #     },step = epoch) 

    # psnr_res = {}
    # mssim_res = {}
    # bpp_res = {} 

    # bpp_res["ours"] = bpp_list
    # psnr_res["ours"] = psnr_list
    # mssim_res["ours"] = mssim_list

    # plot_rate_distorsion(bpp_res, psnr_res, 10, eest="compression", metric = 'PSNR',save_fig=True, log_wandb=False)
    # plot_rate_distorsion(bpp_res, mssim_res, 10, eest="compression_mssim", metric = 'MS-SSIM',save_fig=True, log_wandb=False, is_psnr=False)

    all_mask={}
    parameters_to_prune={}
    amounts = [0.6,0.5,0.4,0.3,0.2,0.0] #[0.7, ...,0.0]
    #lambda_list = [0.0018,0.0035,0.0067,0.0130,0.0250,0.483]
    all_mask["g_a"], parameters_to_prune["g_a"] = generate_mask_from_unstructured(net.g_a, amounts)
    all_mask["g_s"], parameters_to_prune["g_s"] = generate_mask_from_unstructured(net.g_s, amounts)
    print("succes")


main()