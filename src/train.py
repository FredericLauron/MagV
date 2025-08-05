# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

from utils import train_one_epoch, test_epoch,compress_one_epoch, RateDistortionLoss, CustomDataParallel, configure_optimizers, save_checkpoint, seed_all, TestKodakDataset, generate_mask,save_mask, delete_mask,apply_saved_mask,plot_rate_distorsion
import os
import wandb

from lora import get_lora_model, get_vanilla_finetuned_model


# Vanilla: python train.py --batch-size=16 --checkpoint=../pretrained/stf/stf_0483_best.pth.tar --cuda=1 --dataset=../../../data/small_openimages/ --epochs=15 --lambda=0.013 --learning-rate=0.0001 --lora=1                                       --lora-opt=adam --lora-sched=cosine --model=stf --save=1 --save-dir=../results/adapt_models_vanilla/adapt_0483 --test-dir=../../../data/kodak/ --vanilla-adapt=1
# LORA: python train.py    --batch-size=16 --checkpoint=../pretrained/stf/stf_0483_best.pth.tar --cuda=1 --dataset=../../../data/small_openimages/ --epochs=15 --lambda=0.013 --learning-rate=0.0001 --lora=1 --lora-config=../configs/lora_8_8.yml --lora-opt=adam --lora-sched=cosine --model=stf --save=1 --save-dir=../results/adapt_models_lora/adapt_0483    --test-dir=../../../data/kodak/

def main():
    log_wandb = False
    args = parse_args()
    print(args)


    if args.seed is not None:
        seed_all(args.seed)
        args.save_dir = f'{args.save_dir}_seed_{args.seed}'
    
    # if args.lora:
    #     if args.vanilla_adapt:
    #         conf_name = 'vanilla_adapt'
    #     else:
    #         conf_name = str(args.lora_config).split('/')[-1].replace('.yml','').replace('.yaml','')
    #     init_lr = str(args.learning_rate).replace('0.','0_')
    #     args.save_dir = f'{args.save_dir}_conf_{conf_name}_opt_{args.lora_opt}_sched_{args.lora_sched}_lr_{init_lr}'

    if log_wandb:
        wandb.init(
            project='ALICE',
            name=f'{args.save_dir}',
            config=vars(args)
        )

    if args.save:
        print(f'Results will be saved in: {args.save_dir}')
        os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f'n gpus: {torch.cuda.device_count()}')

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)


    # Kodak test set
    kodak_dataset = TestKodakDataset(data_dir = args.test_dir)

    

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    val_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    kodak_dataloader = DataLoader(
        kodak_dataset, 
        shuffle=False, 
        batch_size=1, 
        pin_memory=(device == "cuda"), 
        num_workers= args.num_workers 
    )
    print(f'Training Dataloader: {len(train_dataloader)}')

    net = models[args.model]()
    net = net.to(device)

    


    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=4)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0

    best_val_loss = float("inf")
    best_kodak_loss = float("inf")

    if args.checkpoint is not None:  
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])

        if args.resume_train:
            print('Loading elements from checkpoint to resume the training')
            last_epoch = checkpoint["epoch"] + 1

            optimizer.load_state_dict(checkpoint["optimizer"])
            if aux_optimizer is not None and checkpoint["aux_optimizer"] is not None:
                aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

            best_val_loss = checkpoint["best_val_loss"]
            best_kodak_loss = checkpoint["best_kodak_loss"]

    # if args.lora:
    #     for param in net.parameters():
    #         param.requires_grad = False

    #     if args.vanilla_adapt:
    #         net = get_vanilla_finetuned_model(net)
    #     else:
    #         net = get_lora_model(net, args.lora_config)
    #     net = net.to(device)


    #     # print('Trainable parameters')
    #     # for name, param in net.named_parameters():
    #     #     if param.requires_grad:
    #     #         print(name)
    #     # print(f"{name}: {param.requires_grad}")
        
    #     total_params  = sum(p.numel() for p in net.parameters() if p.requires_grad)
    #     print(f'Total number of trainable parameters: {total_params}')

    #     # redefine optimizers and scheduler
        
    #     optimizer, aux_optimizer = configure_optimizers(net, args, assert_intersection=False, opt = args.lora_opt)
    #     if args.lora_sched == 'lr_plateau':
    #         print('Using ReduceLROnPlateau scheduler')
    #         lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=4)
    #     elif args.lora_sched == 'cosine':
    #         print('Using Cosine scheduler')
    #         lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


        

    if args.cuda and torch.cuda.device_count() > 1:
        print(f'Training on: {torch.cuda.device_count()} GPUs')
        net = CustomDataParallel(net)
    else:
        print(f'Training on a single GPU')

    #mask and pruning
    if args.mask and args.model=="cheng":
        amounts = [0.6,0.5,0.4,0.3,0.2,0.0]
        lambda_list = [0.0018,0.0035,0.0067,0.0130,0.0250,0.483]
        all_mask, parameters_to_prune = generate_mask(net.g_a, amounts)


    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        # if args.mask and args.model=="cheng":
        #     index = torch.randint(0,6,(1,))
        #     print("index:", index)
        #     mask = all_mask[index]
        #     lambda_value = lambda_list[index]

        #     apply_saved_mask(net.g_a, mask)
        #     criterion.lmbda = lambda_value


        # Training 
        loss_tot_train, bpp_train, mse_train, aux_train_loss = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            args_mask=args.mask,
            all_mask=all_mask if args.mask and args.model=="cheng" else None,
            lambda_list=lambda_list if args.mask and args.model=="cheng" else None,
            parameters_to_prune=parameters_to_prune if args.mask and args.model=="cheng" else None
        )

        if log_wandb:
            wandb.log({
                "train/epoch":epoch,
                "train/loss": loss_tot_train,
                "train/bpp_loss": bpp_train,
                "train/mse_loss": mse_train,
                "train/aux_loss":aux_train_loss
            },step = epoch)      


        # test on validation set
        #TODO test for every rate
        loss_tot_val, bpp_loss_val, mse_loss_val, aux_loss_val, psnr_val, ssim_val = test_epoch(epoch, val_dataloader, net, criterion, tag = 'Val')
        if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(loss_tot_val)
        else:
            lr_scheduler.step()

        if log_wandb:
            wandb.log({
                "val/epoch":epoch,
                "val/loss": loss_tot_val,
                "val/bpp_loss": bpp_loss_val,
                "val/mse_loss": mse_loss_val,
                "val/aux_loss":aux_loss_val,
                "val/psnr":psnr_val,
                "val/mssim":ssim_val,
            },step = epoch)      


        # save checkpoint according to val_loss
        is_best_val = loss_tot_val < best_val_loss
        best_val_loss = min(loss_tot_val, best_val_loss)
        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_kodak_loss":best_kodak_loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else None,
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best_val,
                args.save_dir,
                filename=f"{str(args.lmbda).replace('0.','')}_checkpoint.pth.tar"
            )


        # test on kodak
        # TODO test for every rate
        loss_tot_kodak, bpp_loss_kodak, mse_loss_kodak, aux_loss_kodak, psnr_kodak, ssim_kodak = test_epoch(epoch, kodak_dataloader, net, criterion, tag = 'Kodak')
        if log_wandb:
            wandb.log({
                "kodak/epoch":epoch,
                "kodak/loss": loss_tot_kodak,
                "kodak/bpp_loss": bpp_loss_kodak,
                "kodak/mse_loss": mse_loss_kodak,
                "kodak/aux_loss":aux_loss_kodak,
                "kodak/psnr":psnr_kodak,
                "kodak/mssim":ssim_kodak,
            },step = epoch)   

        # save best kodak model        
        is_best_kodak = loss_tot_kodak < best_kodak_loss
        best_kodak_loss = min(loss_tot_val, best_kodak_loss)
        if(args.save and is_best_kodak):
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_kodak_loss":best_kodak_loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else None,
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                False, out_dir = args.save_dir, filename=f"{str(args.lmbda).replace('0.','')}_checkpoint_best_kodak.pth.tar"
            )


        # try to estimate metrics with the real Arithmetic coding (AC) 
        if epoch%10==0:
            bpp_list = []
            psnr_list = []
            mssim_list = []       

            print("Make actual compression")
            net.update(force = True)

            for index in range(len(amounts)):
                # aplly the mask and lambda value
                apply_saved_mask(net.g_a, all_mask[index])

            bpp_ac, psnr_ac, mssim_ac = compress_one_epoch(net, kodak_dataloader, device)
            bpp_list.append(bpp_ac)
            psnr_list.append(psnr_ac)
            mssim_list.append(mssim_ac)

            if log_wandb:
                wandb.log({
                    f"kodak_compress/bpp_with_ac": bpp_ac,
                    f"kodak_compress/psnr_with_ac": psnr_ac,
                    f"kodak_compress/mssim_with_ac":mssim_ac
                },step = epoch) 

            psnr_res = {}
            mssim_res = {}
            bpp_res = {} 

            bpp_res["ours"] = bpp_list
            psnr_res["ours"] = psnr_list
            mssim_res["ours"] = mssim_list

            plot_rate_distorsion(bpp_res, psnr_res, epoch, eest="compression", metric = 'PSNR',save_fig=False, log_wandb=log_wandb)
            plot_rate_distorsion(bpp_res, mssim_res, epoch, eest="compression_mssim", metric = 'MS-SSIM',save_fig=False, log_wandb=log_wandb, is_psnr=False)
    
    if log_wandb:
        wandb.run.finish()


if __name__ == "__main__":
    main()
