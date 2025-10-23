import torch
from utils.functions import compute_psnr, compute_metrics, compute_msssim
from compressai.ops import compute_padding
from utils.masks import delete_mask,apply_saved_mask
from utils.chengBA2 import set_cheng2020Attention_index

from compressai.models.waseda import Cheng2020Attention
from custom_comp.models import Cheng2020Attention_BA2

import torch.nn.functional as F

from torch.profiler import profile, record_function, ProfilerActivity
import sys
import numpy as np

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(
        model, 
        criterion, 
        train_dataloader, 
        optimizer, 
        aux_optimizer, 
        epoch, 
        clip_max_norm,
        args_mask=None,
        all_mask=None,
        lambda_list=None,
        parameters_to_prune=None,
        probs=None):
    
    model.train()
    device = next(model.parameters()).device


    loss_tot_metric = AverageMeter()
    bpp_loss_metric = AverageMeter()
    mse_loss_metric = AverageMeter()
    aux_loss_metric = AverageMeter()

    for i, d in enumerate(train_dataloader):
        
        d = d.to(device)

        # Mask selection. 
        # A mask is ramdomly selected (uniform distribution) from the list of masks.
        # The associated lambda is then selected from the lambda_list.     
        if args_mask is not None and isinstance(model, Cheng2020Attention) and lambda_list is not None:
            
            index = np.random.choice(np.arange(len(lambda_list)), p=probs)
            #    index = torch.randint(0,6,(1,))

            lambda_value = lambda_list[index]
            #print("index:", index, "lambda_value:", lambda_value)

            apply_saved_mask(model.g_a, all_mask["g_a"][index])
            apply_saved_mask(model.g_s, all_mask["g_s"][index])

            criterion.lmbda = lambda_value

        #Adapter
        elif isinstance(model,Cheng2020Attention) and args_mask is None:
            # Index selection. 
            # An index is ramdomly selected (uniform distribution).
            # The associated lambda is then selected from the lambda_list.     
                
            # Selection of the index    
            index = np.random.choice(np.arange(6), p=probs)

            # Selection of the relacted lambda value
            lambda_value = lambda_list[index]

            # Update the index in all the adapters
            #model.set_index(index)
            set_cheng2020Attention_index(model,index)

            # Update the lambda in the optimizer
            criterion.lmbda = lambda_value

            # print("Index has been set")
            # print(f"Adapter - index: {index}, lambda: {lambda_value}")

        optimizer.zero_grad()
        if aux_optimizer is not None:
            aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        
        
        optimizer.step()

        aux_loss = model.aux_loss()
        if aux_optimizer is not None:
            
            aux_loss.backward()
            aux_optimizer.step()

        if i % 5 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item() * 255 ** 2 / 3:.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}" 
            )

        loss_tot_metric.update(out_criterion["loss"].clone().detach())
        bpp_loss_metric.update(out_criterion["bpp_loss"].clone().detach())
        mse_loss_metric.update(out_criterion["mse_loss"].clone().detach())
        aux_loss_metric.update(aux_loss.clone().detach())

        if args_mask and isinstance(model, Cheng2020Attention):
            delete_mask(model.g_a, parameters_to_prune["g_a"])
            delete_mask(model.g_s, parameters_to_prune["g_s"])

    return loss_tot_metric.avg, bpp_loss_metric.avg, mse_loss_metric.avg, aux_loss_metric.avg


def test_epoch(epoch, test_dataloader, model, criterion, tag = 'Val'):
    model.eval()
    device = next(model.parameters()).device

    loss_tot_metric = AverageMeter()
    bpp_loss_metric = AverageMeter()
    mse_loss_metric = AverageMeter()
    aux_loss_metric = AverageMeter()

    psnr_metric = AverageMeter()
    ssim_metric = AverageMeter()

    with torch.no_grad():
        for i,d in enumerate(test_dataloader):

            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            psnr_metric.update(compute_psnr(d, out_net["x_hat"]))
            ssim_metric.update(compute_msssim(d, out_net["x_hat"]))

            loss_tot_metric.update(out_criterion["loss"])
            bpp_loss_metric.update(out_criterion["bpp_loss"])
            mse_loss_metric.update(out_criterion["mse_loss"])
            aux_loss_metric.update(model.aux_loss())

    print(
        f"{tag} epoch {epoch}: Average losses:"
        f"\tLoss: {loss_tot_metric.avg:.3f} |"
        f"\tMSE loss: {mse_loss_metric.avg * 255 ** 2 / 3:.3f} |"
        f"\tBpp loss: {bpp_loss_metric.avg:.2f} |"
        f"\tAux loss: {aux_loss_metric.avg:.2f}\n"
    )
    return loss_tot_metric.avg, bpp_loss_metric.avg, mse_loss_metric.avg, aux_loss_metric.avg, psnr_metric.avg, ssim_metric.avg





def compress_one_epoch(model, test_dataloader, device):
    bpp_metric = AverageMeter()
    psnr_metric = AverageMeter()
    mssim_metric = AverageMeter()

    
    with torch.no_grad():
        for i,d in enumerate(test_dataloader): 
            print("-------------    image ",i,"  --------------------------------")
    
            d = d.to(device)
        
            x_padded, padding = pad(d, 128)
            
            
            out_enc = model.compress(x_padded)
            out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
            
            out_dec["x_hat"] = crop(out_dec["x_hat"], padding)

            
            metrics = compute_metrics(d, out_dec["x_hat"], 255)
            num_pixels = d.size(0) * d.size(2) * d.size(3)
            bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
            
            psnr_metric.update(metrics["psnr"])
            mssim_metric.update(metrics["ms-ssim"]) 
            bpp_metric.update(bpp)  

    print(
        f"Average metrics:"
        f"\tPSNR: {psnr_metric.avg:.3f} |"
        f"\tMSSIM: {mssim_metric.avg * 255 ** 2 / 3:.3f} |"
        f"\tBpp: {bpp_metric.avg:.2f} \n"
    )

    
    return bpp_metric.avg, psnr_metric.avg, mssim_metric.avg


def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)


def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )