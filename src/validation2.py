# import sys
# sys.path.append("/home/ids/flauron-23/STF-QVRF")
# from compressai.entropy_models import GaussianConditional
# from compressai.models.priors import CompressionModel

#from Cheng2020Attention import Cheng2020Attention

from opt import parse_args
from utils import seed_all
# import os
# import wandb

from experiment import Experiment, Context
from utils import save_checkpoint

from custom_comp.zoo import models

from custom_comp.models import SymmetricalTransFormer, WACNN,QVRFCheng,stfQVRF
from custom_comp.zoo import load_model_from_checkpoint,load_mask,models
from custom_comp.zoo.pretrained import load_pretrained as load_state_dict
from utils.dataset import TestKodakDataset,TestClicDataset
from utils.engine import AverageMeter
from utils import compute_metrics, crop,pad,apply_saved_mask,delete_mask

from compressai.models.waseda import Cheng2020Attention
from compressai.models.google import MeanScaleHyperprior
from compressai.zoo import cheng2020_attn,mbt2018_mean
# from compressai.zoo.pretrained import load_pretrained

from torch import load,zeros_like
from torch.utils.data import DataLoader
from typing import Union

import torch
from tqdm import tqdm
import numpy as np
def inference(model,x, x_padded, padding):

    out_enc = model.compress(x_padded)
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])

    out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
    # out_dec["x_hat"] = F.pad(out_dec["x_hat"], padding) 

    metrics = compute_metrics(x, out_dec["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    rate = bpp*num_pixels 

    return metrics, torch.tensor([bpp]), rate, out_dec["x_hat"]

def inference_with_scale(model,x, x_padded, padding,factor,s=2):
    
    out_enc = model.compress(x_padded, s, factor)
    # out_enc = model.compress(x_padded)
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"], s, factor)

    out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
    # out_dec["x_hat"] = F.pad(out_dec["x_hat"], padding) 

    metrics = compute_metrics(x, out_dec["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    rate = bpp*num_pixels

    return metrics, torch.tensor([bpp]), rate, out_dec["x_hat"]

def eval_models(models, dataloader, device):

    print("Starting inferences")
    # estimate_entropy = False # False -> Use the AC for compression 

    res_metrics = {}
    for j,x in enumerate(tqdm(dataloader)):

        if j>= 30:
            break

        x = x.to(device)
            
        # TCM padding 
        x_padded, padding = pad(x, 128)
        
        
        for model_type in list(models.keys()):
            print("Evaluating model type:", model_type)


            for qp in list(models[model_type].keys()):
                print("Evaluating model:", qp)
                model = models[model_type][qp]['model']


                # if not estimate_entropy:
                #     inference_fn = inference
                # else:
                #     inference_fn = inference_entropy_estimation

                if model_type == "ours":
                    parameters_to_prune = {}
                    parameters_to_prune["g_a"] = [(module, "weight") for module in filter(lambda m: type(m) in [torch.nn.Conv2d, torch.nn.Linear,torch.nn.ConvTranspose2d], model.g_a.modules())]
                    parameters_to_prune["g_s"] = [(module, "weight") for module in filter(lambda m: type(m) in [torch.nn.Conv2d, torch.nn.Linear,torch.nn.ConvTranspose2d], model.g_s.modules())]

                    #Apply mask
                    
                    if qp < 5 :
                        apply_saved_mask(model, mask["g_a"][qp])
                        apply_saved_mask(model, mask["g_s"][qp])
                    
                    metrics, bpp, rate, x_hat = inference(model, x, x_padded, padding)

                    if qp < 5 :
                        delete_mask(model, parameters_to_prune["g_a"])
                        delete_mask(model, parameters_to_prune["g_s"])

                elif model_type in ["cheng", "msh"]:

                    metrics, bpp, rate, x_hat = inference(model, x, x_padded, padding)

                elif model_type == "qvrf":
                    factor = qvref_factor[qp]  #-1 qp start from 1
                    metrics, bpp, rate, x_hat = inference_with_scale(model, x, x_padded, padding, factor, s=2)
                
                elif model_type == "gained":
                    model.update(force=True)
                    s= gained_factor[qp]  #-1 qp start from 1
                    metrics, bpp, rate, x_hat = inference_with_scale(model, x, x_padded, padding, 0.0, s=s)


                models[model_type][qp]['psnr'].update(metrics["psnr"])
                models[model_type][qp]['ms_ssim'].update(metrics["ms-ssim"])
                models[model_type][qp]['bpps'].update(bpp.item())
                models[model_type][qp]['rate'].update(rate)


    for model_type in list(models.keys()):
        model_res = {}
        for qp in list(models[model_type].keys()):
            model_res[qp] = {
                'psnr': models[model_type][qp]['psnr'].avg,
                'mssim': models[model_type][qp]['ms_ssim'].avg,
                'bpp': models[model_type][qp]['bpps'].avg,
                'rate': models[model_type][qp]['rate'].avg,
                'loss': models[model_type][qp]['loss'].avg
            }
            print(f'{qp}: {model_res[qp]}')
        res_metrics[model_type] = model_res

    return res_metrics   



def mask_to_device(mask, device="cuda"):
    for key, mask_list in mask.items():
        for i in range(len(mask_list)):
            for k, v in mask_list[i].items():
                mask_list[i][k] = v.to(device)

#Load Cheng magv mask
mask_path="/home/ids/flauron-23/MagV/data/magv_04_cheng_unstructured/masks/mask_magv_04_cheng_unstructured.pth"
mask = load_mask(mask_path=mask_path)
mask_to_device(mask, device="cuda")

#Load STF magv
#checkpoint_path = "/home/ids/flauron-23/MagV/data/magv_04_stf_unstructured/models/magv_04_stf_unstructured_checkpoint_best.pth.tar"
#mask_path="/home/ids/flauron-23/MagV/data/magv_04_stf_unstructured/masks/mask_magv_04_stf_unstructured.pth"
#stf_magv =  load_model_from_checkpoint(SymmetricalTransFormer,checkpoint_path=checkpoint_path)
#stf_mask = load_mask(mask_path=mask_path)
qvref_factor = [0.5, 0.7, 0.9, 1.1, 1.25, 1.45, 1.7, 2.0, 2.4, 2.8, 3.3, 3.8, 4.0,4.6, 5.4, 5.8, 6.5, 6.8, 7.5, 7.9, 8.3, 9.1, 9.4, 9.7, 10.5, 11, 12]
# gained_lambda =  [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0003]
# gained_factor = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
gained_factor = [0,1,2,3,4,5,6]
device = "cpu"
nets = {}
metrics = {}
qp_range = {
    # "cheng": range(1, 7),
    # "msh": range(1, 9),

    # "ours": range(6),   #   index of mask in the mask list 0 to 4 mask , 5 anchor model
    # "qvrf": range(27),  #   index of qvrf factor
    "gained": range(7), #   index of gained factor
}

def make_entry(model):
    return {
        "model": model,
        "psnr": AverageMeter(),
        "ms_ssim": AverageMeter(),
        "bpps": AverageMeter(),
        "rate": AverageMeter(),
        "criterion": None,
        "loss": AverageMeter(),
    }

def build_model(arch, qp, device):
    if arch == "cheng":
        return cheng2020_attn(quality=qp, pretrained=True).eval().to(device)
    if arch == "msh":
        return mbt2018_mean(quality=qp, pretrained=True).eval().to(device)
    # others reuse the same model
    return models[arch]().eval().to(device)


nets = {}
for arch, qps in qp_range.items():
    res = {}

    print("arch:", arch)
    #Non VR models
    if arch in["cheng","msh"]:
        for qp in qps:
            print("  qp:", qp)
            model = build_model(arch, qp, device)
            res[qp] = make_entry(model)
            print(res.keys())
    # VR models
    else:
        model = build_model(arch, None, device)
        for qp in qps:
            print("  qp:", qp)
            res[qp] = make_entry(model)
            print(res.keys())
    
    nets[arch] = res
    print("nets keys:", nets.keys())

for dataset in ["kodak"]:

    if dataset=="kodak":
        dataset = TestKodakDataset("/home/ids/flauron-23/kodak")
    if dataset=="clic":
        dataset = TestClicDataset("/home/ids/flauron-23/clic")   

    test_dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=4)

    metrics[dataset] = eval_models(nets, test_dataloader, device)


print("success")






# for model_arch in ["cheng","ours","qvrf","msh","gained"]:

#     if model_arch in["cheng","msh"]:
#         res = {}
#         if model_arch=="cheng":
#             for qp in range(1,7):
#                 net = cheng2020_attn(quality=qp,pretrained=True).eval().to(device)
#                 res[f'q{qp}-model'] = {   
#                     "model": net,
#                     "psnr": AverageMeter(),
#                     "ms_ssim": AverageMeter(),
#                     "bpps": AverageMeter(),
#                     "rate": AverageMeter(),
#                     "criterion": None,
#                     "loss": AverageMeter()
#                 }
#         if model_arch=="msh":
#             for qp in range(1,9):
#                 net = mbt2018_mean(quality=qp,pretrained=True).eval().to(device)
#                 res[f'q{qp}-model'] = {   
#                     "model": net,
#                     "psnr": AverageMeter(),
#                     "ms_ssim": AverageMeter(),
#                     "bpps": AverageMeter(),
#                     "rate": AverageMeter(),
#                     "criterion": None,
#                     "loss": AverageMeter()
#                 }
#     else:
#         res = {}
#         net = models[model_arch]
#         if model_arch == "ours":
            
#             for qp in range(1,7):
#                 res[f'q{qp}-model'] = {   
#                     "model": net,
#                     "psnr": AverageMeter(),
#                     "ms_ssim": AverageMeter(),
#                     "bpps": AverageMeter(),
#                     "rate": AverageMeter(),
#                     "criterion": None,
#                     "loss": AverageMeter()
#                 }
#         elif model_arch == "qvrf":
#             for qp in range(1,28):
#                 res[f'q{qp}-model'] = {   
#                     "model": net,
#                     "psnr": AverageMeter(),
#                     "ms_ssim": AverageMeter(),
#                     "bpps": AverageMeter(),
#                     "rate": AverageMeter(),
#                     "criterion": None,
#                     "loss": AverageMeter()
#                 }
#         elif model_arch == "gained":
#             for qp in [1,2,3,4,5,6]:
#                 res[f'q{qp}-model'] = {   
#                     "model": net,
#                     "psnr": AverageMeter(),
#                     "ms_ssim": AverageMeter(),
#                     "bpps": AverageMeter(),
#                     "rate": AverageMeter(),
#                     "criterion": None,
#                     "loss": AverageMeter()
#                 }

#     nets[model_arch] = res


    # if model_name in["qvrf","stfqvrf","gained"]:
    #     net = models[model_name]()
    #     net.to("cuda")
    # else:
    #     if model_name == "chengmagv":
    #         net = cheng_magv 
    #     elif model_name == "stfgmagv":
    #         net = stf_magv 
        



            # if qvref
            # for br in [bitrates for bitrates in []
                # out_enc = net.compress(x_padded, s, factor)

            # if ours

            # gained