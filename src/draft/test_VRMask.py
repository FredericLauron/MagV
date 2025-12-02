import torch
import torch.fx as fx
import torch.nn as nn
from torch.nn.utils import prune

from simplify import simplify
import simplify.utils as sutils

from compressai.zoo import cheng2020_attn,mbt2018_mean
from compressai.models.waseda import Cheng2020Attention
from compressai.layers.gdn import GDN
from utils import apply_saved_mask,save_mask,group_by_module,delete_mask
from experiment import Experiment 
from collections import OrderedDict
from custom_comp.zoo import  load_state_dict

from itertools import chain
from collections import defaultdict

from opt import parse_args
import os
import sys
import warnings

import wandb

warnings.simplefilter("ignore", FutureWarning)

warnings.filterwarnings(
    "ignore",
    message=r"autocast",
    category=FutureWarning,
)

amount=0.5
log_wandb = True

def load_model_from_checkpoint(checkpoint_path,factory_function,quality=6,pretrained=True,adapter=False):

    # Load the checkpoint
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    state_dict = load_state_dict(state_dict=state_dict)

    # Create the model
    net = Cheng2020Attention().to("cuda")
    net.load_state_dict(state_dict["state_dict"])

    # #update entropy model
    net.update(force=True)

    return net

def load_mask(mask_path):
    mask = torch.load(mask_path, map_location='cpu')
    # print(type(mask["g_a"][0])) 
    # mask=mask.to("cuda")
    return mask

def remove_pruning(model):
    for module in model.modules():
        for param_name in ["weight", "bias"]:
            # Check if this param was pruned
            if hasattr(module, f"{param_name}_mask"):
                prune.remove(module, param_name)

def count_neurons_weight(model):
    """
    Counts the number of neurons in Conv2d and Linear layers
    based on the weight tensors.
    """
    total_neurons = 0
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            # For Conv2d: one neuron per output channel
            total_neurons += layer.out_channels
        elif isinstance(layer, nn.Linear):
            # For Linear: one neuron per output feature
            total_neurons += layer.out_features
    return total_neurons


###############################################################################################""



def get_superset_mask(mask,idx_inf,idx_sup):
    #Retrive element of the superset mask that are non in the subset mask 
    superset_Mask ={}
    for (k,v),(k2,v2) in zip(mask[idx_inf].items(),mask[idx_sup].items()):
        superset_Mask[k] = torch.bitwise_xor(v.bool(),v2.bool()).float()
    return superset_Mask

def get_superset_nonzero_parameters(net):
    superSet_data = []
    for _, module in net.named_modules():
        if torch.nn.utils.prune.is_pruned(module) and hasattr(module, "weight"):

            w = torch.flatten(module.weight.data)

            # Retrive non zero indices 
            NonZeroIndex = torch.nonzero(w).squeeze()

            # Retrive non zero values
            NonZeroValue=(w[NonZeroIndex])

            superSet_data.extend( [(module, idx,v) for idx ,v in zip(NonZeroIndex,NonZeroValue)])
    return superSet_data

def get_superset_param_to_prune(superSet_data,amount=0.5):
    superSet_data_sorted_desc = sorted(superSet_data, key=lambda x: x[2]) #sort ascending to target param to prune
    # Find the neurons to prune based on the sorted superset data
    superSet_idxPruning = int(len(superSet_data_sorted_desc) * amount)
    superSet_paramToPrune = superSet_data_sorted_desc[:superSet_idxPruning]
    return superSet_paramToPrune

def update_superset_mask(net,module_to_indices):
    # Find the neuron's parameters indices to prune
    # Put a zero into the superset mask at these indices
    for _, module in net.named_modules():
        if torch.nn.utils.prune.is_pruned(module) and hasattr(module, "weight"):
            for idx in module_to_indices[module]:
                idx = torch.unravel_index(idx, module.weight.shape)
                module.weight_mask.data[idx] = 0

def fuse_masks(superset_mask,inter_mask):
    # Fuse the smaller mask with the modified superset mask to create a new mask
    for name in inter_mask.keys():
        inter_mask[name] = inter_mask[name].to("cuda").bool()
        superset_mask[name] = superset_mask[name].bool()
        inter_mask[name] = torch.bitwise_or(inter_mask[name], superset_mask[name])
    return inter_mask

def mask_to_device(mask, device="cuda"):
    for key, mask_list in mask.items():
        for i in range(len(mask_list)):
            for k, v in mask_list[i].items():
                mask_list[i][k] = v.to(device)

if __name__ == "__main__":

    checkpoint_path = "/home/ids/flauron-23/MagV/data/magv_04_cheng_unstructured/models/magv_04_cheng_unstructured_checkpoint_best.pth.tar"
    mask_path="/home/ids/flauron-23/MagV/data/magv_04_cheng_unstructured/masks/mask_magv_04_cheng_unstructured.pth"
    parameters_to_prune_path = "/home/ids/flauron-23/MagV/data/magv_04_cheng_unstructured/masks/parameters_to_prune_magv_04_cheng_unstructured.pth"
    net = load_model_from_checkpoint(checkpoint_path=checkpoint_path,factory_function=cheng2020_attn)
    mask = load_mask(mask_path=mask_path)

    #parameters_to_prune = torch.load(parameters_to_prune_path, map_location='cpu') 
    parameters_to_prune = {}  
    parameters_to_prune["g_a"] = [(module, "weight") for module in filter(lambda m: type(m) in [torch.nn.Conv2d, torch.nn.Linear,torch.nn.ConvTranspose2d], net.g_a.modules())]
    parameters_to_prune["g_s"] = [(module, "weight") for module in filter(lambda m: type(m) in [torch.nn.Conv2d, torch.nn.Linear,torch.nn.ConvTranspose2d], net.g_s.modules())]
    
    mask_to_device(mask, device="cuda")
    net = net.to("cuda")

    # Retrive element of the superset mask that are non in the subset mask 
    superset_Mask_g_a=get_superset_mask(mask["g_a"],0,1)
    superset_Mask_g_s=get_superset_mask(mask["g_s"],0,1)

    # apply the superset mask
    apply_saved_mask(net.g_a, superset_Mask_g_a)
    apply_saved_mask(net.g_s, superset_Mask_g_s)

    # Retrive the non zero parameters of the superset mask
    superSet_data_g_a=get_superset_nonzero_parameters(net.g_a)
    superSet_data_g_s=get_superset_nonzero_parameters(net.g_s)

    # Find the parameters to prune in the superset mask
    superSet_paramToPrune_g_a=get_superset_param_to_prune(superSet_data_g_a,amount)
    superSet_paramToPrune_g_s=get_superset_param_to_prune(superSet_data_g_s,amount)

    # Group the parameters indices by module
    module_to_indices_g_a=group_by_module(superSet_paramToPrune_g_a)
    module_to_indices_g_s=group_by_module(superSet_paramToPrune_g_a)

    # Find the neuron's parameters indices to prune
    # Put a zero into the superset mask at these indices
    update_superset_mask(net.g_a,module_to_indices_g_a)
    update_superset_mask(net.g_s,module_to_indices_g_s)

    # Load the smaller mask (subset)
    inter_mask_g_a = mask["g_a"][0].copy()
    inter_mask_g_s = mask["g_s"][0].copy()

    superset_mask_g_a = save_mask(net.g_a)
    superset_mask_g_s = save_mask(net.g_s)

    # Fuse the smaller mask with the modified superset mask to create a new mask
    inter_mask_g_a = fuse_masks(superset_mask_g_a,inter_mask_g_a)
    inter_mask_g_s = fuse_masks(superset_mask_g_s,inter_mask_g_s)

    # Apply the new mask to both g_a and g_s
    mask["g_a"].insert(1, inter_mask_g_a)
    mask["g_s"].insert(1, inter_mask_g_s)

    #==================
    # test applying the new mask
    #==================
    delete_mask(net.g_a,parameters_to_prune["g_a"])
    delete_mask(net.g_s,parameters_to_prune["g_s"])

    # apply_saved_mask(net.g_a, mask["g_a"][1])
    # apply_saved_mask(net.g_s, mask["g_s"][1])

    project_run_path=os.path.dirname(__file__)

    args = parse_args()

    if log_wandb:
        wandb.init(
            project='training',
            entity='MagV',
            name=f'{args.nameRun}',
            config=vars(args)
        )

    exp = Experiment(args,project_run_path)
    exp.ctx.net = net
    exp.ctx.all_mask = mask
    exp.ctx.parameters_to_prune = parameters_to_prune

    exp.make_plot(epoch=0)

    print("succeess")