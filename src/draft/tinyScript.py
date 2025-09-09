import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from collections import defaultdict

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

import numpy as np
import json

from compressai.zoo import cheng2020_attn
import torch.nn.utils.prune as prune


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 8)
        self.fc3 = nn.Linear(8, 4)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = TinyMLP()



#TODO: add comments
def compute_neuron_norm(parameters_to_prune):
    norms = {}
    for module, _ in parameters_to_prune:
        W = module.weight.data

        if isinstance(module, torch.nn.Conv2d):
            # Norm per filter (output channel)
            neuron_norms = torch.norm(W.view(W.size(0), -1), p=2, dim=1)
  

        elif isinstance(module, torch.nn.Linear):
            # Norm per neuron (row of weight matrix)
            neuron_norms = torch.norm(W, p=2, dim=1)

        norms[module] = neuron_norms

    # Normalization of the norms
    norms = {k: v/len(norms) for k, v in norms.items()}

    return norms


def get_global_list(norms_dict):
    global_list = []  # list of (module, neuron_index, norm_value)

    for module, neuron_norms in norms_dict.items():
        for idx, norm_value in enumerate(neuron_norms):
            global_list.append((module, idx, norm_value.item()))
    
    return global_list


def get_neuron_to_prune(global_list, amount):
    global_list.sort(key=lambda x: x[2], reverse=True)
    total_neurons = len(global_list)
    num_to_prune = int(amount * total_neurons)

    neurons_to_prune = global_list[-num_to_prune:]
    return neurons_to_prune

def group_by_module(neurons_to_prune):
    module_to_indices = defaultdict(list)
    for module, idx, _ in neurons_to_prune:
        module_to_indices[module].append(idx)
    return module_to_indices

def build_and_put_mask_on_module(module_to_indices):
    for module, indices in module_to_indices.items():

        mask = torch.ones_like(module.weight.data)
        mask[indices] = 0  # zero out selected neurons
        prune.custom_from_mask(module, name="weight", mask=mask)

def check_neuron_sparsity(parameters_to_prune):
    total_neurons = 0
    zeroed_neurons = 0
    for module, _ in parameters_to_prune_g_a:
        W = module.weight.data
        if isinstance(module, nn.Conv2d):
            for i in range(W.size(0)):  # output channels
                total_neurons += 1
                if torch.all(W[i] == 0):
                    zeroed_neurons += 1
        elif isinstance(module, nn.Linear):
            for i in range(W.size(0)):  # rows
                total_neurons += 1
                if torch.all(W[i] == 0):
                    zeroed_neurons += 1

    print(f"Neuron sparsity: {zeroed_neurons/total_neurons:.2%}")

print("Model: cheng2020_attn")
#refnet = cheng2020_attn(quality=1,pretrained=True).to("cuda")
refnet = TinyMLP().to("cuda")
#get the parameters to prune
print("Getting parameters to prune")
parameters_to_prune_g_a = [(module, "weight") for module in filter(lambda m: type(m) in [torch.nn.Conv2d, torch.nn.Linear], refnet.modules())]
#parameters_to_prune_g_a = [(module, "weight") for module in filter(lambda m: type(m) in [torch.nn.Conv2d, torch.nn.Linear], refnet.g_a.modules())]
#parameters_to_prune_g_s = [(module, "weight") for module in filter(lambda m: type(m) in [torch.nn.Conv2d, torch.nn.Linear], refnet.g_s.modules())]

#print(parameters_to_prune_g_a)

print("Computing neuron norms")
neurons_norm_ga=compute_neuron_norm(parameters_to_prune_g_a)
global_list_ga=get_global_list(neurons_norm_ga)
module_to_indices_ga=group_by_module(get_neuron_to_prune(global_list_ga,0.2))
build_and_put_mask_on_module(module_to_indices_ga)


print("!!!!!!!!!!!!!!!!!All done!!!!!!!!!!!!!!!!")
print("Run checks")

# W=refnet.g_a[0].conv2.weight
# total_params = W.numel()                 # total number of elements
# zero_params = torch.sum(W == 0).item()   # count of zeros
# print(f"Total params: {total_params}, Zeros: {zero_params}, Sparsity: {zero_params/total_params:.2%}")


check_neuron_sparsity(parameters_to_prune_g_a)

mask_ga=save_mask(refnet)
apply_saved_mask(refnet, mask_ga)


# nb_layers_pruned = 0

# for module in refnet.g_a.modules():  # iterate through all submodules
#     if isinstance(module, (nn.Linear, nn.Conv2d)) and module.weight is not None:
#         # For Linear: rows correspond to neurons
#         # For Conv2d: rows are output channels (dim=0)
#         if torch.any(torch.all(module.weight == 0, dim=1)):
#             nb_layers_pruned += 1

# print(f"Number of layers with at least one pruned neuron: {nb_layers_pruned}")

# print(refnet.g_a[2].conv2.weight)





import torch
import json

state_dict = refnet.state_dict()
readable_dict = {k: v.cpu().numpy().tolist() for k, v in state_dict.items()}

with open("model_weights_pretty.json", "w") as f:
    json.dump(readable_dict, f, indent=2)  # indent=2 adds newlines and spacing

# total_neurons = 0
# zero_neurons = 0

# for module_name, mask in mask_ga.items():
#     if mask.dim() == 4:  # Conv2d
#         for i in range(mask.size(0)):  # output channels
#             total_neurons += 1
#             if torch.all(mask[i] == 0):
#                 zero_neurons += 1
#     elif mask.dim() == 2:  # Linear
#         for i in range(mask.size(0)):  # rows
#             total_neurons += 1
#             if torch.all(mask[i] == 0):
#                 zero_neurons += 1

# print(f"Total neurons: {total_neurons}")
# print(f"Zeroed neurons: {zero_neurons}")
# print(f"Neuron-level sparsity: {zero_neurons/total_neurons:.2%}")
#def global_structured_pruning(parameters_to_prune, amount,n)

#### compute the norm of each neuron in the network

#make mask for module
#apply mask