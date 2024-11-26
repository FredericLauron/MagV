import torch
from torch import nn
from lora.layers import LinearWithLoRA, LoRALayer
from copy import deepcopy
from custom_comp.layers import BasicLayer, SwinTransformerBlock
import lora

def merge_lora(lora):
    fc_eq = nn.Linear(lora.W_a.shape[0], lora.W_b.shape[1], bias=False)
    # alpha is multiplied
    fc_eq.weight = nn.Parameter(lora.alpha * torch.mm(lora.W_a, lora.W_b).T)

    return fc_eq


def sum_parallel_layers(fc1, lora):
    fc_sum = nn.Linear(fc1.in_features, fc1.out_features)
    fc_sum.weight = nn.Parameter(fc1.weight + lora.weight)
    # add check for bias
    fc_sum.bias.data = nn.Parameter(fc1.bias)
    return fc_sum

def get_merged_lora(model):
    for i,layer in enumerate(model.layers): # encoder
        if isinstance(layer, BasicLayer):
            for swin_block in layer.blocks:
                if isinstance(swin_block, SwinTransformerBlock):
                    swin_block.mlp.fc1 = sum_parallel_layers(swin_block.mlp.fc1.linear, merge_lora(swin_block.mlp.fc1.lora))
                if isinstance(swin_block, SwinTransformerBlock):
                    swin_block.mlp.fc2 = sum_parallel_layers(swin_block.mlp.fc2.linear, merge_lora(swin_block.mlp.fc2.lora))

    for i,layer in enumerate(model.syn_layers): # decoder
        if isinstance(layer, BasicLayer):
            for swin_block in layer.blocks:
                # print(swin_block)
                if isinstance(swin_block, SwinTransformerBlock):
                    swin_block.mlp.fc1 = sum_parallel_layers(swin_block.mlp.fc1.linear, merge_lora(swin_block.mlp.fc1.lora))
                if isinstance(swin_block, SwinTransformerBlock):
                    swin_block.mlp.fc2 = sum_parallel_layers(swin_block.mlp.fc2.linear, merge_lora(swin_block.mlp.fc2.lora))
                
    return model

if __name__=='__main__':
    pass