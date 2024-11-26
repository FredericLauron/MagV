from functools import partial
from .layers import LinearWithLoRA
import sys
import torch.nn as nn
import yaml

from custom_comp.zoo import models
from lora.merge import get_merged_lora
import torch
from custom_comp.zoo.pretrained import load_pretrained


from custom_comp.layers import BasicLayer, SwinTransformerBlock

def get_lora_model(model, config, force_alpha = None):
    print('Get LoRA model!')

    with open(config, 'r') as file:
        lora_conf = yaml.safe_load(file)

    lora_r = lora_conf['r'] # 8

    if force_alpha is None:
        lora_alpha = lora_conf['alpha'] # 16
    else:
        lora_alpha = force_alpha

    lora_mlp_fc1 = lora_conf['mlp_fc1'] # True
    lora_mlp_fc2 = lora_conf['mlp_fc2'] # True

    lora_encoder = lora_conf['encoder'] # [True,True,True,True]
    lora_decoder = lora_conf['decoder'] #[True,True,True,True]

    print('Configs:')
    print(f'r: {lora_r}')
    print(f'alpha: {lora_alpha}')
    print(f'mlp_fc1: {lora_mlp_fc1}')
    print(f'mlp_fc2: {lora_mlp_fc2}')
    print(f'encoder: {lora_encoder}')
    print(f'decoder: {lora_decoder}')




    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)
    
    
    

    for i,layer in enumerate(model.layers): # encoder
        if isinstance(layer, BasicLayer) and lora_encoder[i]:
            for swin_block in layer.blocks:
                # print(swin_block)
                if lora_mlp_fc1 and isinstance(swin_block, SwinTransformerBlock):
                    # weigths.append(swin_block.mlp.fc1.weight)
                    swin_block.mlp.fc1 = assign_lora(swin_block.mlp.fc1)
                if lora_mlp_fc2 and isinstance(swin_block, SwinTransformerBlock):
                    swin_block.mlp.fc2 = assign_lora(swin_block.mlp.fc2)

    for i,layer in enumerate(model.syn_layers): # decoder
        if isinstance(layer, BasicLayer) and lora_decoder[i]:
            for swin_block in layer.blocks:
                # print(swin_block)
                if lora_mlp_fc1 and isinstance(swin_block, SwinTransformerBlock):
                    swin_block.mlp.fc1 = assign_lora(swin_block.mlp.fc1)
                if lora_mlp_fc2 and isinstance(swin_block, SwinTransformerBlock):
                    swin_block.mlp.fc2 = assign_lora(swin_block.mlp.fc2)





    return model



def get_vanilla_finetuned_model(model):
    print('Get Vanilla FineTuned model!')


    lora_mlp_fc1 = True
    lora_mlp_fc2 = True

    lora_encoder = [True,True,True,True]
    lora_decoder = [True,True,True,True]

    print('Configs:')
    print(f'mlp_fc1: {lora_mlp_fc1}')
    print(f'mlp_fc2: {lora_mlp_fc2}')
    print(f'encoder: {lora_encoder}')
    print(f'decoder: {lora_decoder}')

    
    
    
    # weigths = []

    for i,layer in enumerate(model.layers): # encoder
        if isinstance(layer, BasicLayer) and lora_encoder[i]:
            for swin_block in layer.blocks:
                # print(swin_block)
                if lora_mlp_fc1 and isinstance(swin_block, SwinTransformerBlock):
                    for param in swin_block.mlp.fc1.parameters():
                        param.requires_grad = True
                    # weigths.append(swin_block.mlp.fc1.weight)
                if lora_mlp_fc2 and isinstance(swin_block, SwinTransformerBlock):
                    for param in swin_block.mlp.fc2.parameters():
                        param.requires_grad = True

    for i,layer in enumerate(model.syn_layers): # decoder
        if isinstance(layer, BasicLayer) and lora_decoder[i]:
            for swin_block in layer.blocks:
                # print(swin_block)
                if lora_mlp_fc1 and isinstance(swin_block, SwinTransformerBlock):
                    for param in swin_block.mlp.fc1.parameters():
                        param.requires_grad = True
                if lora_mlp_fc2 and isinstance(swin_block, SwinTransformerBlock):
                    for param in swin_block.mlp.fc2.parameters():
                        param.requires_grad = True



    return model



if __name__ == '__main__':

    pass