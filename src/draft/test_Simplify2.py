import torch
import torch.fx as fx
import torch.nn as nn
from torch.nn.utils import prune

from simplify import simplify
import simplify.utils as sutils

from compressai.zoo import cheng2020_attn,mbt2018_mean
from compressai.layers.gdn import GDN
from utils import apply_saved_mask
from collections import OrderedDict

# import torch
# from torchvision import models
# from simplify import simplify

# # model = models.resnet18()
# model = models.mobilenet_v3_small(pretrained=False)

# # Apply some pruning strategy or load a pruned checkpoint

# dummy_input = torch.zeros(1, 3, 224, 224)  # Tensor shape is that of a standard input for the given model
# simplified_model = simplify(model, dummy_input)


def load_model_from_checkpoint(checkpoint_path,factory_function,quality=6,pretrained=True):

    # Load the model
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    net = factory_function(quality=quality, pretrained=pretrained)
    net = net.to("cuda")

    # # Load the state dict ignoring _quantized_cdf and _cdf_length
    state_dict = checkpoint['state_dict']
    if 'entropy_bottleneck._quantized_cdf' in state_dict:
        state_dict['entropy_bottleneck._quantized_cdf'] = torch.zeros_like(net.entropy_bottleneck._quantized_cdf)
    if 'entropy_bottleneck._cdf_length' in state_dict:
        state_dict['entropy_bottleneck._cdf_length'] = torch.zeros_like(net.entropy_bottleneck._cdf_length)

    net.load_state_dict(state_dict,strict=False)

    #update entropy model
    net.entropy_bottleneck.update(force=True)

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


def replace_gdn_with_identity(model):
    """
    Recursively replaces all GDN layers in the given model with nn.Identity().
    Modifies the model in place and also returns it.
    """
    for name, module in model.named_children():
        # If this child is a GDN layer → replace it
        if isinstance(module, GDN):
            setattr(model, name, nn.Identity())
        
        else:
            # Recursively check inside submodules
            replace_gdn_with_identity(module)

    return model

##########################################################################

def prepend_identity(model, C):
    id_conv = nn.Conv2d(C, C, kernel_size=1, bias=False).to("cuda")

    with torch.no_grad():
        id_conv.weight.zero_()
        for i in range(C):
            id_conv.weight[i, i, 0, 0] = 1.0

    new_layers = OrderedDict()
    new_layers["identity_conv"] = id_conv

    for name, layer in model.g_s.named_children():
        new_layers[name] = layer

    model.g_s = nn.Sequential(new_layers)
###########################################################################

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

if __name__ == "__main__":

    # checkpoint_path = "/home/ids/flauron-23/MagV/data/magv_02_cheng_structured/models/magv_02_cheng_structured_checkpoint_best.pth.tar"
    # mask_path = "/home/ids/flauron-23/MagV/data/magv_02_cheng_structured/masks/mask_magv_02_cheng_structured.pth"
    checkpoint_path = "/home/ids/flauron-23/MagV/data/magv_04_cheng_unstructured/models/magv_04_cheng_unstructured_checkpoint_best.pth.tar"
    mask_path="/home/ids/flauron-23/MagV/data/magv_04_cheng_unstructured/masks/mask_magv_04_cheng_unstructured.pth"
    net = load_model_from_checkpoint(checkpoint_path=checkpoint_path,factory_function=cheng2020_attn)
    mask = load_mask(mask_path=mask_path)

    net.update(force=True)
    # apply_saved_mask(net.g_a, mask["g_a"][0])
    # apply_saved_mask(net.g_s, mask["g_s"][0])

    for i in range(len(mask["g_a"])):
        
        net = load_model_from_checkpoint(checkpoint_path=checkpoint_path,factory_function=cheng2020_attn)
        mask = load_mask(mask_path=mask_path)

        apply_saved_mask(net.g_a, {k: v.to("cuda") for k, v in mask["g_a"][i].items()})
        apply_saved_mask(net.g_s, {k: v.to("cuda") for k, v in mask["g_s"][i].items()})

        remove_pruning(net.g_a)
        remove_pruning(net.g_s)

        # Replace GDN with Identity    
        net.g_a =  replace_gdn_with_identity(net.g_a)
        net.g_s =  replace_gdn_with_identity(net.g_s)

        print(f"Number of neurons before simplification for mask{i}:")
        total_neurons_before_g_a = count_neurons_weight(net.g_a)
        print(total_neurons_before_g_a)

        # #g_a
        g_a_dummy_input = torch.zeros(1, 3, 512, 768).to("cuda")
        simplified_g_a = simplify(net.g_a, g_a_dummy_input)

        print(f"Number of neurons after g_a simplification for mask{i}:")
        total_neurons_after_g_a = count_neurons_weight(simplified_g_a)
        print(total_neurons_after_g_a)

        #g_s
        # building dummy input for g_s
        y = net.g_a(g_a_dummy_input)
        y_hat = net.gaussian_conditional.quantize(y, "dequantize")

        # Add a convolution layer at the beginning of g_s to make simplify have a convolution as first layer
        # to rely on 
        prepend_identity(net, y_hat.shape[1])

        print(f"Number of neurons before simplification for mask{i}:")
        total_neurons_before_g_s = count_neurons_weight(net.g_s)
        print(total_neurons_before_g_s)

        # Apply simplify  
        simplified_g_s = simplify(net.g_s, y_hat)

        print(f"Number of neurons after simplification for mask{i}:")
        total_neurons_after_g_s = count_neurons_weight(simplified_g_s)
        print(total_neurons_after_g_s)

        # print(net.g_a)
        print("succeess")