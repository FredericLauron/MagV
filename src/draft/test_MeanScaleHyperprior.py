# Test on MeanScaleHyperprior model
import torch
import torch.nn as nn

from utils.masks import *
from compressai.zoo import mbt2018_mean

def test_mask_generation_MeanScaleHyperprior():
    """
    Test mask generation on MeanScaleHyperprior which uses ConvTransposed2d convolution
    """
    alpha = np.linspace(0.0, 0.5, 6)[::-1]
    _ , amounts = lambda_percentage(alpha, amount = 0.5)

    print("amounts",amounts)
    print("lambda",_)

    net = mbt2018_mean(quality=6,pretrained=True)
    all_mask, parameters_to_prune = {},{}

    all_mask["g_a"], parameters_to_prune["g_a"] = generate_mask_from_structured(net.g_a, amounts)
    all_mask["g_s"], parameters_to_prune["g_s"] = generate_mask_from_structured(net.g_s, amounts)


    for index in range(len(amounts)-1): #No mask on anchor model this is why len(amounts)-1

        apply_saved_mask(net.g_a,all_mask["g_a"][index])
        apply_saved_mask(net.g_s,all_mask["g_s"][index])

        assert torch.allclose(  check_neuron_sparsity(parameters_to_prune["g_a"]),\
                                torch.tensor([amounts[index]]),\
                                atol=1e-3),\
                                f"The amount of pruned neuron in g_a is not {amounts[index]}"
        
        assert torch.allclose(  check_neuron_sparsity(parameters_to_prune["g_s"]),\
                                torch.tensor([amounts[index]]),\
                                atol=1e-3),\
                                f"The amount of pruned neuron in g_a is not {amounts[index]}"
        
        delete_mask(net.g_a,parameters_to_prune["g_a"])
        delete_mask(net.g_s,parameters_to_prune["g_s"])

    print("!!!SUCCESS!!!")

if __name__ =="__main__":
    test_mask_generation_MeanScaleHyperprior()