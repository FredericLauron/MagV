# import sys
# sys.path.append("/home/ids/flauron-23/STF-QVRF")
# from compressai.entropy_models import GaussianConditional
# from compressai.models.priors import CompressionModel

#from Cheng2020Attention import Cheng2020Attention

from opt import parse_args
from utils import seed_all
# import os
# import wandb

from experiment import Experiment
from utils import save_checkpoint

# from custom_comp.zoo import models

from custom_comp.models import SymmetricalTransFormer, WACNN,QVRFCheng,stfQVRF,GainedMSHyperprior

from custom_comp.zoo.pretrained import load_pretrained as load_state_dict
from compressai.models.waseda import Cheng2020Attention
from compressai.models.google import MeanScaleHyperprior
from compressai.zoo import cheng2020_attn,mbt2018_mean
# from compressai.zoo.pretrained import load_pretrained

from torch import load,zeros_like
from typing import Union

import torch

def rewire_g_a_s(model:Union[SymmetricalTransFormer,stfQVRF]):
    model.g_a = model.layers
    model.g_s = model.syn_layers

def model_loading (model_class,state_dict_path,preprocess=None,postprocess=None,**kwargs):
    state_dict = load(state_dict_path)

    # Only deal with state_dictionnary
    if "state_dict" in state_dict.keys():
        state_dict = state_dict['state_dict']
  
    if preprocess is not None:
        state_dict=preprocess(state_dict,**kwargs)

    net = model_class() 

    net.load_state_dict(state_dict)

    if postprocess is not None: 
        postprocess(net,**kwargs) 
    
    return state_dict,net


def assert_model_loaded_correctly(model, state_dict, rtol=1e-5, atol=1e-8):
    """
    Asserts that all weights in the model match the provided state_dict.
    """
    model_params = dict(model.named_parameters())
    for key, tensor in state_dict.items():
        # Skip buffers like running_mean/running_var if needed
        if key in model_params:
            if not torch.allclose(model_params[key].data, tensor.data, rtol=rtol, atol=atol):
                raise ValueError(f"Mismatch in parameter: {key}")
        else:
            # Some keys might be buffers, not parameters
            if key in dict(model.named_buffers()):
                buf = dict(model.named_buffers())[key]
                if not torch.allclose(buf.data, tensor.data, rtol=rtol, atol=atol):
                    raise ValueError(f"Mismatch in buffer: {key}")
            else:
                raise KeyError(f"Key {key} not found in model parameters or buffers")
    print("All weights match the state_dict!")

# Example usage:
# state_dict = torch.load("/home/ids/flauron-23/QRAF/Cheng2020VR.pth.tar")
# state_dict = torch.load("/home/ids/flauron-23/QRAF/Cheng2020VR.pth.tar")
# if "state_dict" in state_dict:
#     state_dict = state_dict["state_dict"]

# model = model_loading(QVREFCheng, "/home/ids/flauron-23/QRAF/Cheng2020VR.pth.tar")
#state_dict,model = model_loading(QVREFCheng,"/home/ids/flauron-23/QRAF/Cheng2020VR.pth.tar")
# state_dict,model = model_loading(WACNN,"/home/ids/flauron-23/MagV/pretrained_models/cnn_025_best.pth.tar",load_state_dict)
state_dict,model = model_loading(Cheng2020Attention,"/home/ids/flauron-23/MagV/data/magv_04_cheng_unstructured/models/magv_04_cheng_unstructured_checkpoint_best.pth.tar")


device = next(v.device for v in state_dict.values() if torch.is_tensor(v))
model=model.to(device)

assert_model_loaded_correctly(model, state_dict)


models = {
    'stf': lambda:model_loading(SymmetricalTransFormer,"/home/ids/flauron-23/MagV/pretrained_models/stf_0483.pth.tar",postprocess=rewire_g_a_s)[1],
    'cnn': lambda:model_loading(WACNN,"/home/ids/flauron-23/MagV/pretrained_models/cnn_025_best.pth.tar",load_state_dict)[1],
    'qvref': lambda:model_loading(QVRFCheng,"/home/ids/flauron-23/QRAF/Cheng2020VR.pth.tar")[1],
    'stfqvref':lambda:model_loading(stfQVRF,"/home/ids/flauron-23/STF-QVRF/STFVRImageNetSTE.pth.tar",postprocess=rewire_g_a_s)[1],
    'cheng': lambda:cheng2020_attn(quality=6,pretrained=True),
    'msh': lambda:mbt2018_mean(quality=6,pretrained=True),
    'gained': lambda:model_loading(GainedMSHyperprior,"/home/ids/flauron-23/MagV/pretrained_models/checkpoint_gainmshp_epoch90.pth")[1],
    'ours': lambda:model_loading(Cheng2020Attention,"/home/ids/flauron-23/MagV/data/magv_04_cheng_unstructured/models/magv_04_cheng_unstructured_checkpoint_best.pth.tar")[1]
}

net = models["ours"]()
for name, module in net.named_modules():
    print(name)

# print(next(model.parameters()).device)

# m = models["stf"]()

# models = {
#     "qvref": 




#     'stf': lambda:STF_loading("/home/ids/flauron-23/MagV/pretrained_models/stf_0483.pth.tar"),
#     'cnn': lambda:WACNN_loading("/home/ids/flauron-23/MagV/pretrained_models/cnn_025_best.pth.tar"),
#     'cheng': lambda:cheng2020_attn(quality=6,pretrained=True),
#     'chengBA2': Cheng2020Attention_BA2,
#     'msh': lambda:mbt2018_mean(quality=6,pretrained=True), # TODO same as cheng2020
# }

# compare to qvref to ours

# load model

    #load dataset

        #bitrate


