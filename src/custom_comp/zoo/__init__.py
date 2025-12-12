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


# from custom_compress .models import SymmetricalTransFormer, WACNN
from custom_comp.models import SymmetricalTransFormer, WACNN, Cheng2020Attention_BA2

from .pretrained import load_pretrained as load_state_dict
from compressai.models.waseda import Cheng2020Attention
from compressai.models.google import MeanScaleHyperprior
from compressai.zoo import cheng2020_attn,mbt2018_mean
# from compressai.zoo.pretrained import load_pretrained

from torch import load,zeros_like


def STF_loading(state_dict_path):
    state_dict = load(state_dict_path)
    state_dict = load_state_dict(state_dict)

    state_dict = state_dict['state_dict']
    net = SymmetricalTransFormer()
    
    # if 'entropy_bottleneck._quantized_cdf' in state_dict:
    #     state_dict['entropy_bottleneck._quantized_cdf'] = zeros_like(net.entropy_bottleneck._quantized_cdf)
    # if 'entropy_bottleneck._cdf_length' in state_dict:
    #     state_dict['entropy_bottleneck._cdf_length'] = zeros_like(net.entropy_bottleneck._cdf_length)
    # if 'gaussian_conditional._quantized_cdf' in state_dict:
    #     state_dict['gaussian_conditional._quantized_cdf'] = zeros_like(net.gaussian_conditional._quantized_cdf)
    # if 'gaussian_conditional._cdf_length' in state_dict:
    #     state_dict['gaussian_conditional._cdf_length'] = zeros_like(net.gaussian_conditional._cdf_length)

    net.load_state_dict(state_dict)
    # net.entropy_bottleneck.update(force=True)
    #net.gaussian_conditional.update()

    net.g_a = net.layers
    net.g_s = net.syn_layers
    return net

def WACNN_loading(state_dict_path):

    def clean_cnn_checkpoint(state_dict):
        '''
            All the key in the checkpoint have a prefix "module."
            We need to remove it in order to match the keys of the model
        '''
        new_state_dict = {}
        for k,v in state_dict.items():
            new_key = k.replace("module.","")
            new_state_dict[new_key] = v
        return new_state_dict

    state_dict = load(state_dict_path)
    state_dict["state_dict"] = clean_cnn_checkpoint( state_dict["state_dict"])
    state_dict = load_state_dict(state_dict)
    state_dict = state_dict['state_dict']

    net = WACNN()
    net.load_state_dict(state_dict)
    
    return net

models = {
    'stf': lambda:STF_loading("/home/ids/flauron-23/MagV/pretrained_models/stf_0483.pth.tar"),
    'cnn': lambda:WACNN_loading("/home/ids/flauron-23/MagV/pretrained_models/cnn_025_best.pth.tar"),
    'cheng': lambda:cheng2020_attn(quality=6,pretrained=True),
    'chengBA2': Cheng2020Attention_BA2,
    'msh': lambda:mbt2018_mean(quality=6,pretrained=True), # TODO same as cheng2020
}
