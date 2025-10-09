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
from compressai.zoo import cheng2020_attn
# from compressai.zoo.pretrained import load_pretrained

models = {
    'stf': SymmetricalTransFormer,
    'cnn': WACNN,
    'cheng': lambda:cheng2020_attn(quality=6,pretrained=True),
    'chengBA2': Cheng2020Attention_BA2,
}
