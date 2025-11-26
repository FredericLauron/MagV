from utils.masks import delete_mask, save_mask, generate_mask_from_structured_fisher, \
                        generate_mask_from_structured, apply_saved_mask

from utils.engine import test_epoch,train_one_epoch, compress_one_epoch #, AverageMeter,pad,crop
from evaluate import plot_rate_distorsion

from compressai.zoo import cheng2020_attn


from experiment.context import Context
import os
import json
class Kodak:
    def __init__(self,args, context: Context):
        self.args = args
        self.ctx = context

    
        