import math
import torch
import torch.nn as nn

from compressai.ans import BufferedRansEncoder, RansDecoder

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN 
from compressai.models.utils import conv, deconv, update_registered_buffers
from compressai.ops import quantize_ste as ste_round
from compressai.layers import conv3x3, subpel_conv3x3
from custom_comp.layers import Win_noShift_Attention
from compressai.models.base import CompressionModel

from .hyp.cnn import WACNNHyp


class WACNN(WACNNHyp):
    """CNN based model"""

    def __init__(self, N=192, M=320, **kwargs):
        super().__init__(**kwargs)

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
            Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
        )
        self.g_s = nn.Sequential(
            Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        