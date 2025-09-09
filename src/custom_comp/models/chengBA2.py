import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from torch.hub import load_state_dict_from_url

from compressai.layers.layers import conv3x3,conv1x1,subpel_conv3x3
from compressai.layers.gdn import GDN
from compressai.models.waseda import Cheng2020Anchor,Cheng2020Attention
from compressai.zoo.pretrained import load_pretrained

from itertools import chain



class thresholdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    

class BudgetAwareAdapter(nn.Module):
    def __init__(self,in_channels):
        super(BudgetAwareAdapter, self).__init__()
        self.switch = Parameter(torch.ones(6,in_channels))
        self.index = 0

    def forward(self, x):
            switch = thresholdFunction.apply(self.switch[self.index,:])
            print("switch",switch)
            print("index:", self.index)
            print(self.switch.shape)
            x = x*switch.view(1,switch.size(0),1,1)
            return x

    def set_index(self,index):
        self.index=index
    

class ResidualBlockWithStride_BA2(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.adapter1= BudgetAwareAdapter(out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.adapter2= BudgetAwareAdapter(out_ch)
        self.gdn = GDN(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.adapter1(out)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.adapter2(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        return out

class ResidualBlockUpsample_BA2(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2):
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.adapter1 = BudgetAwareAdapter(out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = conv3x3(out_ch, out_ch)
        self.adapter2 = BudgetAwareAdapter(out_ch)
        self.igdn = GDN(out_ch, inverse=True)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)
        self.adapter3 = BudgetAwareAdapter(out_ch)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(x)
        out += identity
        return out

class ResidualBlock_BA2(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.adapter1= BudgetAwareAdapter(out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.adapter2= BudgetAwareAdapter(out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.adapter1(out)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.adapter2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out
    
class AttentionBlock_BA2(nn.Module):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """

    def __init__(self, N: int):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    BudgetAwareAdapter(N // 2),
                    nn.ReLU(inplace=True),
                    conv3x3(N // 2, N // 2),
                    BudgetAwareAdapter(N // 2),
                    nn.ReLU(inplace=True),
                    conv1x1(N // 2, N),
                    BudgetAwareAdapter(N),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out
    

class Cheng2020Attention_BA2(Cheng2020Anchor):
    """Self-attention model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses self-attention, residual blocks with small convolutions (3x3 and 1x1),
    and sub-pixel convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, **kwargs):
        super().__init__(N=N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride_BA2(3, N, stride=2),
            ResidualBlock_BA2(N, N),
            ResidualBlockWithStride_BA2(N, N, stride=2),
            AttentionBlock_BA2(N),
            ResidualBlock_BA2(N, N),
            ResidualBlockWithStride_BA2(N, N, stride=2),
            ResidualBlock_BA2(N, N),
            conv3x3(N, N, stride=2),
            BudgetAwareAdapter(N),
            AttentionBlock_BA2(N),
        )

        self.g_s = nn.Sequential(
            AttentionBlock_BA2(N),
            ResidualBlock_BA2(N, N),
            ResidualBlockUpsample_BA2(N, N, 2),
            ResidualBlock_BA2(N, N),
            ResidualBlockUpsample_BA2(N, N, 2),
            AttentionBlock_BA2(N),
            ResidualBlock_BA2(N, N),
            ResidualBlockUpsample_BA2(N, N, 2),
            ResidualBlock_BA2(N, N),
            subpel_conv3x3(N, 3, 2),
            BudgetAwareAdapter(3),
        )

    #Call before forward to change index
    def set_index(self,index):
        for m in chain(self.g_a.modules(), self.g_s.modules()):
                if isinstance(m, BudgetAwareAdapter):
                    m.set_index(index)
        # for m in self.g_a.modules():
        #     if isinstance(m,BudgetAwareAdapter):
        #         m.set_index(index)
        
        # for m in self.g_s.modules():
        #     if isinstance(m,BudgetAwareAdapter):
        #         m.set_index(index)

    # Need to redefine this function to modify the call net.load_state_dict(state_dict)
    # to net.load_state_dict(state_dict,strict=False)
    # In this way, extra module, like adapters are ignored when loading the state_dict
    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.conv1.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict,strict=False)
        return net


def load_model(model):
    """
    Load model state dictionnary
    """
    root_url = "https://compressai.s3.amazonaws.com/models/v1"
    url = f"{root_url}/cheng2020_attn-mse-6-730501f2.pth.tar"


    state_dict = load_state_dict_from_url(url, progress=True)
    state_dict = load_pretrained(state_dict)
    model = model.from_state_dict(state_dict)

    return model

def copy_weights(model1: Cheng2020Attention_BA2 ,model2:Cheng2020Attention):
    """
    Copy the model state dictionnaty back to the original Cheng2020Attention model
    model1: cheng2020_BA2
    model2: Cheng2020Attention
    """
    for (name1, module1) in model1.named_modules():

        if not isinstance(module1, nn.Conv2d):
            continue  # skip non-weight modules

        
        module2 = dict(model2.named_modules()).get(name1)

        if not isinstance(module2, nn.ReLU) and module2 is not None:
            module2.weight.data = module1.weight.data
            if module1.bias is not None:
                module2.bias.data = module1.bias.data
    return model2
            


