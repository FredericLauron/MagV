import torch
from torch import nn
from torch.nn import Parameter

from compressai.layers import ResidualBlock, AttentionBlock, ResidualBlockWithStride, ResidualBlockUpsample
from compressai.models import Cheng2020Attention

class thresholdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
class convSwitch(nn.Conv2d):
    def __init__(self,conv:nn.Conv2d):
        super().__init__(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         conv.dilation,
                         conv.groups,
                         bias=(conv.bias is not None),
                         padding_mode=conv.padding_mode)

        if conv.weight is not None:
            self.weight = conv.weight
        if conv.bias is not None:
            self.bias = conv.bias

        self.in_channels=conv.in_channels
        self.out_channels=conv.out_channels
        self.kernel_size=conv.kernel_size
        self.stride=conv.stride
        self.padding=conv.padding
        self.dilation=conv.dilation
        self.groups=conv.groups

        self.switch = Parameter(torch.ones(6,conv.out_channels))
        self.index = 0     

    def forward(self, x):
            switch = thresholdFunction.apply(self.switch[self.index,:])
            out = nn.functional.conv2d(x, 
                                       self.weight * switch.view(-1,1,1,1),
                                       self.bias, 
                                       self.stride,
                                       self.padding, 
                                       self.dilation, 
                                       self.groups)   
            return out
    
    def set_index(self,index):
        self.index=index
    
    def extra_repr(self):
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"dilation={self.dilation}, "
            f"groups={self.groups}, "
            f"switch_shape={self.switch.shape}"
        )

class subpel_convSwitch(nn.Conv2d):
    def __init__(self,subpel_conv:nn.Sequential):
        super().__init__(subpel_conv[0].in_channels,
                         subpel_conv[0].out_channels,
                         subpel_conv[0].kernel_size,
                         subpel_conv[0].stride,
                         subpel_conv[0].padding,
                         subpel_conv[0].dilation,
                         subpel_conv[0].groups,
                         bias=(subpel_conv[0].bias is not None),
                         padding_mode=subpel_conv[0].padding_mode)

        conv = subpel_conv[0]
        if conv.weight is not None:
            self.weight = conv.weight
        if conv.bias is not None:
            self.bias = conv.bias

        self.in_channels=conv.in_channels
        self.out_channels=conv.out_channels
        self.kernel_size=conv.kernel_size
        self.stride=conv.stride
        self.padding=conv.padding
        self.dilation=conv.dilation
        self.groups=conv.groups

        self.r = subpel_conv[1].upscale_factor

        self.switch = Parameter(torch.ones(6,conv.out_channels))
        self.index = 0     

    def forward(self, x):
            switch = thresholdFunction.apply(self.switch[self.index,:])
            out = nn.functional.conv2d(x, 
                                       self.weight * switch.view(-1,1,1,1),
                                       self.bias,
                                       self.stride,
                                       self.padding,
                                       self.dilation, 
                                       self.groups)
               
            out = nn.PixelShuffle(self.r)(out)
            return out
    
    def set_index(self,index):
        self.index=index
    
    def extra_repr(self):
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"dilation={self.dilation}, "
            f"groups={self.groups}, "
            f"r={self.r}, "
            f"switch_shape={self.switch.shape}"
        )    

def get_Cheng2020Attention_with_conv_switch(model:Cheng2020Attention):

    for _,layer in enumerate(model.g_a): # encoder
        if isinstance(layer, ResidualBlock):
                layer.conv1 = convSwitch(layer.conv1)
                layer.conv2 = convSwitch(layer.conv2)

        elif isinstance(layer, ResidualBlockWithStride):
                layer.conv1 = convSwitch(layer.conv1)
                layer.conv2 = convSwitch(layer.conv2)

        elif isinstance(layer, AttentionBlock):
                for units in layer.conv_a:
                    units.conv[2] = convSwitch(units.conv[2])
            
                for units in layer.conv_b:
                    if not isinstance(units, nn.Conv2d): # To avoid last 1x1 conv
                        units.conv[2] = convSwitch(units.conv[2])

    for _,layer in enumerate(model.g_s): # decoder
        if isinstance(layer, ResidualBlock) :
                layer.conv1 = convSwitch(layer.conv1)
                layer.conv2 = convSwitch(layer.conv2)

        elif isinstance(layer, ResidualBlockUpsample):
                layer.subpel_conv = subpel_convSwitch(layer.subpel_conv)
                layer.conv = convSwitch(layer.conv)

        elif isinstance(layer, AttentionBlock):
                for units in layer.conv_a:
                    units.conv[2] = convSwitch(units.conv[2])
                for units in layer.conv_b:
                    if not isinstance(units, nn.Conv2d): # To avoid last 1x1 conv
                        units.conv[2] = convSwitch(units.conv[2])

def set_cheng2020Attention_index(model:Cheng2020Attention,index:int):

    for _,module  in model.g_a.named_modules(): # encoder
        if isinstance(module,convSwitch):
                module.set_index(index) 

    for _,module in model.g_s.named_modules(): # decoder
        if isinstance(module,convSwitch):
                module.set_index(index)


def frozen_cheng2020Attention(model:Cheng2020Attention):
    
    # Freeze all parameters 
    for _,param in model.named_parameters():
        param.requires_grad = False

    # Unfreeze the convSwitch parameters
    for module in model.modules():
        if isinstance(module, convSwitch) or isinstance(module, subpel_convSwitch):
            module.switch.requires_grad = True

