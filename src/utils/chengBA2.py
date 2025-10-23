import torch
import torch.nn as nn
from torch.nn import Parameter
import sys

from compressai.layers import ResidualBlock, AttentionBlock, ResidualBlockWithStride, ResidualBlockUpsample
from compressai.models import Cheng2020Attention

from functools import partial

class thresholdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class ConvAdapter(torch.nn.Module):
    def __init__(self, in_channels, out_channels, rank, alpha, 
                kernel_size=3, padding=1, stride=1, groups=1, dilation=1, act_layer=None):
        super().__init__()

        if act_layer is None:
            act_layer = nn.Identity()

        # depth-wise conv
        self.conv1 = nn.Conv2d(in_channels, rank, kernel_size=kernel_size, stride=stride, groups=groups, padding=padding, dilation=dilation)

        self.act = act_layer

        # poise-wise conv
        self.conv2 = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1)
        self.alpha = alpha

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = out * self.alpha

        return out
    

class ConvUpscaleAdapter(torch.nn.Module):
    def __init__(self, in_channels, out_channels, rank, alpha, 
                kernel_size=3, padding=1, stride=1, groups=1, dilation=1, act_layer=None):
        super().__init__()

        if act_layer is None:
            act_layer = nn.Identity()

        # depth-wise conv
        self.conv1 = nn.Conv2d(in_channels, rank * 4, kernel_size=kernel_size, stride=stride, groups=groups, padding=padding, dilation=dilation)

        self.act = act_layer
        self.upscale = nn.PixelShuffle(2)

        # poise-wise conv
        self.conv2 = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1)
        self.alpha = alpha

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.upscale(out)
        out = self.conv2(out)
        out = out * self.alpha

        return out


class SubpelConvWithAdapterSwitch(torch.nn.Module):
    def __init__(self, subpel_conv, rank, alpha, activation = nn.Identity, divide_rank = False) -> None:
        super().__init__()
        assert isinstance(subpel_conv[0], nn.Conv2d)

        if divide_rank:
            rank = subpel_conv[0].in_channels // rank
        self.subpel_conv = subpel_conv
        self.r = subpel_conv[1].upscale_factor
        self.rank = rank
        self.alpha = alpha

        self.adapter = ConvUpscaleAdapter(
            in_channels=subpel_conv[0].in_channels, 
            out_channels=subpel_conv[0].out_channels // 4,
            rank= rank,
            alpha=alpha,
            kernel_size=subpel_conv[0].kernel_size,
            padding=subpel_conv[0].padding,
            stride=subpel_conv[0].stride,
            groups= rank,
            dilation=subpel_conv[0].dilation,
            act_layer=activation)
        
        # Switchable channels
        self.switch = Parameter(torch.ones(6,self.subpel_conv[0].out_channels))
        self.index = 0 

    def forward(self, x):
        switch = thresholdFunction.apply(self.switch[self.index,:])
        out = nn.functional.conv2d(x,
                                  self.subpel_conv[0].weight* switch.view(-1,1,1,1),
                                  self.subpel_conv[0].bias,
                                  stride=self.subpel_conv[0].stride,
                                  padding=self.subpel_conv[0].padding,
                                  dilation=self.subpel_conv[0].dilation,
                                  groups=self.subpel_conv[0].groups)
        
        out = nn.PixelShuffle(self.r)(out)

        lora = self.adapter(x)

        return out + lora
    
    def set_index(self,index):
        self.index=index

    def extra_repr(self):
        return (
            f"in_channels={self.subpel_conv[0].in_channels}, "
            f"out_channels={self.subpel_conv[0].out_channels}, "
            f"kernel_size={self.subpel_conv[0].kernel_size}, "
            f"stride={self.subpel_conv[0].stride}, "
            f"padding={self.subpel_conv[0].padding}, "
            f"dilation={self.subpel_conv[0].dilation}, "
            f"groups={self.subpel_conv[0].groups}, "
            f"switch_shape={self.switch.shape},"
            f"rank={self.rank},"
            f"alpha={self.alpha}"
        )
    

class ConvWithAdapterSwitch(torch.nn.Module):
    def __init__(self, conv:nn.Conv2d, rank, alpha, activation = nn.Identity, divide_rank = False):
        super().__init__()
        self.conv = conv

        if divide_rank:
            rank = conv.in_channels // rank
        
        rank = max(rank, 1)
        group_rank = rank

        if conv.in_channels % group_rank != 0:
            group_rank = 1

        # if rank > conv.in_channels:
        #     group_rank = 1
        
        # print(conv.in_channels, group_rank)

        self.rank = rank
        self.alpha = alpha

        self.adapter = ConvAdapter(
            in_channels=conv.in_channels, 
            out_channels=conv.out_channels,
            rank= rank,
            alpha=alpha,
            kernel_size=conv.kernel_size,
            padding=conv.padding,
            stride=conv.stride,
            groups=group_rank, 
            dilation=conv.dilation,
            act_layer=activation)
        
        # Switchable channels
        self.switch = Parameter(torch.ones(6,conv.out_channels))
        self.index = 0     
        
    def forward(self, x):
        switch = thresholdFunction.apply(self.switch[self.index,:])
        out=nn.functional.conv2d(x,
                                  self.conv.weight* switch.view(-1,1,1,1),
                                  self.conv.bias,
                                  stride=self.conv.stride,
                                  padding=self.conv.padding,
                                  dilation=self.conv.dilation,
                                  groups=self.conv.groups)
        lora=self.adapter(x)
        return out + lora
    
    def set_index(self,index):
        self.index=index

    def extra_repr(self):
        return (
            f"in_channels={self.conv.in_channels}, "
            f"out_channels={self.conv.out_channels}, "
            f"kernel_size={self.conv.kernel_size}, "
            f"stride={self.conv.stride}, "
            f"padding={self.conv.padding}, "
            f"dilation={self.conv.dilation}, "
            f"groups={self.conv.groups}, "
            f"switch_shape={self.switch.shape},"
            f"rank={self.rank},"
            f"alpha={self.alpha}"
        )
    

def get_Cheng2020Attention_with_conv_switch(model:Cheng2020Attention,rank=4,alpha=1.0,act=None,divide_rank = False):

    assign_conv_adapter = partial(ConvWithAdapterSwitch, rank=rank, alpha=alpha, activation = act, divide_rank = divide_rank)
    assign_subpel_conv_adapter = partial(SubpelConvWithAdapterSwitch, rank=rank, alpha=alpha, activation = act, divide_rank = divide_rank)

    for _,layer in enumerate(model.g_a): # encoder
        if isinstance(layer, ResidualBlock):
                layer.conv1 = assign_conv_adapter(layer.conv1)
                layer.conv2 = assign_conv_adapter(layer.conv2)

        elif isinstance(layer, ResidualBlockWithStride):
                layer.conv1 = assign_conv_adapter(layer.conv1)
                layer.conv2 = assign_conv_adapter(layer.conv2)

        elif isinstance(layer, AttentionBlock):
                for units in layer.conv_a:
                    units.conv[2] = assign_conv_adapter(units.conv[2])
            
                for units in layer.conv_b:
                    if not isinstance(units, nn.Conv2d): # To avoid last 1x1 conv
                        units.conv[2] = assign_conv_adapter(units.conv[2])

    for _,layer in enumerate(model.g_s): # decoder
        if isinstance(layer, ResidualBlock) :
                layer.conv1 = assign_conv_adapter(layer.conv1)
                layer.conv2 = assign_conv_adapter(layer.conv2)

        elif isinstance(layer, ResidualBlockUpsample):
                layer.subpel_conv = assign_subpel_conv_adapter(layer.subpel_conv)
                layer.conv = assign_conv_adapter(layer.conv)

        elif isinstance(layer, AttentionBlock):
                for units in layer.conv_a:
                    units.conv[2] = assign_conv_adapter(units.conv[2])
                for units in layer.conv_b:
                    if not isinstance(units, nn.Conv2d): # To avoid last 1x1 conv
                        units.conv[2] = assign_conv_adapter(units.conv[2])

def set_cheng2020Attention_index(model:Cheng2020Attention,index:int):

    for _,module  in model.g_a.named_modules(): # encoder
        if isinstance(module,ConvWithAdapterSwitch):
                module.set_index(index) 

    for _,module in model.g_s.named_modules(): # decoder
        if isinstance(module,ConvWithAdapterSwitch) or isinstance(module,SubpelConvWithAdapterSwitch):
                module.set_index(index)

def frozen_cheng2020Attention(model:Cheng2020Attention):
    
    # Freeze all parameters 
    for _,param in model.named_parameters():
        param.requires_grad = False

    # Unfreeze the convSwitch parameters
    for module in model.modules():
        if isinstance(module, ConvWithAdapterSwitch) or isinstance(module, SubpelConvWithAdapterSwitch):
            for param in module.parameters():
                param.requires_grad = True

