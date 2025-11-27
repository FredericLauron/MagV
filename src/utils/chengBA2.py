import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import Parameter
import sys

from compressai.layers import ResidualBlock, AttentionBlock, ResidualBlockWithStride, ResidualBlockUpsample
from compressai.models import Cheng2020Attention, MeanScaleHyperprior

from functools import partial
import wandb

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

class DeconvAdapter(torch.nn.Module):
    def __init__(self, in_channels, out_channels, rank, alpha, 
                kernel_size=3, padding=1,output_padding=0,stride=1, groups=1, dilation=1, act_layer=None):
        super().__init__()

        if act_layer is None:
            act_layer = nn.Identity()

        # depth-wise conv
        self.deconv1 = nn.ConvTranspose2d(in_channels, 
                                          rank, 
                                          kernel_size=kernel_size, 
                                          stride=stride,
                                          output_padding=output_padding,
                                          groups=groups, 
                                          padding=padding,
                                          dilation=dilation)

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
    

class SubpelConvWithAdapterSwitch(torch.nn.Module):
    def __init__(self, subpel_conv, rank, alpha, activation = nn.Identity, divide_rank = False) -> None:
        super().__init__()
        assert isinstance(subpel_conv[0], nn.Conv2d)

        if divide_rank:
            rank = subpel_conv[0].in_channels // rank
        self.layer = subpel_conv
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
        self.switch = Parameter(torch.ones(5,self.layer[0].out_channels))

        # Non learnable switch for anchor model biterate
        #self.anchor_switch = torch.ones(1,self.subpel_conv[0].out_channels)
        self.register_buffer("switch_anchor", torch.ones(1,self.layer[0].out_channels))
        self.index = 0

    def forward(self, x):
        if self.index < 5:
            switch = thresholdFunction.apply(self.switch[self.index,:])
            out = nn.functional.conv2d(x,
                            self.layer[0].weight* switch.view(-1,1,1,1),
                            self.layer[0].bias,
                            stride=self.layer[0].stride,
                            padding=self.layer[0].padding,
                            dilation=self.layer[0].dilation,
                            groups=self.layer[0].groups)
            out = nn.PixelShuffle(self.r)(out)

            lora = self.adapter(x)
            return out + lora
        # No adapter for anchor model
        else:
            #switch = self.switch[5] / (self.switch[5] + 1e-8)
            switch = thresholdFunction.apply(self.switch_anchor)
            out = nn.functional.conv2d(x,
                            self.layer[0].weight* switch.view(-1,1,1,1),
                            self.layer[0].bias,
                            stride=self.layer[0].stride,
                            padding=self.layer[0].padding,
                            dilation=self.layer[0].dilation,
                            groups=self.layer[0].groups)
            out = nn.PixelShuffle(self.r)(out)
            # No adapter for anchor model
            return out

    def set_index(self,index):
        self.index=index

    def extra_repr(self):
        return (
            f"in_channels={self.layer[0].in_channels}, "
            f"out_channels={self.layer[0].out_channels}, "
            f"kernel_size={self.layer[0].kernel_size}, "
            f"stride={self.layer[0].stride}, "
            f"padding={self.layer[0].padding}, "
            f"dilation={self.layer[0].dilation}, "
            f"groups={self.layer[0].groups}, "
            f"switch_shape={self.switch.shape},"
            f"rank={self.rank},"
            f"alpha={self.alpha}"
        )
    
class ConvWithAdapterSwitch(torch.nn.Module):
    def __init__(self, conv:nn.Conv2d, rank, alpha, activation = nn.Identity, divide_rank = False):
        super().__init__()
        self.layer = conv

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
        self.switch = Parameter(torch.ones(5,conv.out_channels))

        # Non learnable switch for anchor model biterate
        #self.switch_anchor = torch.ones(1,conv.out_channels)
        self.register_buffer("switch_anchor", torch.ones(1, conv.out_channels))
        self.index = 0     
        
    def forward(self, x):
        if self.index < 5:
            switch = thresholdFunction.apply(self.switch[self.index,:])
            out=nn.functional.conv2d(x,
                                  self.layer.weight* switch.view(-1,1,1,1),
                                  self.layer.bias,
                                  stride=self.layer.stride,
                                  padding=self.layer.padding,
                                  dilation=self.layer.dilation,
                                  groups=self.layer.groups)
            lora=self.adapter(x)
            return out + lora
        
        # No adapter for anchor model
        else:
            #switch = self.switch[5] / (self.switch[5] + 1e-8)
            switch = thresholdFunction.apply(self.switch_anchor)
            out=nn.functional.conv2d(x,
                                  self.layer.weight* switch.view(-1,1,1,1),
                                  self.layer.bias,
                                  stride=self.layer.stride,
                                  padding=self.layer.padding,
                                  dilation=self.layer.dilation,
                                  groups=self.layer.groups)
            
            return out
        
    
    def set_index(self,index):
        self.index=index

    def extra_repr(self):
        return (
            f"in_channels={self.layer.in_channels}, "
            f"out_channels={self.layer.out_channels}, "
            f"kernel_size={self.layer.kernel_size}, "
            f"stride={self.layer.stride}, "
            f"padding={self.layer.padding}, "
            f"dilation={self.layer.dilation}, "
            f"groups={self.layer.groups}, "
            f"switch_shape={self.switch.shape},"
            f"rank={self.rank},"
            f"alpha={self.alpha}"
        )

class DeconvWithAdapterSwitch(torch.nn.Module):
    def __init__(self, deconv:nn.ConvTranspose2d, rank, alpha, activation = nn.Identity, divide_rank = False):
        super().__init__()
        self.layer = deconv

        if divide_rank:
            rank = deconv.in_channels // rank
        
        rank = max(rank, 1)
        group_rank = rank

        if deconv.in_channels % group_rank != 0:
            group_rank = 1

        # if rank > conv.in_channels:
        #     group_rank = 1
        
        # print(conv.in_channels, group_rank)

        self.rank = rank
        self.alpha = alpha

        self.adapter = DeconvAdapter(
            in_channels=deconv.in_channels, 
            out_channels=deconv.out_channels,
            rank= rank,
            alpha=alpha,
            kernel_size=deconv.kernel_size,
            padding=deconv.padding,
            output_padding=deconv.output_padding,
            stride=deconv.stride,
            groups=group_rank, 
            dilation=deconv.dilation,
            act_layer=activation)
        
        # Switchable channels
        self.switch = Parameter(torch.ones(5,deconv.out_channels))

        # Non learnable switch for anchor model biterate
        #self.switch_anchor = torch.ones(1,conv.out_channels)
        self.register_buffer("switch_anchor", torch.ones(1, deconv.out_channels))
        self.index = 0     
        
    def forward(self, x):
        if self.index < 5:
            switch = thresholdFunction.apply(self.switch[self.index,:])
            out=nn.functional.conv_transpose2d(x,
                                  self.layer.weight* switch.view(-1,1,1,1),
                                  self.layer.bias,
                                  stride=self.layer.stride,
                                  padding=self.layer.padding,
                                  output_padding=self.layer.output_padding,
                                  dilation=self.layer.dilation,
                                  groups=self.layer.groups)
            lora=self.adapter(x)
            return out + lora
        # No adapter for ancher model
        else:
            #switch = self.switch[5] / (self.switch[5] + 1e-8) # avoid division by zero and trick to have some learned parameters
            switch = thresholdFunction.apply(self.switch_anchor)
            out = nn.functional.conv_transpose2d(x,
                                  self.layer.weight* switch.view(-1,1,1,1),
                                  self.layer.bias,
                                  stride=self.layer.stride,
                                  padding=self.layer.padding,
                                  output_padding=self.layer.output_padding,
                                  dilation=self.layer.dilation,
                                  groups=self.layer.groups)

            return out 
        
    
    def set_index(self,index):
        self.index=index

    def extra_repr(self):
        return (
            f"in_channels={self.layer.in_channels}, "
            f"out_channels={self.layer.out_channels}, "
            f"kernel_size={self.layer.kernel_size}, "
            f"stride={self.layer.stride}, "
            f"padding={self.layer.padding}, "
            f"dilation={self.layer.dilation}, "
            f"groups={self.layer.groups}, "
            f"switch_shape={self.switch.shape},"
            f"rank={self.rank},"
            f"alpha={self.alpha}"
        )

def inject_adapter(model,rank=4,alpha=1.0,act=None,divide_rank = False):

    assign_conv_adapter = partial(ConvWithAdapterSwitch, rank=rank, alpha=alpha, activation = act, divide_rank = divide_rank)
    assign_subpel_conv_adapter = partial(SubpelConvWithAdapterSwitch, rank=rank, alpha=alpha, activation = act, divide_rank = divide_rank)
    assign_deconv_adapter = partial(DeconvWithAdapterSwitch, rank=rank, alpha=alpha, activation = act, divide_rank = divide_rank)

    for branch_name in ["g_a", "g_s"]: #encoder and decoder
            if not hasattr(model, branch_name):
                continue  # skip if model doesn't have this branch

            branch = getattr(model, branch_name)

            for i,layer in enumerate(branch): # iterate only on the top level of the sequential

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

                elif isinstance(layer, ResidualBlockUpsample):
                    layer.subpel_conv = assign_subpel_conv_adapter(layer.subpel_conv)
                    layer.conv = assign_conv_adapter(layer.conv)

                # branch[i] is used here because "layer" is a local temporary reference to the targeted layer
                # layer.conv = ... target the submodule of the current module
                # but with msh model doing layer = assign_deconv_adapter(layer) just modify the variable layer not the model's layer
                # elif isinstance(layer, nn.ConvTranspose2d):
                #     branch[i] = assign_deconv_adapter(layer)

                # elif isinstance(layer, nn.Conv2d):
                #     branch[i] = assign_conv_adapter(layer)

def set_index_switch(model,index:int):

    for module  in model.modules(): # encoder
        if isinstance(module,(ConvWithAdapterSwitch, SubpelConvWithAdapterSwitch, DeconvWithAdapterSwitch)):
                module.set_index(index) 

def freeze_model_with_switch(model):
    
    # Freeze all parameters 
    for name,module in model.named_children():
        if name in ['g_a','g_s']:
            for param in module.parameters():
                param.requires_grad = False

    # Unfreeze the convSwitch parameters
    for module in model.modules():
        if isinstance(module,(ConvWithAdapterSwitch, SubpelConvWithAdapterSwitch, DeconvWithAdapterSwitch)):
            for param in module.parameters():
                param.requires_grad = True

def measure_switch_sparcity(model,epoch):
    
    nb_zeros_per_switch = torch.zeros(5)
    nb_elements_per_switch = torch.zeros(5)

    for _,module in model.named_modules():
        if isinstance(module,(ConvWithAdapterSwitch, SubpelConvWithAdapterSwitch, DeconvWithAdapterSwitch)):
            switch = module.switch.detach().cpu()
            nb_switch = switch.size(0)

            for index in range(nb_switch):    # ignore the last switch (anchor model)
                s = switch[index]
                nb_zeros_per_switch[index] += (s <= 0.0).float().sum().item()
                nb_elements_per_switch[index] += s.size(0)
    
    sparsity = nb_zeros_per_switch / nb_elements_per_switch
          
    switch_labels =[f"Switch {i+1}" for i in range(len(sparsity))]

    plt.figure(figsize=(10, len(sparsity) * 0.4))
    im = plt.imshow(sparsity.unsqueeze(0).cpu().numpy(), cmap="viridis", aspect="auto", interpolation="nearest",vmin=0, vmax=1)

    plt.colorbar(im, label="Sparsity (fraction of zeros)")
    plt.yticks([])
    plt.xticks(range(len(switch_labels)), switch_labels)
    #plt.xlabel("Switch index")
    plt.ylabel("Layer name")
    plt.title("Switch sparsity per layer")

    for i in range(len(sparsity)):
        value = sparsity[i]
        plt.text(
            i,0, f"{value:.2f}",  # show 2 decimal places
            ha="center", va="center",
            color="white" if value > 0.5 else "black",  # contrast text color
            fontsize=8
        )



    plt.tight_layout()
    plt.savefig("switch_sparsity.png")
    #print("Saved figure as switch_sparsity.png")

    wandb.log({f"switch sparcity":epoch,
                 f"switch sparcity": wandb.Image(plt)}, step=epoch)


def measure_sparsity_induce_by_switch(model,epoch):
    
    # nb_zeros_per_switch = torch.zeros(5)
    # nb_elements_per_switch = torch.zeros(5)
    total_nb_pruned_param_per_switch = torch.zeros(5)
    total_nb_param = torch.tensor(0)
    

    for _,module in model.named_modules():

        # count the number of param in every layer of the model
        # Does the module has learnable weight param
        if hasattr(module, 'weight') and module.weight is not None:
            total_nb_param += module.weight.data.numel()

        if isinstance(module,(ConvWithAdapterSwitch, SubpelConvWithAdapterSwitch, DeconvWithAdapterSwitch)):

            # Each custom layer contains a self.conv,self.deconv or self.subpel_conv
            # Need to uniformize that into "self.layer"
            switch = module.switch.detach().cpu()
            

            if isinstance(module, SubpelConvWithAdapterSwitch):
                nb_param_per_neuron = module.layer[0].weight.data[0,:].numel()
            else:
                nb_param_per_neuron = module.layer.weight.data[0,:].numel()
                
            nb_switch = switch.size(0)

            for index in range(nb_switch): # ignore the last switch (anchor model)
                s = switch[index]

                # Count the number of "zeros" in the switch
                nb_zeros_per_switch= (s <= 0.0).float().sum().item()

                # Count the number of parameter pruned induced by the number of zeros
                total_nb_pruned_param_per_switch[index] += nb_zeros_per_switch * nb_param_per_neuron

    sparsity = total_nb_pruned_param_per_switch / total_nb_param
          
    switch_labels =[f"Switch {i+1}" for i in range(len(sparsity))]

    plt.figure(figsize=(10, len(sparsity) * 0.4))
    im = plt.imshow(sparsity.unsqueeze(0).cpu().numpy(), cmap="viridis", aspect="auto", interpolation="nearest",vmin=0, vmax=1)

    plt.colorbar(im, label="Sparsity (fraction of zeros)")
    plt.yticks([])
    plt.xticks(range(len(switch_labels)), switch_labels)
    #plt.xlabel("Switch index")
    plt.ylabel("Layer name")
    plt.title("Network sparsity  induded by the switch")

    for i in range(len(sparsity)):
        value = sparsity[i]
        plt.text(
            i,0, f"{value:.2f}",  # show 2 decimal places
            ha="center", va="center",
            color="white" if value > 0.5 else "black",  # contrast text color
            fontsize=8
        )



    plt.tight_layout()
    plt.savefig("network_sparsity.png")
    #print("Saved figure as network_sparsity.png")

    wandb.log({f"network sparcity":epoch,
                 f"network sparcity": wandb.Image(plt)}, step=epoch)