import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from compressai.zoo import cheng2020_attn


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from custom_comp.models import Cheng2020Attention_BA2
from custom_comp.models.chengBA2 import *

# class thresholdFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         return (input > 0.0).float()

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output
    

# class BudgetAwareAdapter(nn.Module):
#     def __init__(self,in_channels):
#         super(BudgetAwareAdapter, self).__init__()
#         self.switch = Parameter(torch.ones(in_channels))


#     def forward(self, x):
#             switch = thresholdFunction.apply(self.switch)
#             x = x*switch.view(1,self.switch.size(0),1,1)
#             return x
    



# class TinyNet(nn.Module):
#     def __init__(self):
#         super(TinyNet,self).__init__()
#         self.conv1 = nn.Conv2d(2, 4, 3, padding=1)
#         self.adapter1 = BudgetAwareAdapter(4)
#         self.conv2 = nn.Conv2d(4, 2, 3, padding=1)
#         self.adapter2 = BudgetAwareAdapter(2)

#     def forward(self,x):
#         x = self.conv1(x)    
    # def set_index(self,index:int):
    #     self.index=index
#         x = self.adapter1(x)
#         x = self.conv2(x)
#         x = self.adapter2(x)
#         return x
    

# m = TinyNet()
# loss_fn = nn.MSELoss()
# optimizer = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)

# for epoch in range(50):
#     x = torch.randn(1,2,8,8)
#     y = m(x)
#     loss = 10*loss_fn(y, torch.rand(1,2,8,8))    
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()

# # x = torch.randn(1,2,8,8)
# # y = m(x)
# # print(m.adapter1.switch)
# # print(m.adapter2.switch)
# print(y)

# m=cheng2020_attn(quality=5, pretrained=True)
# print(m)
# for name, module in m.named_modules():
#     if isinstance(module,nn.Conv2d):
#         print("name:",name,"module:",module)

# m=nn.Conv2d(3,6,3,padding=1)
# print(type(m))
# m.add_module("adapter",BudgetAwareAdapter(m.out_channels))
# m.adapter.switch.data[:]=0.0

# x=torch.randn(1,3,8,8)
# y=m(x)
# print(y)

# create the work model
n = Cheng2020Attention_BA2()

#load pretrained weight of the model cheng2020_atttn quality = 6
n = load_model(n)
#Load the original model via cheng2020_attn
# m = cheng2020_attn(quality=6, pretrained=True)

# #Copy the weight from the original model to the new model, layer by layer
# for (name1, module1) in n.named_modules():

#     if not isinstance(module1, nn.Conv2d):
#         continue  

#     print("name1:",name1,"module1:",module1)
#     module2 = dict(m.named_modules()).get(name1)

#     if not isinstance(module2, nn.ReLU) and module2 is not None:
#         print("module2:",module2)
#         assert torch.allclose(module1.weight, module2.weight), f"Type mismatch: {name1}"


# print("All convolution  weights are the same, copy the weight and bias to the new model")

for (_,module) in n.named_modules():
    if isinstance(module,BudgetAwareAdapter):

        #module.switch.data.zero_()
        with torch.no_grad():
            module.switch[0,:].fill_(0.0)
        print(module.switch)


n.set_index(1)
x=torch.randn(1,3,256,256)
y=n(x)

print(y["x_hat"])