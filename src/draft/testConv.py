import torch
import torch.nn as nn
from torch.nn.utils import prune
from collections import defaultdict
from math import ceil,floor

def check_neuron_sparsity(parameters_to_prune):
    total_neurons = 0
    zeroed_neurons = 0
    for module, _ in parameters_to_prune:
        W = module.weight.data
        if isinstance(module, nn.Conv2d):
            for i in range(W.size(0)):  # output channels
                total_neurons += 1
                if torch.all(W[i] == 0):
                    zeroed_neurons += 1
        elif isinstance(module, nn.Linear):
            for i in range(W.size(0)):  # rows
                total_neurons += 1
                if torch.all(W[i] == 0):
                    zeroed_neurons += 1

    print(f"Neuron sparsity: {zeroed_neurons/total_neurons:.2%}")

class TinyConvNet(nn.Module):
    def __init__(self):
        super(TinyConvNet,self).__init__()
        self.conv1 = nn.Conv2d(2, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 2, 3, padding=1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x    
    

m = TinyConvNet()
m.conv1.weight.data[0] = 5.0
m.conv1.weight.data[1] = 4.0
m.conv1.weight.data[2] = 3.0
m.conv1.weight.data[3] = 2.0

m.conv2.weight.data[0] = 1.0
m.conv2.weight.data[1] = 0.0

amount=0.5
parameters_to_prune = [(module,"weight") for module in filter(lambda m: type(m) in [nn.Conv2d, nn.Linear], m.modules())]

for module, _ in parameters_to_prune:
    prune.ln_structured(module,name="weight",amount=amount,n=2,dim=0)

print(m.conv1.weight)
print(m.conv2.weight)

check_neuron_sparsity(parameters_to_prune)

def compute_neuron_norm(parameters_to_prune):
    norms = {}
    for module, _ in parameters_to_prune:
        W = module.weight.data

        if isinstance(module, torch.nn.Conv2d):
            # Norm per filter (output channel)
            neuron_norms = torch.norm(W.view(W.size(0), -1), p=2, dim=1)
  

        elif isinstance(module, torch.nn.Linear):
            # Norm per neuron (row of weight matrix)
            neuron_norms = torch.norm(W, p=2, dim=1)

        norms[module] = neuron_norms

    # Normalization of the norms
    norms = {k: v/len(norms) for k, v in norms.items()}
    
    return norms


def get_global_list(norms_dict):
    global_list = []  # list of (module, neuron_index, norm_value)

    for module, neuron_norms in norms_dict.items():
        for idx, norm_value in enumerate(neuron_norms):
            global_list.append((module, idx, norm_value.item()))
    
    return global_list


def get_neuron_to_prune(global_list, amount):
    global_list.sort(key=lambda x: x[2], reverse=True)
    total_neurons = len(global_list)
    num_to_prune = floor(amount * total_neurons)

    neurons_to_prune = global_list[-num_to_prune:]
    return neurons_to_prune

def group_by_module(neurons_to_prune):
    module_to_indices = defaultdict(list)
    for module, idx, _ in neurons_to_prune:
        module_to_indices[module].append(idx)
    return module_to_indices

def build_and_put_mask_on_module(module_to_indices):
    for module, indices in module_to_indices.items():

        amount = len(indices) / module.weight.data.size(0)
        print(f"Pruning {amount:.2%} of neurons in module {module}")

        prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)


        # mask = torch.ones_like(module.weight.data)
        # mask = torch.ones_like(module.weight.data)
        # mask[indices] = 0  # zero out selected neurons
        # prune.custom_from_mask(module, name="weight", mask=mask)
        # module to indices and then estimate the percentage of pruned neurons in the layer 

def global_structured(parameters_to_prune,amount):
    
    #Compute neurons norms et recover the list of all neurons 
    #Structure of the list : (module, neuron_index, norm_value)
    global_list= get_global_list(compute_neuron_norm(parameters_to_prune))

    #Get the list of neurons to prune
    #Group them by module
    #Build the mask and put it on the module
    build_and_put_mask_on_module(group_by_module(get_neuron_to_prune(global_list, amount)))

def delete_mask(parameters_to_prune):
    """ 
    Deletes the pruning mask from the model and preserves the pruned weights. 
    Arguments:
        model: The model from which to delete the pruning mask.
        parameters_to_prune: A list of tuples containing the modules and their parameters to prune.
    Raises:
        AssertionError: If no parameters to prune are provided.        
    """        
    assert parameters_to_prune is not None and len(parameters_to_prune) > 0, "No parameters to prune provided."

    #Look for any pruned module in the model and modify the mask
    # to preserve all the weiths of the module
    # and remove the pruning from the module
    for module, _ in parameters_to_prune:
        if hasattr(module, 'weight_orig'):
            with torch.no_grad():
                # create a mask full of ones to preserved the pruned weights
                module.weight_mask = torch.ones_like(module.weight_mask)
            prune.remove(module, 'weight')


delete_mask(parameters_to_prune)

global_structured(parameters_to_prune,amount)

print(m.conv1.weight)
print(m.conv2.weight)

check_neuron_sparsity(parameters_to_prune)

delete_mask(parameters_to_prune)
for module, _ in parameters_to_prune:

    if isinstance(module, nn.Conv2d):
        print(torch.norm(module.weight.data.view(module.weight.data.size(0), -1), p=2, dim=1))

    elif isinstance(module, nn.Linear):
        print(torch.norm(module.weight.data, p=2, dim=1))