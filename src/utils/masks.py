import torch
from torch.nn.utils import prune
from collections import defaultdict

def delete_mask(model,parameters_to_prune):
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


def save_mask(model):
    """ 
    Saves the pruning masks from the model into a dictionary. 
    Arguments:
        model: The model from which to save the pruning masks.
    Returns:
        mask_dict: A dictionary containing the names of the modules and their corresponding pruning masks.
    Raises:
        ValueError: If no masks are found in the model.
    """
    mask_dict = {}

    #Look for any pruned module in the model and save the mask into mask_dict
    for name, module in model.named_modules():
        if hasattr(module, 'weight_orig') and hasattr(module, 'weight_mask'):
            mask_dict[name] = module.weight_mask.detach().clone()

    # if mask_dict is empty, raise an error
    if not mask_dict:
        raise ValueError("No masks found in the model. Ensure that pruning has been applied.")
    
    return mask_dict

def apply_saved_mask(model, mask_dict):
    """
    Applies saved pruning masks to the model.
    Arguments:
        model: The model to which the masks will be applied.
        mask_dict: A dictionary containing the names of the modules and their corresponding pruning masks.  
    Raises:
        AssertionError: If the mask_dict is empty or None.
    """
    
    assert mask_dict is not None and len(mask_dict) > 0, "Mask dictionary is empty or None."

    for name, module in model.named_modules():
        if name in mask_dict:
            # Ensure the module has 'weight' to prune
            if hasattr(module, 'weight'):
                prune.custom_from_mask(module, name='weight', mask=mask_dict[name])

############################################################################################################################################
############################################################UNSTRUCTURED PRUNING############################################################
############################################################################################################################################
def generate_mask_from_unstructured(model,amounts:list):
    """ 
    Generates pruning masks for the model based on the specified amounts.
    Arguments:
        model: The model for which to generate pruning masks.
        amounts: A list of amounts in % specifying the fraction of weights to prune.
        Returns:    
            out_all_mask: A list of dictionaries containing the pruning masks for each amount.
            parameters_to_prune: A list of tuples containing the modules and their parameters to prune.
    Raises:
        AssertionError: If the model is None or amounts is empty.
    """
    # checks
    assert model is not None
    assert amounts is not None and len(amounts) > 0

    #register all the parameters for the model that are available for pruning
    parameters_to_prune = [(module, "weight") for module in filter(lambda m: type(m) in [torch.nn.Conv2d, torch.nn.Linear], model.modules())]

    out_all_mask = []

    for index in amounts:

        # generate the pruning masks
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured,amount=index)
        
        # Save pruning masks
        out_all_mask.append(save_mask(model))

        #cleaning the pruning masks from the model
        delete_mask(model, parameters_to_prune)

    return out_all_mask ,parameters_to_prune

############################################################################################################################################
############################################################END UNSTRUCTURED PRUNING########################################################
############################################################################################################################################

############################################################################################################################################
############################################################STRUCTURED PRUNING##############################################################
############################################################################################################################################

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
    num_to_prune = int(amount * total_neurons)

    neurons_to_prune = global_list[-num_to_prune:]
    return neurons_to_prune

def group_by_module(neurons_to_prune):
    module_to_indices = defaultdict(list)
    for module, idx, _ in neurons_to_prune:
        module_to_indices[module].append(idx)
    return module_to_indices

def build_and_put_mask_on_module(module_to_indices):
    for module, indices in module_to_indices.items():

        mask = torch.ones_like(module.weight.data)
        mask[indices] = 0  # zero out selected neurons
        prune.custom_from_mask(module, name="weight", mask=mask)

def global_structured(parameters_to_prune,amount):
    
    #Compute neurons norms et recover the list of all neurons 
    #Structure of the list : (module, neuron_index, norm_value)
    global_list= get_global_list(compute_neuron_norm(parameters_to_prune))

    #Get the list of neurons to prune
    #Group them by module
    #Build the mask and put it on the module
    build_and_put_mask_on_module(group_by_module(get_neuron_to_prune(global_list, amount)))


def generate_mask_from_structured(model,amounts:list):
    """ 
    Generates pruning masks for the model based on the specified amounts.
    Arguments:
        model: The model for which to generate pruning masks.
        amounts: A list of amounts in % specifying the fraction of weights to prune.
        Returns:    
            out_all_mask: A list of dictionaries containing the pruning masks for each amount.
            parameters_to_prune: A list of tuples containing the modules and their parameters to prune.
    Raises:
        AssertionError: If the model is None or amounts is empty.
    """
    # checks
    assert model is not None
    assert amounts is not None and len(amounts) > 0

    #register all the parameters for the model that are available for pruning
    parameters_to_prune = [(module, "weight") for module in filter(lambda m: type(m) in [torch.nn.Conv2d, torch.nn.Linear], model.modules())]

    out_all_mask = []

    for index in amounts:

        # generate the pruning masks
        #prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured,amount=index)
        global_structured(parameters_to_prune, index)
        
        # Save pruning masks
        out_all_mask.append(save_mask(model))

        #cleaning the pruning masks from the model
        delete_mask(model, parameters_to_prune)

    return out_all_mask ,parameters_to_prune

############################################################################################################################################
############################################################END STRUCTURED PRUNING##########################################################
############################################################################################################################################