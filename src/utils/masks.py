import torch
from torch.nn.utils import prune


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