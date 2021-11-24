import torch

def global_magnitude_pruning(parameters, sparsity):
    #paramaters = dict
    #sparsity: frac of weights to prune
    cutoff = torch.cat([tensor.view(-1) for tensor in parameters.values()]).detach().abs().quantile(sparsity)
    mask = {}
    for key, value in parameters.items():
        mask[key] = (value.abs() >= cutoff).float()
    return mask    

def default_pruning(parameters):
    return {key: torch.ones(value.shape) for key, value in parameters.items()}