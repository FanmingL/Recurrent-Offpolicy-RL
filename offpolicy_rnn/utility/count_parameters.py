import torch
import numpy as np
def count_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.

    Args:
    model (nn.Module): The PyTorch model to count parameters for.

    Returns:
    int: The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
