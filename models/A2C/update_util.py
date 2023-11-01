import numpy as np
import torch
import torch.autograd as Variable

def soft_update(target, source, tau):
    """
    Update target network(y) using source network(x) by:
    y' = TAU*y + (1-TAU)*x
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * tau + param.data * (1.0 - tau)
        )

def hard_update(target, source):
    """
    Update target network by replacing it with source network
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)