# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import clamp, Tensor
from torch.nn import Module

from ..sampling import laplacian_of_gaussian_2d

class ContrastLoss (Module):
    """
    Contrast loss, from Mertens et al.
    """

    def __init__ (self):
        super(ContrastLoss, self).__init__()

    def forward (self, input: Tensor, target: Tensor):
        input_laplacian = laplacian_of_gaussian_2d(input)
        target_laplacian = laplacian_of_gaussian_2d(target)
        delta = clamp(target_laplacian - input_laplacian, min=0.)
        loss = delta.sum() / delta.nelement()
        return loss