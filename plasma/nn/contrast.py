# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import Tensor
from torch.nn import Module

class ContrastLoss (Module):
    """
    Contrast loss, from Mertens et al.
    """

    def __init__ (self):
        super(ContrastLoss, self).__init__()

    def forward (self, input: Tensor, target: Tensor):
        pass