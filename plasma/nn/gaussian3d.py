# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import Tensor
from torch.nn import Module

class Gaussian3D (Module):
    """
    3D Gaussian smoothing layer.
    """

    def __init__ (self):
        super(Gaussian3D, self).__init__()

    def forward (self, input):
        pass
