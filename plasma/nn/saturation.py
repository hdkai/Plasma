# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import Tensor
from torch.nn import Module

class SaturationLoss (Module): # INCOMPLETE
    """
    Saturation loss.
    """

    def __init__ (self):
        super(SaturationLoss, self).__init__()

    def forward (self, input: Tensor, target: Tensor):
        pass
