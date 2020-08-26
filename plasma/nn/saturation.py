# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import clamp, norm, Tensor
from torch.nn import Module

from ..conversion import rgb_to_yuv

class SaturationLoss (Module):
    """
    Saturation loss.
    """

    def __init__ (self):
        super(SaturationLoss, self).__init__()

    def forward (self, input: Tensor, target: Tensor):
        input_uv = rgb_to_yuv(input)[:,1:,:,:]
        target_uv = rgb_to_yuv(target)[:,1:,:,:]
        input_sat = norm(input_uv, dim=1)
        target_sat = norm(target_uv, dim=1)
        delta = clamp(target_sat - input_sat, min=0.)
        loss = delta.sum() / delta.nelement()
        return loss