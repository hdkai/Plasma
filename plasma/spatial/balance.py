# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import Tensor
from typing import Union

def color_balance (input: Tensor, blue_adj: Union[Tensor, float], red_adj: Union[Tensor, float]) -> Tensor:
    """
    Apply color balance on a given image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in [-1., 1.]
        blue_adj (Tensor | float): Additive blue hue adjustment with shape (N,1,H,W) in [-1., 1.]
        red_adj (Tensor | float): Additive red hue adjustment with shape (N,1,H,W) in [-1., 1.]
        
    Returns:
        Tensor: Result image with shape (N,3,H,W) in [-1., 1.]
    """
    pass