# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import Tensor

def blend_pyramid (input: Tensor, weight: Tensor) -> Tensor: # INCOMPLETE
    """
    Blend a stack of images using Burt & Adelson (1983).

    http://www.wisdom.weizmann.ac.il/~vision/courses/2003_1/pyramid83.pdf

    Parameters:
        input (Tensor): Input image stack with shape (N,3M,H,W) in range [-1., 1.].
        weight (Tensor): Weight map stack with shape (N,M,H,W) in range [0., 1.].

    Returns:
        Tensor: Fused image with shape (N,3,H,W) in range [-1., 1.].
    """
    pass