# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import cat, clamp, tensor, Tensor

def contrast (input: Tensor, weight: Tensor) -> Tensor: # INCOMPLETE
    """
    Apply contrast adjustment to an image.

    Parameters:
        input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].
        weight (Tensor | float): Weight map with shape (N,1,H,W) in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    pass

def exposure (input: Tensor, weight: Tensor) -> Tensor: # INCOMPLETE
    """
    Apply exposure adjustment to an image.

    Parameters:
        input (Tensor): Input image with shape (N,3,H,W) in [-1., 1.].
        weight (float | Tensor): Weight map with shape (N,1,H,W) in [-1., 1.].
    
    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in [-1., 1.].
    """
    pass

def saturation (input: Tensor, weight: Tensor) -> Tensor: # INCOMPLETE
    """
    Apply saturation adjustment to an image.

    Parameters:
        input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].
        weight (Tensor | float): Weight map with shape (N,1,H,W) in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    pass

def color_balance (input: Tensor, weight: Tensor) -> Tensor: # INCOMPLETE
    """
    Apply color balance adjustment on an image.

    Parameters:
        input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].
        weight (Tensor): Temperature and tint weight map with shape (N,2,H,W) in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    pass