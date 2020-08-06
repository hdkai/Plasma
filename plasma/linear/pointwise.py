# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import clamp, Tensor
from typing import Union

def contrast (input: Tensor, weight: Union[float, Tensor]) -> Tensor:
    """
    Apply contrast adjustment to an image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in [-1., 1.].
        weight (float | Tensor): Scalar weight in [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in [-1., 1.].
    """
    result = clamp(input * weight, -1., 1.)
    return result

def exposure (input: Tensor, weight: Union[float, Tensor]) -> Tensor: # INCOMPLETE
    """
    Apply exposure adjustment to an image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in [-1., 1.].
        weight (float | Tensor): Scalar weight in [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in [-1., 1.].
    """
    pass

def saturation (input: Tensor, weight: Union[float, Tensor]) -> Tensor: # INCOMPLETE
    """
    Apply saturation adjustment to an image.

    Parameters:
        input (Tensor): RGB image with shape (N,3,H,W) in [-1., 1.].
        weight (float | Tensor): Scalar weight in [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in [-1., 1.].
    """
    pass

def temperature (input: Tensor, weight: Union[float, Tensor]) -> Tensor: # INCOMPLETE
    """
    Apply temperature adjustment to an image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in [-1., 1.].
        weight (float | Tensor): Scalar weight in [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in [-1., 1.].
    """
    pass

def tint (input: Tensor, weight: Union[float, Tensor]) -> Tensor: # INCOMPLETE
    """
    Apply tint adjustment to an image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in [-1., 1.].
        weight (float | Tensor): Scalar weight in [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in [-1., 1.].
    """
    pass