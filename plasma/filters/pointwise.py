# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import cat, clamp, Tensor
from typing import Union

from ..conversion import rgb_to_yuv, yuv_to_rgb

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
    return input

def saturation (input: Tensor, weight: Union[float, Tensor]) -> Tensor: # TEST
    """
    Apply saturation adjustment to an image.

    Parameters:
        input (Tensor): RGB image with shape (N,3,H,W) in [-1., 1.].
        weight (float | Tensor): Scalar weight in [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in [-1., 1.].
    """
    yuv = rgb_to_yuv(input)
    y, u, v = yuv.split(1, dim=1)
    u *= (weight + 1.)
    u *= (weight + 1.)
    yuv = cat([y, u, v], dim=1)
    result = yuv_to_rgb(yuv)
    return result

def temperature (input: Tensor, weight: Union[float, Tensor]) -> Tensor: # INCOMPLETE
    """
    Apply temperature adjustment to an image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in [-1., 1.].
        weight (float | Tensor): Scalar weight in [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in [-1., 1.].
    """
    return input

def tint (input: Tensor, weight: Union[float, Tensor]) -> Tensor: # INCOMPLETE
    """
    Apply tint adjustment to an image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in [-1., 1.].
        weight (float | Tensor): Scalar weight in [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in [-1., 1.].
    """
    return input