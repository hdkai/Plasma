# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import clamp, lerp, where, Tensor
from typing import Union

from ..conversion import rgb_to_luminance
from ..sampling import bilateral_filter_2d

def clarity (input: Tensor, weight: Union[float, Tensor]) -> Tensor:
    """
    Apply local contrast to an image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in [-1., 1.].
        weight (float | Tensor): Scalar weight in [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in [-1., 1.].
    """
    base_layer = bilateral_filter_2d(input, kernel_size=(5, 5, 11))
    base_layer = clamp((base_layer + 1.) * 1.1 - 1., max=1.) # Brighten to mimic LR
    result = lerp(base_layer, input, 1. + weight)
    result = clamp(result, min=-1., max=1.)
    return result

def highlights (input: Tensor, weight: Union[float, Tensor], tonal_range: float = 1.) -> Tensor:
    """
    Apply highlight attentuation to an image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in [-1., 1.].
        weight (float | Tensor): Scalar weight in [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in [-1., 1.].
    """
    luma = rgb_to_luminance(input)
    mask = -bilateral_filter_2d(luma, kernel_size=(5, 5, 11))
    highlight_mask = -weight * clamp(mask + (1. - tonal_range), max=0.)
    result = _blend_overlay(input, highlight_mask)
    return result

def shadows (input: Tensor, weight: Union[float, Tensor], tonal_range: float = 1.) -> Tensor:
    """
    Apply shadow attentuation to an image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in [-1., 1.].
        weight (float | Tensor): Scalar weight in [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in [-1., 1.].
    """
    luma = rgb_to_luminance(input)
    mask = -bilateral_filter_2d(luma, kernel_size=(5, 5, 11))
    shadow_mask = weight * clamp(mask - (1. - tonal_range), min=0.)
    result = _blend_overlay(input, shadow_mask)
    return result

def sharpen (input: Tensor, weight: Union[float, Tensor]) -> Tensor: # INCOMPLETE
    """
    Apply sharpness enhancement to an image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in [-1., 1.].
        weight (float | Tensor): Scalar weight in [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in [-1., 1.].
    """
    pass

def _blend_overlay (base: Tensor, overlay: Tensor):
    base, overlay = (base + 1.) / 2., (overlay + 1.) / 2.
    multiply = 2. * base * overlay
    screen = 1. - 2. * (1. - base) * (1. - overlay)
    result = where(base < 0.5, multiply, screen)
    return result * 2. - 1.