# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import clamp, lerp, where, Tensor
from typing import Union

from ..conversion import rgb_to_luminance
from ..sampling import bilateral_filter_2d, gaussian_blur_2d

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
    # Compute mask
    luma = -rgb_to_luminance(input)
    mask = bilateral_filter_2d(luma, kernel_size=(7, 7, 7))
    mask = mask + (1. - tonal_range)
    mask = clamp(mask, min=-1., max=0.)
    # Blend
    mask = -weight * mask
    result = _blend_overlay(input, mask)
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
    # Compute mask
    luma = -rgb_to_luminance(input)
    mask = bilateral_filter_2d(luma, kernel_size=(7, 7, 7))
    mask = mask - (1. - tonal_range)
    mask = clamp(mask, min=0., max=1.)
    # Blend
    mask = weight * mask
    result = _blend_soft_light(input, mask)
    # Contrast scale
    contrast = 1. + 0.2 * abs(weight)
    result = clamp(result * contrast, min=-1., max=1.)
    return result

def sharpen (input: Tensor, weight: Union[float, Tensor]) -> Tensor:
    """
    Apply sharpness enhancement to an image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in [-1., 1.].
        weight (float | Tensor): Scalar weight in [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in [-1., 1.].
    """
    base_layer = gaussian_blur_2d(input, (5, 5))
    result = lerp(base_layer, input, 1. + weight)
    result = clamp(result, min=-1., max=1.)
    return result

def _blend_overlay (base: Tensor, overlay: Tensor) -> Tensor:
    # Rescale
    base = (base + 1.) / 2.
    overlay = (overlay + 1.) / 2.
    # Compute sub blending modes
    multiply = 2. * base * overlay
    screen = 1. - 2. * (1. - base) * (1. - overlay)
    # Blend and rescale
    result = where(base < 0.5, multiply, screen)
    result = 2. * result - 1.
    return result

def _blend_soft_light (base: Tensor, overlay: Tensor) -> Tensor: # Use Photoshop blending
    # Rescale
    base = (base + 1.) / 2.
    overlay =  (overlay + 1.) / 2.
    # Blend
    result = (1. - 2. * overlay) * base.pow(2.) + 2. * base * overlay
    ps_correct = 2 * base * (1. - overlay) + base.sqrt() * (2. * overlay - 1.)
    result = where(overlay < 0.5, result, ps_correct)
    # Rescale
    result = 2. * result - 1.
    return result