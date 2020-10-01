# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import cat, lerp, where, Tensor

from ..conversion import rgb_to_luminance, rgb_to_yuv, yuv_to_rgb
from ..sampling import bilateral_filter_2d, gaussian_blur_2d
from .functional import blend_soft_light

def clarity (input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply coarse local contrast to an image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in range [-1., 1.].
        weight (float | Tensor): Scalar weight in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    y, u, v = rgb_to_yuv(input).split(1, dim=1)
    y = bilateral_filter_2d(y, kernel_size=(7, 7), grid_size=(36, 256, 256))
    yuv = cat([y, u, v], dim=1)
    base_layer = yuv_to_rgb(yuv)
    result = lerp(base_layer, input, 1. + weight)
    result = result.clamp(min=-1., max=1.)
    return result

def highlights (input: Tensor, weight: Tensor, tonal_range: float=1.) -> Tensor:
    """
    Apply highlight attentuation to an image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in range [-1., 1.].
        weight (float | Tensor): Scalar weight in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    # Compute mask
    mask = -rgb_to_luminance(input)
    mask = mask + (1. - tonal_range)
    mask = mask.clamp(min=-1., max=0.)
    # Blend
    mask = -weight * mask
    result = blend_soft_light(input, mask)
    return result

def shadows (input: Tensor, weight: Tensor, tonal_range: float=1.) -> Tensor:
    """
    Apply shadow attentuation to an image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in range [-1., 1.].
        weight (float | Tensor): Scalar weight in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    # Compute mask
    luma = -rgb_to_luminance(input)
    mask = bilateral_filter_2d(luma, kernel_size=(5, 11), grid_size=(16, 64, 64))
    mask = mask - (1. - tonal_range)
    mask = mask.clamp(min=0., max=1.)
    # Blend
    mask = weight * mask
    result = blend_soft_light(input, mask)
    # Contrast scale
    contrast = 1. + 0.2 * abs(weight)
    result = result * contrast
    result = result.clamp(min=-1., max=1.)
    return result

def sharpen (input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply sharpness enhancement to an image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in range [-1., 1.].
        weight (float | Tensor): Scalar weight in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    base_layer = gaussian_blur_2d(input, (5, 5))
    result = lerp(base_layer, input, 1. + weight)
    result = result.clamp(min=-1., max=1.)
    return result

def texture (input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply fine local contrast to an image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in range [-1., 1.].
        weight (float | Tensor): Scalar weight in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    base_layer = bilateral_filter_2d(input, kernel_size=(5, 5), grid_size=(32, 1000, 1000))
    result = lerp(base_layer, input, 1. + weight)
    result = result.clamp(min=-1., max=1.)
    return result