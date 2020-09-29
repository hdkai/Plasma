# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import cat, clamp, Tensor

from ..conversion import rgb_to_yuv, yuv_to_rgb

def contrast (input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply contrast adjustment to an image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in [-1., 1.].
        weight (float | Tensor): Scalar weight in [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in [-1., 1.].
    """
    result = input * (weight + 1.)
    result = result.clamp(min=-1., max=1.)
    return result

def exposure (input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply exposure adjustment to an image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in [-1., 1.].
        weight (float | Tensor): Scalar weight in [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in [-1., 1.].
    """
    input = (input + 1.) / 2.
    result = input * (weight + 1.)
    result = 2. * result - 1.
    result = result.clamp(min=-1., max=1.)
    return result

def saturation (input: Tensor, weight: Tensor) -> Tensor:
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
    u = u * (weight + 1.)
    v = v * (weight + 1.)
    yuv = cat([y, u, v], dim=1)
    result = yuv_to_rgb(yuv)
    return result

def temperature (input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply temperature adjustment to an image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in [-1., 1.].
        weight (float | Tensor): Scalar weight in [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in [-1., 1.].
    """
    yuv = rgb_to_yuv(input)
    y, u, v = yuv.split(1, dim=1)
    u = u - 0.1 * weight
    v = v + 0.1 * weight
    yuv = cat([y, u, v], dim=1)
    result = yuv_to_rgb(yuv)
    return result

def tint (input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply tint adjustment to an image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in [-1., 1.].
        weight (float | Tensor): Scalar weight in [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in [-1., 1.].
    """
    yuv = rgb_to_yuv(input)
    y, u, v = yuv.split(1, dim=1)
    u = u + 0.1 * weight
    v = v + 0.1 * weight
    yuv = cat([y, u, v], dim=1)
    result = yuv_to_rgb(yuv)
    return result