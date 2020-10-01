# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import cat, clamp, tensor, Tensor

from ..conversion import rgb_to_yuv, yuv_to_rgb
from .tone import tone_curve

def contrast (input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply contrast adjustment to an image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in range [-1., 1.].
        weight (float | Tensor): Scalar weight in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    result = input * (weight + 1.)
    result = result.clamp(min=-1., max=1.)
    return result

def exposure (input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply exposure adjustment to an image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in range [-1., 1.].
        weight (Tensor | float): Scalar weight with shape (N,1) in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    samples, _, _, _ = input.shape
    # Define anchors
    ANCHORS = tensor([
        # x = [-1, 0, 1]
        [-1., -1., -1.],            # t_0
        [-0.874, -1. / 3., 0.318],  # t_1
        [-0.686, 1. / 3., 0.812],   # t_2
        [-0.254, 1., 1.]            # t_3
    ])
    ANCHORS = ANCHORS.repeat(samples, 1, 1).to(input.device)
    # Interpolate control points with lagrange polynomials
    control = 0.5 * ANCHORS[:,:,0] * weight * (weight - 1.) - ANCHORS[:,:,1] * (weight + 1) * (weight - 1) + 0.5 * ANCHORS[:,:,2] * weight * (weight + 1)
    # Apply tone curve
    result = tone_curve(input, control)
    return result

def saturation (input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply saturation adjustment to an image.

    Parameters:
        input (Tensor): RGB image with shape (N,3,H,W) in range [-1., 1.].
        weight (float | Tensor): Scalar weight in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
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
        input (Tensor): Input RGB image with shape (N,3,H,W) in range [-1., 1.].
        weight (float | Tensor): Scalar weight in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
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
        input (Tensor): Input RGB image with shape (N,3,H,W) in range [-1., 1.].
        weight (float | Tensor): Scalar weight in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    yuv = rgb_to_yuv(input)
    y, u, v = yuv.split(1, dim=1)
    u = u + 0.1 * weight
    v = v + 0.1 * weight
    yuv = cat([y, u, v], dim=1)
    result = yuv_to_rgb(yuv)
    return result