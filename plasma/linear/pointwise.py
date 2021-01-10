# 
#   Plasma
#   Copyright (c) 2021 Homedeck, LLC.
#

from torch import cat, clamp, tensor, Tensor

from ..conversion import rgb_to_yuv, yuv_to_rgb
from .tone import tone_curve

def contrast (input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply contrast adjustment to an image.

    Parameters:
        input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].
        weight (Tensor | float): Scalar weight with shape (N,1) in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    _, channels, width, height = input.shape
    result = input.flatten(start_dim=1) * (weight + 1.)
    result = result.view(-1, channels, width, height).clamp(min=-1., max=1.)
    return result

def exposure (input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply tonal exposure adjustment to an image.

    Parameters:
        input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].
        weight (Tensor | float): Scalar weight with shape (N,1) in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    samples, _, _, _ = input.shape
    ANCHORS = tensor([
        # x = [-1, 0, 1]
        [-1., -1., -1.],            # c_0
        [-0.874, -1. / 3., 0.318],  # c_1
        [-0.686, 1. / 3., 0.812],   # c_2
        [-0.254, 1., 1.]            # c_3
    ])
    ANCHORS = ANCHORS.repeat(samples, 1, 1).to(input.device)
    control = 0.5 * ANCHORS[:,:,0] * weight * (weight - 1.) - ANCHORS[:,:,1] * (weight + 1) * (weight - 1) + 0.5 * ANCHORS[:,:,2] * weight * (weight + 1)
    result = tone_curve(input, control)
    result = result.clamp(min=-1., max=1.)
    return result

def saturation (input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply saturation adjustment to an image.

    Parameters:
        input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].
        weight (Tensor | float): Scalar weight with shape (N,1) in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    _, _, height, width = input.shape
    yuv = rgb_to_yuv(input)
    y, u, v = yuv.split(1, dim=1)
    u = u.flatten(start_dim=1) * (weight + 1.)
    v = v.flatten(start_dim=1) * (weight + 1.)
    u = u.view(-1, 1, height, width)
    v = v.view(-1, 1, height, width)
    y = y.expand_as(u)
    yuv = cat([y, u, v], dim=1)
    result = yuv_to_rgb(yuv)
    return result

def color_balance (input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply color balance adjustment on an image.

    Parameters:
        input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].
        weight (Tensor): Scalar temperature and tint weights with shape (N,2) in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    _, _, height, width = input.shape
    yuv = rgb_to_yuv(input)
    y, u, v = yuv.split(1, dim=1)
    temp, tint = weight.split(1, dim=1)
    u = u.flatten(start_dim=1) + 0.1 * (tint - temp)
    v = v.flatten(start_dim=1) + 0.1 * (tint + temp)
    u = u.view(-1, 1, height, width)
    v = v.view(-1, 1, height, width)
    y = y.expand_as(u)
    yuv = cat([y, u, v], dim=1)
    result = yuv_to_rgb(yuv)
    return result