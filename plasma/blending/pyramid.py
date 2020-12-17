# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from math import log2
from torch import logspace, stack, zeros, Tensor
from torch.nn.functional import interpolate
from typing import List

def blend_pyramid (input: Tensor, weights: Tensor, levels: float=0.8) -> Tensor: # INCOMPLETE # Reparameterize `levels`
    """
    Blend a stack of images using Burt & Adelson (1983).

    http://www.wisdom.weizmann.ac.il/~vision/courses/2003_1/pyramid83.pdf

    Parameters:
        input (Tensor): Input image stack with shape (N,3M,H,W) in range [-1., 1.].
        weights (Tensor): Weight map stack with shape (N,M,H,W) in range [0., 1.].
        levels (float): Relative number of levels to use for blending, in range [0., 1.].

    Returns:
        Tensor: Fused image with shape (N,3,H,W) in range [-1., 1.].
    """
    _, channels, height, width = input.shape
    exposures = channels // 3
    # Compute pyramid levels
    max_levels = log2(min(width, height))
    levels = int(levels * max_levels)    
    # Iterate
    result_pyramid = []
    for image, weight in zip(input.split(3, dim=1), weights.split(1, dim=1)):
        # Compute pyramids
        image_pyramid = laplacian_pyramid(image, levels)
        weight_pyramid = gaussian_pyramid(weight, levels)
        # Blend
        blended_levels = [image_pyramid[i] * weight_pyramid[i] for i in range(levels)]
        result_pyramid.append(blended_levels)
    # Fuse pyramid
    result_pyramid = [stack([result_pyramid[i][j] for i in range(exposures)], dim=0).sum(dim=0) for j in range(levels)]    
    result = collapse_pyramid(result_pyramid)
    return result

def gaussian_pyramid (input: Tensor, levels: int) -> Tensor:
    """
    Compute Gaussian pyramid for an image.

    Parameters:
        input (Tensor): Input image stack with shape (N,C,H,W) in range [-1., 1.].
        levels (int): Pyramid levels.

    Returns:
        list: Guassian levels, with each level being a low pass octave tensor.
    """
    result = [input]
    for i in range(levels-1):
        level = interpolate(result[i], scale_factor=0.5, mode="bilinear", align_corners=False)
        result.append(level)
    return result

def laplacian_pyramid (input: Tensor, levels: int) -> Tensor:
    """
    Compute Laplacian pyramid for an image.

    Parameters:
        input (Tensor): Input image stack with shape (N,C,H,W) in range [-1., 1.].
        levels (int): Pyramid levels.

    Returns:
        list: Laplacian pyramid, with each level being a high pass octave tensor.
    """
    result = []
    gaussian_levels = gaussian_pyramid(input, levels)
    for i in range(levels - 1):
        level = gaussian_levels[i] - interpolate(gaussian_levels[i+1], gaussian_levels[i].shape[2:], mode="bilinear", align_corners=False)
        result.append(level)
    result.append(gaussian_levels[-1])
    return result

def collapse_pyramid (input: List[Tensor]) -> Tensor:
    """
    Collapse a Laplacian pyramid to create an image.

    Parameters:
        input (list): Input Laplacian pyramid.

    Result:
        Tensor: Resulting image with shape (N,C,H,W).
    """
    for i in range(len(input) - 1, 0, -1):
        input[i-1] = input[i-1] + interpolate(input[i], input[i-1].shape[2:], mode="bilinear", align_corners=False)
    return input[0]