# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import stack, zeros_like, Tensor
from torch.nn.functional import grid_sample

def color_sample_1d (input: Tensor, lut: Tensor): # INCOMPLETE
    """
    Apply a 1D look-up table to an image.

    Parameters:
        input (Tensor): RGB image with shape (N,3,H,W) in [-1., 1.].
        lut (Tensor): Lookup table with shape (L,) in [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in [-1., 1.].
    """
    # Create volume
    batch,_,_,_ = input.shape
    volume = lut.repeat(batch, 1, 1, 1)
    # Create grid
    colors = input.permute(0, 2, 3, 1)
    wg = colors.flatten(2)
    hg = zeros_like(wg)
    grid = stack([wg, hg], dim=3)
    # Sample
    result = grid_sample(volume, grid, mode="bilinear", align_corners=False)
    result = result.squeeze(dim=1).view_as(colors).permute(0, 3, 1, 2)
    return result

def color_sample_3d (input: Tensor, lut: Tensor): # INCOMPLETE
    """
    Apply a 3D look-up table to an image.

    Parameters:
        input (Tensor): RGB image with shape (N,3,H,W) in [-1., 1.].
        lut (Tensor): Lookup table with shape (L,L,L) in [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in [-1., 1.].
    """
    pass
