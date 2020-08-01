# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import cat, clamp, linspace, meshgrid, ones_like, stack, Tensor
from torch.nn.functional import grid_sample, interpolate, pad
from typing import Optional, Tuple

from .gaussian import gaussian_blur_3d

def bilateral_filter_2d (input: Tensor, kernel_size: Tuple[int, int], grid_size: Optional[Tuple[int, int, int]] = None):
    """
    Apply the bilateral filter to a 2D image.

    Parameters:
        input (Tensor): Input image with shape (N,C,H,W).
        kernel_size (tuple): Kernel size in intensity and spatial dimensions (Ki,Ks).
        grid_size (tuple): Bilateral grid size. If `None`, a suitable default will be used.

    Returns:
        Tensor: Filtered image with shape (N,C,H,W).
    """
    _,channels,_,_ = input.shape
    kernel_size = (kernel_size[0], kernel_size[1], kernel_size[1])
    grid_size = grid_size if grid_size is not None else (64, 64, 16)
    # Filter each channel independently
    channels = input.split(1, dim=1)
    filtered_channels = []
    for channel in channels:
        # Construct grid
        intensity_grid, weight_grid = splat_bilateral_grid(channel, channel, grid_size)
        # Filter
        intensity_grid = gaussian_blur_3d(intensity_grid, kernel_size)
        weight_grid = gaussian_blur_3d(weight_grid, kernel_size)
        # Slice
        filtered_channel = slice_bilateral_grid(intensity_grid, channel, weight_grid)
        filtered_channels.append(filtered_channel)
    # Stack
    result = cat(filtered_channels, dim=1)
    return result

def splat_bilateral_grid (input: Tensor, guide: Tensor, grid_size: Tuple[int, int, int]) -> Tuple[Tensor, Tensor]:
    """
    Splat a 2D image into a 3D bilateral grid.

    Parameters:
        input (Tensor): Input image with shape (N,C,H,W).
        guide (Tensor): Splatting guide map with shape (N,1,H,W) in [-1., 1.].
        grid_size (tuple): Grid size in each dimension (I,Sy,Sx).

    Returns:
        tuple: Bilateral grid with shape (N,C,I,Sy,Sx); and weight grid with same shape in [0., 1.].
    """
    samples, _,_,_ = input.shape
    intensity_bins, spatial_bins_y, spatial_bins_x = grid_size
    # Downsample
    downsampled_input = interpolate(input, size=(spatial_bins_x, spatial_bins_y), mode="bilinear", align_corners=True) # NxCxSxS
    downsampled_guide = interpolate(guide, size=(spatial_bins_x, spatial_bins_y), mode="bilinear", align_corners=True) # Nx1xSxS
    # Create volumes
    input_grid = downsampled_input.unsqueeze(dim=2) # NxCx1xSxS
    volume_padding = (0, 0, 0, 0, 0, intensity_bins - 1, 0, 0)
    input_volume = pad(input_grid, volume_padding, "constant", 0.) # NxCxIxSxS
    weight_volume = pad(ones_like(input_grid), volume_padding, "constant", 0.)
    # Create sample grid
    ig, hg, wg = meshgrid(linspace(-1., 1., intensity_bins), linspace(-1., 1., spatial_bins_y), linspace(-1., 1., spatial_bins_x))
    ig = ig.repeat(samples, 1, 1, 1).to(input.device) - (downsampled_guide + 1.)
    hg = hg.repeat(samples, 1, 1, 1).to(input.device)
    wg = wg.repeat(samples, 1, 1, 1).to(input.device)
    sample_grid = stack([wg, hg, ig], dim=4)
    # Sample
    intensity_grid = grid_sample(input_volume, sample_grid, mode="bilinear", align_corners=True)
    weight_grid = grid_sample(weight_volume, sample_grid, mode="bilinear", align_corners=True)
    # Return
    return intensity_grid, weight_grid

def slice_bilateral_grid (input: Tensor, guide: Tensor, weight: Optional[Tensor] = None) -> Tensor:
    """
    Slice a 3D bilateral grid into a 2D image.

    Parameters:
        input (Tensor): Input bilateral grid with shape (N,C,I,Sy,Sx).
        guide (Tensor): Slicing guide map with shape (N,1,H,W) in [-1., 1.].
        weight (Tensor): Bilateral weight grid with shape (N,C,I,Sy,Sx) in [0., 1.]. If `None`, then no homogenous divide is performed.

    Returns:
        Tensor: Sliced image with shape (N,C,H,W).
    """
    samples, _, height, width = guide.shape
    # Create slice grid
    hg, wg = meshgrid(linspace(-1., 1., height), linspace(-1., 1., width))
    hg = hg.repeat(samples, 1, 1).unsqueeze(dim=3).to(input.device)
    wg = wg.repeat(samples, 1, 1).unsqueeze(dim=3).to(input.device)
    slice_grid = guide.permute(0, 2, 3, 1).contiguous()     # NxHxWx1
    slice_grid = cat([wg, hg, slice_grid], dim=3)           # NxHxWx3
    slice_grid = slice_grid.unsqueeze(dim=1)                # Nx1xHxWx3
    # Sample
    result = grid_sample(input, slice_grid, mode="bilinear", align_corners=True).squeeze(dim=2)
    weight = grid_sample(weight, slice_grid, mode="bilinear", align_corners=True).squeeze(dim=2) if weight is not None else ones_like(result)
    # Normalize # Prevent divide by zero
    weight = clamp(weight, min=1e-3)
    result = result / weight
    return result
