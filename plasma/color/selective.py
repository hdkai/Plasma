# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import cat, split, stack, Tensor
from torch.nn.functional import cosine_similarity

from .yuv import rgb_to_yuv, yuv_to_rgb

def selective_color (input: Tensor, colors: Tensor, hue_adj: Tensor, sat_adj: Tensor, exp_adj: Tensor) -> Tensor:
    """
    Apply selective color adjustment on a given image.
    All `M` filters are applied simultaneously.
    Parameters:
        input (Tensor): Input image with shape (N,3,H,W) in [-1., 1.]
        colors (Tensor): Basis RGB colors with shape (M,3) in [0., 1.]
        hue_adj (Tensor): Hue adjustment with shape (N,M,H,W) in [-1., 1.]
        sat_adj (Tensor): Saturation adjustment with shape (N,M,H,W) in [-1., 1.]
        exp_adj (Tensor): Exposure adjustment with shape (N,M,H,W) in [-1., 1.]
    Returns:
        Tensor: Result image with shape (N,3,H,W) in [-1., 1.]
    """
    # Compute weights
    batch, _, height, width  = input.shape
    weight_map = _selective_color_weight_map(input, colors)
    weights = split(weight_map, 1, dim=1)
    # Convert to YUV
    yuv = rgb_to_yuv(input)
    y, uv = yuv[:,:1,:,:], yuv[:,1:,:,:]
    # Adjust hue
    colors = uv.permute(0, 2, 3, 1).unsqueeze(dim=4)
    hues = hue_adj.permute(0, 2, 3, 1).split(1, dim=3)
    hues = [weight.permute(0, 2, 3, 1) * hue for weight, hue in zip(weights, hues)]
    rotations = [cat([hue.cos(), -hue.sin(), hue.sin(), hue.cos()], dim=3).view(batch, height, width, 2, 2) for hue in hues]
    rotation = rotations[0]
    for i in range(1, len(weights)):
        rotation = rotations[i].matmul(rotation)
    rotated_uv = rotation.matmul(colors).squeeze(dim=4).permute(0, 3, 1, 2)
    # Adjust saturation
    sats = sat_adj.split(1, dim=1)
    sats = [weight * sat for weight, sat in zip(weights, sats)]
    result_uv = rotated_uv + stack([uv * sat for sat in sats], dim=0).sum(dim=0)
    # Adjust exposure
    exps = exp_adj.split(1, dim=1)
    exps = [weight * exp for weight, exp in zip(weights, exps)]
    result_y = y + stack([y * exp for exp in exps], dim=0).sum(dim=0)
    # Convert to RGB
    result_yuv = cat([result_y, result_uv], dim=1)
    result = yuv_to_rgb(result_yuv)
    return result

def _selective_color_weight_map (input: Tensor, color: Tensor) -> Tensor:
    """
    Compute the color weight map for selective coloring.
    Parameters:
        input (Tensor): Input image with shape (N,3,H,W) in [-1., 1.]
        color (Tensor): Basis colors with shape (M,3) in [0., 1.]
    Returns:
        Tensor: Color weight map with shape (N,M,H,W) in [0., 1.]
    """    
    # Convert basis range
    color = (2.0 * color - 1.0).unsqueeze(dim=2)
    # Convert all to YUV
    uv_colors = rgb_to_yuv(input).flatten(start_dim=2)[:,1:,:]
    uv_basis = rgb_to_yuv(color)[:,1:,:]
    # Compute weight maps
    batch, _, height, width = input.shape
    uv_basis = split(uv_basis, 1, dim=0)
    weights = [cosine_similarity(uv_colors, basis.expand_as(uv_colors)).view(batch, 1, height, width) for basis in uv_basis]
    # Stack
    weight_map = cat(weights, dim=1).clamp(min=0.)
    return weight_map
