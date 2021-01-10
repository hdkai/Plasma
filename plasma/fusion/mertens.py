# 
#   Plasma
#   Copyright (c) 2021 Homedeck, LLC.
#

from torch import cat, Tensor
from torch.nn.functional import normalize

from ..blending import blend_pyramid

def exposure_fusion (input: Tensor, omega_exp: Tensor, omega_sat: Tensor, peak_level: int=2) -> Tensor:
    """
    Fuse a set of exposure using Mertens et al (2009).
    We use a slightly modified implementation that drops the contrast term.

    https://mericam.github.io/papers/exposure_fusion_reduced.pdf
    https://www.ipol.im/pub/art/2018/230/article.pdf

    Parameters:
        input (Tensor): Input exposure stack with shape (N,3M,H,W) in range [-1., 1.].
        omega_exp (Tensor): Exposure weight with shape (N,1) in range [0., 1.].
        omega_sat (Tensor): Saturation weight with shape (N,1) in range [0., 1.].
        peak_level (int): Size of peak Laplacian pyramid level during blending.

    Returns:
        Tensor: Fused image with shape (N,3,H,W) in range [-1., 1.].
    """
    _, _, height, width = input.shape    
    # Compute weights
    exp_alpha = 0.0     # median exposure
    exp_sigma = 0.4     # sigma
    exposures = input.split(3, dim=1)
    exposure_weights = [(-(exposure - exp_alpha).pow(2.) / (2 * exp_sigma ** 2)).exp().prod(dim=1, keepdim=True) for exposure in exposures]
    saturation_weights = [exposure.std(dim=1, keepdim=True) for exposure in exposures]
    # Fuse weights
    omegas = [omega_exp, omega_sat]
    weights = [cat([weight.flatten(start_dim=1).pow(omega).view(-1, 1, height, width) for weight, omega in zip(weights, omegas)], dim=1).prod(dim=1, keepdim=True) for weights in zip(exposure_weights, saturation_weights)]
    weights = cat(weights, dim=1)
    weights = normalize(weights, p=1, dim=1, eps=1e-6)
    # Pyramid blend
    result = blend_pyramid(input, weights, peak_level=peak_level)
    return result
