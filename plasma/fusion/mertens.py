# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import cat, Tensor
from torch.nn.functional import normalize

from ..conversion import rgb_to_luminance
from ..filters import laplacian_of_gaussian_filter

def exposure_fusion (input: Tensor, omega_con: Tensor, omega_exp: Tensor, omega_sat: Tensor) -> Tensor: # INCOMPLETE
    """
    Fuse a set of exposure using Mertens et al (2009).

    https://mericam.github.io/papers/exposure_fusion_reduced.pdf

    Parameters:
        input (Tensor): Input exposure stack with shape (N,3M,H,W) in range [-1., 1.].
        omega_con (Tensor): Contrast weight with shape (N,1) in range [0., 1.].
        omega_exp (Tensor): Exposure weight with shape (N,1) in range [0., 1.].
        omega_sat (Tensor): Saturation weight with shape (N,1) in range [0., 1.].

    Returns:
        Tensor: Fused image with shape (N,3,H,W) in range [-1., 1.].
    """
    _, _, height, width = input.shape
    # Split exposures and compute luminance
    exposures = input.split(3, dim=1)
    luminances = [rgb_to_luminance(exposure) for exposure in exposures]
    # Compute weights
    exp_alpha = 0.0     # median exposure
    exp_sigma = 0.5     # Gaussian std
    contrast_weights = [laplacian_of_gaussian_filter(luminance) for luminance in luminances]
    exposure_weights = [(-(exposure - exp_alpha).pow(2.) / (2 * exp_sigma ** 2)).exp().prod(dim=1, keepdim=True) for exposure in exposures]
    saturation_weights = [exposure.std(dim=1, keepdim=True) for exposure in exposures]
    # Fuse weights
    omegas = [omega_con, omega_exp, omega_sat]
    weights = [cat([weight.flatten(start_dim=1).pow(omega).view(-1, 1, height, width) for weight, omega in zip(weights, omegas)], dim=1).prod(dim=1, keepdim=True) for weights in zip(contrast_weights, exposure_weights, saturation_weights)]
    weights = cat(weights, dim=1)
    weights = normalize(weights, p=1, dim=1, eps=1e-8)
    
    
    return weights[:,:1,:,:]
