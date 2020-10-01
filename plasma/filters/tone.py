# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import Tensor

def tone_curve (input: Tensor, control: Tensor) -> Tensor:
    """
    Apply a tone curve to an image.

    We use a natural cubic curve to perform the mapping.
    The control query points are fixed at [-1.0, -0.33, 0.33, 1.0].

    Parameters:
        input (Tensor): Input image with shape (N,...) in range [-1., 1.].
        control (Tensor): Control value points with shape (N,4) in range [-1., 1.].

    Returns:
        Tensor: Result image with shape (N,...) in range [-1., 1.].
    """
    pass