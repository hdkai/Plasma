# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import Tensor
from torch.nn import Module

from ..sampling import gaussian_blur_2d, gaussian_blur_3d

class Gaussian2D (Module):
    """
    2D Gaussian smoothing layer.
    """

    def __init__ (self, size):
        super(Gaussian2D, self).__init__()
        self.__size = size if isinstance(size, tuple) else (size, size)

    def forward (self, input):
        result = gaussian_blur_2d(input, self.__size)
        return result

class Gaussian3D (Module):
    """
    3D Gaussian smoothing layer.
    """

    def __init__ (self, size):
        super(Gaussian3D, self).__init__()
        self.__size = size if isinstance(size, tuple) else (size, size, size)

    def forward (self, input):
        result = gaussian_blur_3d(input, self.__size)
        return result