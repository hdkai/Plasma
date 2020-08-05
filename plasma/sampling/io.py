# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from PIL import Image
from torch import Tensor
from torchvision.transforms import Compose, Grayscale, Normalize, Resize, ToTensor

def lutread (path: str) -> Tensor:
    """
    Load a 1D LUT from file.

    Parameters:
        path (str): Path to LUT file.

    Returns:
        Tensor: 1D LUT with shape (L,) in [-1., 1.].
    """
    image = Image.open(path)
    to_tensor = Compose([
        Grayscale(),
        Resize((1, image.width)),
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5])
    ])
    lut = to_tensor(image)
    lut = lut.squeeze()
    return lut

def cuberead (path: str) -> Tensor: # INCOMPLETE
    """
    Load a 3D LUT from file.

    Parameters:
        path (str): Path to CUBE file.

    Returns:
        Tensor: 3D LUT with shape (L,L,L) in [-1., 1.].
    """
    pass
