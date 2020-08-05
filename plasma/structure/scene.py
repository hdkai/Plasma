# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from enum import IntEnum
from torch import Tensor

class ImageScene (IntEnum):
    """
    The scene of a given real estate photograph.
    """
    Interior = 0
    Exterior = 1
    Twilight = 2
    Aerial = 3

def image_scene (input: Tensor) -> ImageScene: # INCOMPLETE
    """
    Determine the scene of an image.

    Parameters:
        image (Tensor): Input RGB image with shape (N,3,H,W) in [-1., 1.].

    Returns:
        ImageScene: Image scene.
    """
    pass