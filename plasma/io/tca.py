# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from PIL import Image

def tca_correction (*images: Image.Image) -> Image.Image:
    """
    Appply transverse chromatic aberration correction on an image.
    Parameters:
        images (PIL.Image | list): Input image(s).
        coefficients (Tensor): Cubic red-blue TCA coefficients with shape (2,4). If `None`, it will be computed (can be slow).
    Returns:
        PIL.Image | list: Corrected image(s).
    """
    pass