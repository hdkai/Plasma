# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from PIL import Image
from numpy import asarray
from typing import List, Union

def align_exposures (input: Union[Image.Image, List[Image.Image]], target: Image.Image) -> Union[Image.Image, List[Image.Image]]: # INCOMPLETE
    """
    Align one or more images with a target image.

    Parameters:
        input (PIL.Image | list): Source image(s).
        target (PIL.Image): Target image.

    Returns:
        PIL.Image | list: Aligned images.
    """
    # Always use input[0]
    pass