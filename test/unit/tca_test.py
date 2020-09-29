# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from PIL import Image
from pytest import fixture, mark

from plasma.io import tca_correction

IMAGE_PATHS = [
    "test/media/tca/1.jpg",
    "test/media/tca/2.jpg",
]

@mark.parametrize("image_path", IMAGE_PATHS)
def test_tca (image_path):
    image = Image.open(image_path)
    result = tca_correction(image)
    result.save(f"tca.jpg")