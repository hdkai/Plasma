# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from PIL import Image
from pytest import fixture, mark

from plasma.io import align_exposures

IMAGE_PAIRS = [
    ("test/media/align/aurora.jpg", "test/media/align/delivery.jpg")
]

@mark.parametrize("input_path,target_path", IMAGE_PAIRS)
def test_align (input_path, target_path):
    input = Image.open(input_path)
    target = Image.open(target_path)
    result = align_exposures(input, target)
    result.save("align.jpg")