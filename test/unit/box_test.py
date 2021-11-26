# 
#   Plasma
#   Copyright (c) 2021 Yusuf Olokoba.
#

from pytest import fixture, mark
from torch import zeros, zeros_like
from .common import tensorread, tensorwrite

from torchplasma.filters import box_filter

IMAGE_PATHS = [
    "test/media/filter/1.jpg",
    "test/media/filter/2.jpg",
    "test/media/filter/3.jpg",
    "test/media/filter/4.jpg",
    "test/media/filter/5.jpg",
    "test/media/filter/6.jpg",
    "test/media/filter/7.jpg",
    "test/media/filter/8.jpg",
    "test/media/filter/9.jpg",
]

@mark.parametrize("image_path", IMAGE_PATHS)
def test_box_filter (image_path):
    image = tensorread(image_path, size=1024)
    result = box_filter(image, 5)
    tensorwrite("box.jpg", result)