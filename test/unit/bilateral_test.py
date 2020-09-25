# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from pytest import fixture, mark
from torch import isnan, zeros, zeros_like
from .common import tensorread, tensorwrite

from plasma.sampling import bilateral_filter_2d, slice_bilateral_grid

IMAGE_PATHS = [
    "test/media/filter/1.jpg",
    "test/media/filter/2.jpg",
    "test/media/filter/3.jpg",
    "test/media/filter/4.jpg",
    "test/media/filter/5.jpg",
    "test/media/filter/6.jpg",
    "test/media/filter/7.jpg",
]

@mark.parametrize("image_path", IMAGE_PATHS)
def test_bilateral_filter (image_path):
    image = tensorread(image_path)
    result = bilateral_filter_2d(image, kernel_size=(5, 9), grid_size=(16, 512, 512))
    tensorwrite("bilateral.jpg", result)

def test_bilateral_slice ():
    grid = zeros(1, 1, 16, 64, 64)
    guide = zeros(1, 1, 128, 128)
    weight = zeros_like(grid)
    sliced = slice_bilateral_grid(grid, guide, weight)
    assert not isnan(sliced).any()