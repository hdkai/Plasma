# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from pytest import fixture, mark
from torch import isnan, zeros, zeros_like
from .common import tensorread, tensorwrite

from plasma.linear import clarity, texture
from plasma.sampling import bilateral_filter_2d, slice_bilateral_grid

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

def test_bilateral_slice ():
    grid = zeros(1, 1, 16, 64, 64)
    guide = zeros(1, 1, 128, 128)
    weight = zeros_like(grid)
    sliced = slice_bilateral_grid(grid, guide, weight)
    assert not isnan(sliced).any()

@mark.parametrize("image_path", IMAGE_PATHS)
def test_coarse_local_contrast (image_path):
    image = tensorread(image_path, size=None)
    result = clarity(image, -1.)
    tensorwrite("coarse.jpg", result)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_fine_local_contrast (image_path):
    image = tensorread(image_path, size=None)
    result = texture(image, -1.)
    tensorwrite("fine.jpg", result)