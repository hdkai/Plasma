# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from pytest import fixture, mark
from torch import cat, isnan, linspace, zeros, zeros_like
from .common import tensorread, tensorwrite

from plasma.conversion import rgb_to_luminance, rgb_to_yuv, yuv_to_rgb
from plasma.filters import bilateral_filter, slice_bilateral_grid

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
def test_bilateral_coarse_local_contrast (image_path):
    image = tensorread(image_path, size=1024)
    weight = linspace(-1., 1., 20).view(-1, 1)
    # Compute base layer
    y, u, v = rgb_to_yuv(image).split(1, dim=1)
    y = bilateral_filter(y, y, kernel_size=(7, 7), grid_size=(36, 256, 256))
    yuv = cat([y, u, v], dim=1)
    base_layer = yuv_to_rgb(yuv)
    # Interpolate
    base_colors = base_layer.flatten(start_dim=1)
    image_colors = image.flatten(start_dim=1)
    result_colors = base_colors.lerp(image_colors, weight + 1.)
    # Reshape
    _, _, height, width = image.shape
    result = result_colors.view(-1, 3, height, width)
    result = result.clamp(min=-1., max=1.)
    # Write
    tensorwrite("clarity_bilateral.gif", *result.split(1, dim=0))

@mark.parametrize("image_path", IMAGE_PATHS)
def test_bilateral_fine_local_contrast (image_path):
    image = tensorread(image_path, size=1024)
    weight = linspace(-1., 1., 20).view(-1, 1)
    # Compute base layer
    luminance = rgb_to_luminance(image)
    base_layer = bilateral_filter(image, luminance, kernel_size=(5, 5), grid_size=(32, 1000, 1000))
    # Interpolate
    base_colors = base_layer.flatten(start_dim=1)
    image_colors = image.flatten(start_dim=1)
    result_colors = base_colors.lerp(image_colors, weight + 1.)
    # Reshape
    _, _, height, width = image.shape
    result = result_colors.view(-1, 3, height, width)
    result = result.clamp(min=-1., max=1.)
    # Write
    tensorwrite("texture_bilateral.gif", *result.split(1, dim=0))