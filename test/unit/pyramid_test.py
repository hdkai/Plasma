# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from pytest import fixture, mark
from torch import cat, linspace, zeros_like
from .common import tensorread, tensorwrite

from plasma.blending.pyramid import gaussian_pyramid, laplacian_pyramid, collapse_pyramid

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
    "test/media/filter/10.jpg",
]

@mark.parametrize("image_path", IMAGE_PATHS)
def test_gaussian_pyramid (image_path):
    image = tensorread(image_path)
    pyramid = gaussian_pyramid(image, 5)
    for i, level in enumerate(pyramid):
        tensorwrite(f"gaussian_{i}.jpg", level)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_laplacian_pyramid (image_path):
    image = tensorread(image_path)
    pyramid = laplacian_pyramid(image, 5)
    for i, level in enumerate(pyramid):
        tensorwrite(f"laplacian_{i}.jpg", level)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_laplacian_pyramid_identity (image_path):
    image = tensorread(image_path)
    pyramid = laplacian_pyramid(image, 5)
    identity = collapse_pyramid(pyramid)
    error = 2. * (identity - image).abs() - 1.
    tensorwrite(f"laplacian_identity.jpg", identity)
    tensorwrite(f"laplacian_error.jpg", error)