# 
#   Plasma
#   Copyright (c) 2021 Homedeck, LLC.
#

from imageio import imwrite
from numpy import linspace, tile, uint16
from pytest import fixture, mark
from .common import tensorread, tensorwrite

from plasma.filters import color_sample_1d, color_sample_3d, cuberead, lutread

IMAGE_PATHS = [
    "test/media/filter/1.jpg",
    "test/media/filter/2.jpg",
    "test/media/filter/3.jpg",
    "test/media/filter/4.jpg",
    "test/media/filter/10.jpg",
]

def test_create_identity_lut ():
    lut = linspace(0., 1., num=4096)
    lut = tile(lut, (16, 1))
    lut = (lut * 65535).astype(uint16)
    imwrite("identity.tif", lut)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_lut (image_path):
    image = tensorread(image_path)
    lut = lutread("test/media/lut/ramp.tif")
    result = color_sample_1d(image, lut)
    tensorwrite("lut.jpg", result)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_load_cube (image_path):
    image = tensorread(image_path)
    cube = cuberead("test/media/lut/identity.cube")
    result = color_sample_3d(image, cube)
    tensorwrite("cube.jpg", result)
