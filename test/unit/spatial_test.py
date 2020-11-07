# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from pytest import fixture, mark
from torch import cat, linspace, zeros_like
from .common import tensorread, tensorwrite

from plasma.spatial import color_balance, contrast, exposure, saturation

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
def test_spatial_contrast (image_path):
    image = tensorread(image_path)
    _, _, height, width = image.shape
    weights = linspace(-1., 1., 20).view(-1, 1, 1, 1).repeat(1, 1, height, width)
    weights[:,:,:,width//2:] *= -1.
    results = contrast(image, weights)
    tensorwrite("contrast.gif", *results.split(1, dim=0))

@mark.parametrize("image_path", IMAGE_PATHS)
def test_spatial_exposure (image_path):
    image = tensorread(image_path)
    _, _, height, width = image.shape
    weights = linspace(-1., 1., 20).view(-1, 1, 1, 1).repeat(1, 1, height, width)
    weights[:,:,:,width//2:] *= -1.
    results = exposure(image, weights)
    tensorwrite("exposure.gif", *results.split(1, dim=0))

@mark.parametrize("image_path", IMAGE_PATHS)
def test_spatial_saturation (image_path):
    image = tensorread(image_path)
    _, _, height, width = image.shape
    weights = linspace(-1., 1., 20).view(-1, 1, 1, 1).repeat(1, 1, height, width)
    weights[:,:,:,width//2:] *= -1.
    results = saturation(image, weights)
    tensorwrite("saturation.gif", *results.split(1, dim=0))

@mark.parametrize("image_path", IMAGE_PATHS)
def test_spatial_temperature (image_path):
    image = tensorread(image_path)
    _, _, height, width = image.shape
    weights = linspace(-1., 1., 20).view(-1, 1, 1, 1).repeat(1, 1, height, width)
    weights[:,:,:,width//2:] *= -1.
    weights = cat([weights, zeros_like(weights)], dim=1)
    results = color_balance(image, weights)
    tensorwrite("temperature.gif", *results.split(1, dim=0))

@mark.parametrize("image_path", IMAGE_PATHS)
def test_spatial_tint (image_path):
    image = tensorread(image_path)
    _, _, height, width = image.shape
    weights = linspace(-1., 1., 20).view(-1, 1, 1, 1).repeat(1, 1, height, width)
    weights[:,:,:,width//2:] *= -1.
    weights = cat([zeros_like(weights), weights], dim=1)
    results = color_balance(image, weights)
    tensorwrite("tint.gif", *results.split(1, dim=0))