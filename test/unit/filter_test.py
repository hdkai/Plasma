# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from pytest import fixture, mark
from torch import linspace
from .common import tensorread, tensorwrite

from plasma.filters import contrast, clarity, exposure, highlights, saturation, shadows, sharpen, temperature, texture, tint

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
def test_clarity (image_path):
    image = tensorread(image_path)
    weights = linspace(-1., 1., 20).unsqueeze(dim=1)
    results = clarity(image, weights)
    tensorwrite("clarity.gif", *results.split(1, dim=0))

@mark.parametrize("image_path", IMAGE_PATHS)
def test_contrast (image_path):
    image = tensorread(image_path)
    weights = linspace(-1., 1., 20).unsqueeze(dim=1)
    results = contrast(image, weights)
    tensorwrite("contrast.gif", *results.split(1, dim=0))

@mark.parametrize("image_path", IMAGE_PATHS)
def test_highlights (image_path):
    image = tensorread(image_path)
    weights = linspace(-1., 1., 20).unsqueeze(dim=1)
    results = highlights(image, weights)
    tensorwrite("highlights.gif", *results.split(1, dim=0))

@mark.parametrize("image_path", IMAGE_PATHS)
def test_saturation (image_path):
    image = tensorread(image_path)
    weights = linspace(-1., 1., 20).unsqueeze(dim=1)
    results = saturation(image, weights)
    tensorwrite("saturation.gif", *results.split(1, dim=0))

@mark.parametrize("image_path", IMAGE_PATHS)
def test_shadows (image_path):
    image = tensorread(image_path)
    weights = linspace(-1., 1., 20).unsqueeze(dim=1)
    results = shadows(image, weights)
    tensorwrite("shadows.gif", *results.split(1, dim=0))

@mark.parametrize("image_path", IMAGE_PATHS)
def test_shadows_highlights (image_path):
    image = tensorread(image_path, size=None)
    result = highlights(image, -0.9)
    result = shadows(result, 0.75)
    tensorwrite("shadhi.jpg", result)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_sharpen (image_path):
    image = tensorread(image_path)
    weights = linspace(-1., 1., 20).unsqueeze(dim=1)
    results = sharpen(image, weights)
    tensorwrite("sharpen.gif", *results.split(1, dim=0))

@mark.parametrize("image_path", IMAGE_PATHS)
def test_temperature (image_path):
    image = tensorread(image_path)
    weights = linspace(-1., 1., 20).unsqueeze(dim=1)
    result = temperature(image, weights)
    tensorwrite("temperature.gif", *result.split(1, dim=0))

@mark.parametrize("image_path", IMAGE_PATHS)
def test_texture (image_path):
    image = tensorread(image_path)
    weights = linspace(-1., 1., 20).unsqueeze(dim=1)
    results = texture(image, weights)
    tensorwrite("texture.gif", *results.split(1, dim=0))

@mark.parametrize("image_path", IMAGE_PATHS)
def test_tint (image_path):
    image = tensorread(image_path)
    weights = linspace(-1., 1., 20).unsqueeze(dim=1)
    result = tint(image, weights)
    tensorwrite("tint.gif", *result.split(1, dim=0))