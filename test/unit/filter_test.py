# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from numpy import linspace
from pytest import fixture, mark
from .common import tensorread, tensorwrite

from plasma.filters import contrast, clarity, exposure, highlights, saturation, shadows, sharpen, temperature, tint

IMAGE_PATHS = [
    "test/media/filter/1.jpg",
    "test/media/filter/2.jpg",
    "test/media/filter/3.jpg",
    "test/media/filter/4.jpg",
    "test/media/filter/5.jpg",
    "test/media/filter/6.jpg",
    "test/media/filter/7.jpg",
    "test/media/filter/8.jpg",
]

@mark.parametrize("image_path", IMAGE_PATHS)
def test_clarity (image_path):
    image = tensorread(image_path)
    results = [clarity(image, weight) for weight in linspace(-1., 1., 20)]
    tensorwrite("clarity.gif", *results)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_contrast (image_path):
    image = tensorread(image_path)
    results = [contrast(image, weight) for weight in linspace(-1., 1., 20)]
    tensorwrite("contrast.gif", *results)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_exposure (image_path):
    image = tensorread(image_path)
    results = [exposure(image, weight) for weight in linspace(-1., 1., 20)]
    tensorwrite("exposure.gif", *results)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_highlights (image_path):
    image = tensorread(image_path)
    results = [highlights(image, weight) for weight in linspace(-1., 1., 20)]
    tensorwrite("highlights.gif", *results)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_saturation (image_path):
    image = tensorread(image_path)
    results = [saturation(image, weight) for weight in linspace(-1., 1., 20)]
    tensorwrite("saturation.gif", *results)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_shadows (image_path):
    image = tensorread(image_path)
    results = [shadows(image, weight) for weight in linspace(-1., 1., 20)]
    tensorwrite("shadows.gif", *results)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_shadows_highlights (image_path):
    image = tensorread(image_path, size=None)
    result = image
    result = highlights(result, -0.9)
    result = shadows(result, 0.75)
    tensorwrite("shadhi.jpg", result)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_sharpen (image_path):
    image = tensorread(image_path)
    results = [sharpen(image, weight) for weight in linspace(-1., 1., 20)]
    tensorwrite("sharpen.gif", *results)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_temperature (image_path):
    image = tensorread(image_path)
    results = [temperature(image, weight) for weight in linspace(-1., 1., 20)]
    tensorwrite("temperature.gif", *results)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_tint (image_path):
    image = tensorread(image_path)
    results = [tint(image, weight) for weight in linspace(-1., 1., 20)]
    tensorwrite("tint.gif", *results)