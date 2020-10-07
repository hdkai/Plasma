# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from pytest import fixture, mark
from torch import cat, linspace, tensor, zeros, zeros_like
from .common import tensorread, tensorwrite

from plasma.filters import color_balance
import plasma.filters.chromaticity as chromaticity

IMAGE_PATHS = [
    "test/media/conversion/input.jpg",
    "test/media/conversion/linear.jpg",
]

@mark.parametrize("image_path", IMAGE_PATHS)
def test_identity_chromaticity (image_path):
    image = tensorread(image_path)
    weight = zeros(1, 2)
    result = color_balance(image, weight)
    tensorwrite("identity_chroma.jpg", result)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_chroma_temperature (image_path):
    image = tensorread(image_path)
    weights = linspace(-0.5, -0.3, 20).unsqueeze(dim=1)
    weights = cat([weights, zeros_like(weights)], dim=1)
    results = color_balance(image, weights)
    tensorwrite("temperature.gif", *results.split(1, dim=0))

def test_manual_chroma_temperature ():
    image = tensorread("test/media/conversion/linear.jpg", size=None)
    weight = tensor([[-0.38, 0.]])
    result = color_balance(image, weight)
    tensorwrite("temperature.jpg", result)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_chroma_tint (image_path):
    image = tensorread(image_path)
    weights = linspace(-1., 1., 20).unsqueeze(dim=1)
    weights = cat([zeros_like(weights), weights], dim=1)
    results = color_balance(image, weights)
    tensorwrite("tint.gif", *results.split(1, dim=0))