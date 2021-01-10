# 
#   Plasma
#   Copyright (c) 2021 Homedeck, LLC.
#

from pytest import fixture, mark
from torch import cat, linspace, stack, tensor, zeros, zeros_like
from .common import tensorread, tensorwrite

from plasma.linear import selective_color
import plasma.linear.selective as selective

IMAGE_PATHS = [
    "test/media/filter/1.jpg",
    "test/media/filter/2.jpg",
    "test/media/filter/3.jpg",
    "test/media/filter/4.jpg",
]

@fixture
def red_basis ():
    basis = tensor([
        [255., 0., 0.],     # red
    ]) / 255.
    return basis

@fixture
def red_green_basis ():
    basis = tensor([
        [255., 0., 0.],     # red
        [0., 255., 0.]      # green
    ]) / 255.
    return basis 

@fixture
def lr_bases ():
    bases = tensor([
        [255., 0., 0.],     # red
        [255., 165., 0.],   # orange
        [0., 255., 0.],     # green
        [0., 255., 255.],   # cyan
        [30., 144., 255.],  # dodger blue
        [138., 43., 226.]   # blue violet
    ]) / 255.
    return bases

@mark.parametrize("path", IMAGE_PATHS)
def test_selective_weight (path, lr_bases):
    image = tensorread(path)
    weight_map = selective._selective_color_weight_map(image, lr_bases)
    weights = weight_map.split(1, dim=1)
    names = ["red", "orange", "green", "cyan", "blue", "violet"]
    for weight, name in zip(weights, names):
        tensorwrite(f"{name}.jpg", weight)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_selective_color (image_path, red_green_basis):
    image = tensorread(image_path)
    weight = tensor([[
        [0.2, 0., 0.], # red
        [0., -0.8, 0.] # green
    ]])
    result = selective_color(image, red_green_basis, weight)
    tensorwrite("selective_color.jpg", result)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_selective_hue (image_path, red_basis):
    image = tensorread(image_path)
    weight = linspace(-1., 1., 20).unsqueeze(dim=1)
    weight = stack([ weight, zeros_like(weight), zeros_like(weight) ], dim=2)
    result = selective_color(image, red_basis, weight)
    tensorwrite("selective_hue.gif", *result.split(1, dim=0))

@mark.parametrize("image_path", IMAGE_PATHS)
def test_selective_saturation (image_path, red_basis):
    image = tensorread(image_path)
    weight = linspace(-1., 1., 20).unsqueeze(dim=1)
    weight = stack([ zeros_like(weight), weight, zeros_like(weight) ], dim=2)
    result = selective_color(image, red_basis, weight)
    tensorwrite("selective_sat.gif", *result.split(1, dim=0))

@mark.parametrize("image_path", IMAGE_PATHS)
def test_selective_luminance (image_path, red_basis):
    image = tensorread(image_path)
    weight = linspace(-1., 1., 20).unsqueeze(dim=1)
    weight = stack([ zeros_like(weight), zeros_like(weight), weight ], dim=2)
    result = selective_color(image, red_basis, weight)
    tensorwrite("selective_lum.gif", *result.split(1, dim=0))