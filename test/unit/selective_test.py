# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from pytest import fixture, mark
from torch import cat, split, stack, tensor, tensor, zeros
from .common import tensorread, tensorwrite

from plasma.filters import selective_color
import plasma.filters.selective as selective

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
    weights = split(weight_map, 1, dim=1)
    names = ["red", "orange", "green", "cyan", "blue", "violet"]
    for weight, name in zip(weights, names):
        tensorwrite(f"{name}.jpg", weight)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_selective_hue (image_path, red_green_basis):
    image = tensorread(image_path)
    _, _, height, width = image.shape
    zero_weight = zeros(1, 2, height, width)
    hue_weight = zeros(1, 2, height, width)
    hue_weight[:,0,:,:] -= 0.7
    hue_weight[:,1:,:] += 0.9
    result = selective_color(image, red_green_basis, hue_weight, zero_weight, zero_weight)
    tensorwrite("selective_hue.jpg", result)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_selective_saturation (image_path, red_green_basis):
    image = tensorread(image_path)
    _, _, height, width = image.shape
    zero_weight = zeros(1, 1, height, width)
    others = zeros(1, 2, height, width)
    sat_weight = cat([
        zero_weight - 0.7,
        zero_weight + 0.5
    ], dim=1)
    result = selective_color(image, red_green_basis, others, sat_weight, others)
    tensorwrite("selective_sat.jpg", result)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_selective_exposure (image_path, red_basis):
    image = tensorread(image_path)
    _, _, height, width = image.shape
    zero_weight = zeros(1, 1, height, width)
    exp_weight = zero_weight - 0.5
    result = selective_color(image, red_basis, zero_weight, zero_weight, exp_weight)
    tensorwrite("selective_exp.jpg", result)