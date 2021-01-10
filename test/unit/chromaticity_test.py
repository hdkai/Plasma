# 
#   Plasma
#   Copyright (c) 2021 Homedeck, LLC.
#

from pytest import fixture, mark
from torch import cat, linspace, tensor, zeros, zeros_like
from .common import tensorread, tensorwrite

from plasma.conversion import rgb_to_xyz, xyz_to_lab, lab_to_xyz, xyz_to_rgb, srgb_to_linear, linear_to_srgb
from plasma.linear import color_balance

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
    weights = linspace(-1., 1., 20).unsqueeze(dim=1)
    weights = cat([weights, zeros_like(weights)], dim=1)
    results = color_balance(image, weights)
    tensorwrite("temperature.gif", *results.split(1, dim=0))

@mark.parametrize("image_path", IMAGE_PATHS)
def test_chroma_tint (image_path):
    image = tensorread(image_path)
    weights = linspace(-1., 1., 20).unsqueeze(dim=1)
    weights = cat([zeros_like(weights), weights], dim=1)
    results = color_balance(image, weights)
    tensorwrite("tint.gif", *results.split(1, dim=0))

@mark.parametrize("image_path", IMAGE_PATHS)
def test_lab_temperature (image_path):
    #image_path = "/Users/yusuf/Desktop/white.jpg"
    image = tensorread(image_path)
    _, _, height, width = image.shape
    weights = linspace(-1., 1., 20).unsqueeze(dim=1)
    rgb = image #srgb_to_linear(image)
    xyz = rgb_to_xyz(rgb)
    lab = xyz_to_lab(xyz)
    l, a, b = lab.split(1, dim=1)
    l = l.repeat(20, 1, 1, 1)
    a = a.repeat(20, 1, 1, 1)
    b = (b.flatten(start_dim=1) + weights * 50).view(20, 1, height, width)
    lab = cat([l, a, b], dim=1)
    xyz = lab_to_xyz(lab)
    rgb = xyz_to_rgb(xyz)
    results = rgb #linear_to_srgb(rgb)
    tensorwrite("temperature.gif", *results.split(1, dim=0))

@mark.parametrize("image_path", IMAGE_PATHS)
def test_lab_tint (image_path):
    image = tensorread(image_path)
    _, _, height, width = image.shape
    weights = linspace(-1., 1., 20).unsqueeze(dim=1)
    rgb = srgb_to_linear(image)
    xyz = rgb_to_xyz(rgb)
    lab = xyz_to_lab(xyz)
    l, a, b = lab.split(1, dim=1)
    l = l.repeat(20, 1, 1, 1)
    a = (a.flatten(start_dim=1) + weights * 25).view(20, 1, height, width)
    b = b.repeat(20, 1, 1, 1)
    lab = cat([l, a, b], dim=1)
    xyz = lab_to_xyz(lab)
    rgb = xyz_to_rgb(xyz)
    results = linear_to_srgb(rgb)
    tensorwrite("tint.gif", *results.split(1, dim=0))

def test_manual_chroma_temperature ():
    image = tensorread("test/media/conversion/linear.jpg", size=None)
    weight = tensor([[-0.38, 0.]])
    result = color_balance(image, weight)
    tensorwrite("temperature.jpg", result)