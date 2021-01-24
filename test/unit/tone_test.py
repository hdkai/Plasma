# 
#   Plasma
#   Copyright (c) 2021 Homedeck, LLC.
#

from numpy import linspace
from pytest import fixture, mark
from torch import float32, tensor
from .common import tensorread, tensorwrite

from plasma.curves import natural_cubic_curve

IMAGE_PATHS = [
    "test/media/filter/1.jpg",
    "test/media/filter/2.jpg",
    "test/media/filter/3.jpg",
    "test/media/filter/4.jpg",
    "test/media/filter/10.jpg",
]

@mark.parametrize("image_path", IMAGE_PATHS)
def test_natural_cubic_curve (image_path):
    image = tensorread(image_path)
    control = tensor([
        #[-1., 0.318, 0.812, 1.], # max LR exposure
        [-1, -0.874, -0.686, -0.254] # min LR exposure
    ])
    result = natural_cubic_curve(image, control)
    tensorwrite("tone.jpg", result)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_natural_cubic_shadow_control (image_path):
    image = tensorread(image_path)
    results = []
    for i in linspace(-0.8, 0.8, 21):
        control = tensor([ [-1., i, 0.33, 1.] ]).to(float32)
        result = natural_cubic_curve(image, control)
        results.append(result)
    tensorwrite("tone_shadow.gif", *results)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_natural_cubic_highlight_control (image_path):
    image = tensorread(image_path)
    results = []
    for i in linspace(-0.8, 0.8, 21):
        control = tensor([ [-1., -0.33, i, 1.] ]).to(float32)
        result = natural_cubic_curve(image, control)
        results.append(result)
    tensorwrite("tone_highlight.gif", *results)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_natural_cubic_contrast_control (image_path):
    image = tensorread(image_path)
    results = []
    for i in linspace(-0.4, 0.4, 21):
        control = tensor([ [-1., i - 0.33, 0.33 - i, 1.] ]).to(float32)
        result = natural_cubic_curve(image, control)
        results.append(result)
    tensorwrite("tone_contrast.gif", *results)