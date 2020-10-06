# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from pytest import fixture, mark
from torch import cat, linspace
from .common import tensorread, tensorwrite

from plasma.conversion import rgb_to_lab, lab_to_rgb

def test_lab_temp ():
    image = tensorread("test/media/conversion/linear.jpg")
    _, _, height, width = image.shape
    weights = linspace(-1., 1., 20).unsqueeze(dim=1)
    lab = rgb_to_lab(image)
    l, a, b = lab.split(1, dim=1)
    l = l.repeat(20, 1, 1, 1)
    a = a.repeat(20, 1, 1, 1)
    b = (b.flatten(start_dim=1) + weights * 50).view(20, 1, height, width)
    lab = cat([l, a, b], dim=1)
    results = lab_to_rgb(lab)
    tensorwrite("lab_temp.gif", *results.split(1, dim=0))

def test_lab_tint ():
    image = tensorread("test/media/conversion/linear.jpg")
    _, _, height, width = image.shape
    weights = linspace(-1., 1., 20).unsqueeze(dim=1)
    lab = rgb_to_lab(image)
    l, a, b = lab.split(1, dim=1)
    l = l.repeat(20, 1, 1, 1)
    a = (a.flatten(start_dim=1) + weights * 25).view(20, 1, height, width)
    b = b.repeat(20, 1, 1, 1)
    lab = cat([l, a, b], dim=1)
    results = lab_to_rgb(lab)
    tensorwrite("lab_tint.gif", *results.split(1, dim=0))