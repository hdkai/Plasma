# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from pytest import fixture, mark
from torch import zeros
from .common import tensorwrite

from plasma.filters.functional import radial_gradient, top_bottom_gradient, bottom_top_gradient, left_right_gradient, right_left_gradient

def test_radial_gradient ():
    input = zeros(1, 1, 720, 1280)
    mask = radial_gradient(input, 0.75)
    tensorwrite("radial.jpg", mask)

def test_top_bottom_gradient ():
    input = zeros(1, 1, 720, 1280)
    mask = top_bottom_gradient(input, 0.25)
    tensorwrite("top_bottom.jpg", mask)

def test_bottom_top_gradient ():
    input = zeros(1, 1, 720, 1280)
    mask = bottom_top_gradient(input, 0.25)
    tensorwrite("bottom_top.jpg", mask)

def test_left_right_gradient ():
    input = zeros(1, 1, 720, 1280)
    mask = left_right_gradient(input, 0.75)
    tensorwrite("left_right.jpg", mask)

def test_right_left_gradient ():
    input = zeros(1, 1, 720, 1280)
    mask = right_left_gradient(input, 0.75)
    tensorwrite("right_left.jpg", mask)