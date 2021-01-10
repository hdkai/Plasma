# 
#   Plasma
#   Copyright (c) 2021 Homedeck, LLC.
#

from pytest import fixture, mark
from torch import cat, linspace, tensor, zeros, zeros_like
from .common import tensorread, tensorwrite

from plasma.fusion import exposure_fusion

@fixture
def window_exposures ():
    return [
        "test/media/fusion/1.jpg",
        "test/media/fusion/2.jpg",
        "test/media/fusion/3.jpg"
    ]

@fixture
def bright_kitchen_exposures ():
    return [
        "test/media/fusion/4.jpg",
        "test/media/fusion/5.jpg",
        "test/media/fusion/6.jpg"
    ]

def test_exposure_fusion (window_exposures):
    exposures = [tensorread(path) for path in window_exposures]
    exposure_stack = cat(exposures, dim=1)
    fusion = exposure_fusion(exposure_stack, 1., 1.)
    fusion = fusion.clamp(min=-1., max=1.)
    tensorwrite("fusion.jpg", fusion)

def test_exposure_fusion_highlights (bright_kitchen_exposures):
    exposures = [tensorread(path) for path in bright_kitchen_exposures]
    exposure_stack = cat(exposures, dim=1)
    fusion = exposure_fusion(exposure_stack, 1., 1.)
    fusion = fusion.clamp(min=-1., max=1.)
    tensorwrite("fusion.jpg", fusion)