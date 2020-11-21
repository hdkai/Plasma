# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
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

def test_exposure_fusion (window_exposures):
    exposures = [tensorread(path) for path in window_exposures]
    exposure_stack = cat(exposures, dim=1)
    fusion = exposure_fusion(exposure_stack, 0., 0., 1.)
    tensorwrite("fusion.jpg", fusion)