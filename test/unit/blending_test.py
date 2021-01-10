# 
#   Plasma
#   Copyright (c) 2021 Homedeck, LLC.
#

from pytest import fixture, mark
from .common import tensorread, tensorwrite

from plasma.blending import blend_overlay, blend_soft_light

IMAGE_PAIRS = [
    ("", "")
]

@mark.parametrize("base_path,overlay_path", IMAGE_PAIRS)
def test_blend_overlay (base_path, overlay_path):
    pass

@mark.parametrize("base_path,overlay_path", IMAGE_PAIRS)
def test_blend_soft_light (base_path, overlay_path):
    pass