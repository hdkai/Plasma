# 
#   Plasma
#   Copyright (c) 2021 Yusuf Olokoba.
#

from pytest import fixture, mark
from .common import tensorread, tensorwrite

from torchplasma.blending import blend_overlay, blend_soft_light

IMAGE_PAIRS = [
    ("", "")
]

@mark.parametrize("base_path,overlay_path", IMAGE_PAIRS)
def test_blend_overlay (base_path, overlay_path):
    pass

@mark.parametrize("base_path,overlay_path", IMAGE_PAIRS)
def test_blend_soft_light (base_path, overlay_path):
    pass