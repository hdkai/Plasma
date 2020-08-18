# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from pytest import fixture, mark
from .common import tensorread

from plasma.structure import image_scene, ImageScene

@mark.parametrize("image_path,type", [
    ("test/media/filter/1.jpg", ImageScene.Interior),
    ("test/media/filter/2.jpg", ImageScene.Exterior),
    ("test/media/filter/3.jpg", ImageScene.Interior),
    ("test/media/filter/4.jpg", ImageScene.Exterior),
    ("test/media/filter/5.jpg", ImageScene.Exterior),
    ("test/media/filter/6.jpg", ImageScene.Exterior),
])
def test_scene (image_path, type):
    image = tensorread(image_path)
    scene = image_scene(image)
    assert scene == type, "Inferred scene type should match canonical type"