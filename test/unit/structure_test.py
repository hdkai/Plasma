# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from pytest import fixture, mark
from .common import tensorread

from plasma.structure import image_scene, ImageScene

@mark.parametrize("image_path,type", [
    ("test/media/scene/interior.jpg", ImageScene.Interior),
    ("test/media/scene/exterior.jpg", ImageScene.Exterior)
])
def test_scene (image_path, type):
    image = tensorread(image_path)
    scene = image_scene(image)
    assert scene == type, "Inferred scene type should match canonical type"