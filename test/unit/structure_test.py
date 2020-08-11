# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from PIL import Image
from pytest import fixture, mark
from torchvision.transforms import Compose, Normalize, Resize, ToPILImage, ToTensor

from plasma.structure import image_scene, ImageScene

def tensorread (path, size=1024):
    to_tensor = Compose([
        Resize(size),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(path)
    image = to_tensor(image).unsqueeze(dim=0)
    return image

@mark.parametrize("image_path,type", [
    ("test/media/scene/interior.jpg", ImageScene.Interior),
    ("test/media/scene/exterior.jpg", ImageScene.Exterior)
])
def test_scene (image_path, type):
    image = tensorread(image_path)
    scene = image_scene(image)
    assert scene == type, "Inferred scene type should match canonical type"