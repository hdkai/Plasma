# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from PIL import Image
from pytest import fixture, mark
from torchvision.transforms import Compose, Normalize, Resize, ToPILImage, ToTensor

from plasma.sampling import color_sample_1d, cuberead, lutread

IMAGE_PATHS = [
    "media/filter/1.jpg",
    "media/filter/2.jpg",
    "media/filter/3.jpg",
    "media/filter/4.jpg",
]

def tensorread (path, size=1024):
    to_tensor = Compose([
        Resize(size),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(path)
    image = to_tensor(image).unsqueeze(dim=0)
    return image

def tensorwrite (name, *images):
    to_image = Compose([
        Normalize(mean=[-1., -1., -1.], std=[2., 2., 2.]),
        ToPILImage()
    ])
    images = [to_image(image.squeeze()) for image in images]
    if len(images) > 1:
        images[0].save(name, save_all=True, append_images=images[1:], duration=100, loop=0)
    else:
        images[0].save(name)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_lut (image_path):
    image = tensorread(image_path)
    lut = lutread("media/lut/ramp.tif")
    result = color_sample_1d(image, lut)
    tensorwrite("lut.jpg", result)

def test_load_cube ():
    #cube = cuberead("media/lut/identity.cube")
    pass