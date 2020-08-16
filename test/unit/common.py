# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToPILImage, ToTensor

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
    channels = images[0].shape[1]
    to_image = Compose([
        Normalize(mean=channels * [-1.], std=channels * [2.]),
        ToPILImage()
    ])
    images = [to_image(image.squeeze()) for image in images]
    if len(images) > 1:
        images[0].save(name, save_all=True, append_images=images[1:], duration=100, loop=0)
    else:
        images[0].save(name)