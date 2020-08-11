# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from enum import IntEnum
from pkg_resources import resource_filename
from torch import Tensor
from torch.autograd import no_grad
from torch.jit import load
from torch.nn.functional import interpolate

_scene_classifier = None

class ImageScene (IntEnum):
    """
    The scene of a given real estate photograph.
    """
    Interior = 0
    Exterior = 1
    Twilight = 2
    Aerial = 3

def image_scene (input: Tensor) -> ImageScene:
    """
    Determine the scene of an image.

    Parameters:
        image (Tensor): Input RGB image with shape (N,3,H,W) in [-1., 1.].

    Returns:
        ImageScene: Image scene.
    """
    global _scene_classifier
    if _scene_classifier is None:
        model_path = resource_filename("plasma.pretrained", "scene_classifier.pt")
        _scene_classifier = load(model_path)
    _scene_classifier = _scene_classifier.to(input.device)
    input = interpolate(input, (512, 512), mode="bilinear")
    with no_grad():
        logits = _scene_classifier(input)
    result = logits.argmax().item()
    result = ImageScene(result)
    return result