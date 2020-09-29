# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from PIL import Image
from pytest import fixture, mark

from plasma.io import lens_correction

def test_single_lens_correction_no_exif ():
    image_path = "test/media/lens/1.jpg"
    image = Image.open(image_path)
    result = lens_correction(image)
    result.save(f"lens.jpg")

def test_single_lens_correction ():
    image_path = "test/media/lens/1.jpg"
    image = Image.open(image_path)
    result = lens_correction(image)
    result.save(f"lens.jpg")
    assert result.getexif() is not None

def test_multiple_lens_correction ():
    image_paths = [
        "test/media/lens/1.jpg",
        "test/media/lens/2.jpg",
        "test/media/lens/3.jpg"
    ]
    images = [Image.open(image_path) for image_path in image_paths]
    results = lens_correction(*images)
    [result.save(f"lens_{i}.jpg") for i, result in enumerate(results)]