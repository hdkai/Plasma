# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from PIL import Image
from pytest import fixture, mark

from plasma.io.group import group_exposures, timestamp_similarity
from plasma.io.group.timestamp import exposure_timestamp

@mark.parametrize("image_path", [
    "test/media/group/1.jpg"
])
def test_image_timestamp (image_path):
    image = Image.open(image_path)
    timestamp = exposure_timestamp(image)
    assert timestamp > -1

def test_timestamp_group_flash_group_a ():
    exposure_paths = [
        "test/media/group/1.jpg",
        "test/media/group/2.jpg",
        "test/media/group/3.jpg",
        "test/media/group/4.jpg",
        "test/media/group/5.jpg",
    ]
    groups = group_exposures(exposure_paths, timestamp_similarity())
    assert len(groups) == 3

def test_timestamp_group_flash_group_b ():
    exposure_paths = [
        "test/media/group/17.jpg",
        "test/media/group/18.jpg",
    ]
    groups = group_exposures(exposure_paths, timestamp_similarity())
    assert len(groups) == 2