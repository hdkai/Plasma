# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from cv2 import drawMatches
from imageio import imwrite
from numpy import asarray
from pathlib import Path
from PIL import Image
from pytest import fixture, mark

from plasma.io.group import feature_similarity, group_exposures
import plasma.io.group.feature as feature

def test_group_image ():
    exposure_paths = [
        "test/media/group/1.jpg",
        "test/media/group/2.jpg",
        "test/media/group/3.jpg",
        "test/media/group/4.jpg",
        "test/media/group/5.jpg",
    ]
    groups = group_exposures(exposure_paths, feature_similarity())
    assert len(groups) == 1 and len(groups[0]) == 5

def test_flash_group_a ():
    exposure_paths = [
        "test/media/group/17.jpg",
        "test/media/group/18.jpg",
    ]
    groups = group_exposures(exposure_paths, feature_similarity())
    assert len(groups) == 1 and len(groups[0]) == 2

def test_flash_group_b ():
    exposure_paths = [
        "test/media/group/11.jpg",
        "test/media/group/12.jpg",
        "test/media/group/13.jpg",
    ]
    groups = group_exposures(exposure_paths, feature_similarity())
    assert len(groups) == 1 and len(groups[0]) == 3

def test_aerial_group ():
    exposure_paths = [
        "test/media/group/19.jpg",
        "test/media/group/20.jpg",
        "test/media/group/21.jpg",
        "test/media/group/22.jpg",
        "test/media/group/23.jpg",
        "test/media/group/24.jpg",
    ]
    groups = group_exposures(exposure_paths, feature_similarity())
    assert len(groups) == 2 and all([len(group) == 3 for group in groups])

def test_visualize_matches ():
    #exposure_paths = ["test/media/group/12.jpg", "test/media/group/13.jpg"]
    #exposure_paths = ["test/media/group/23.jpg", "test/media/group/24.jpg"]
    exposure_paths = ["/Users/yusuf/Desktop/Captain/6.JPG", "/Users/yusuf/Desktop/Captain/7.JPG"]
    image_a, image_b = [Image.open(path) for path in exposure_paths]
    keypoints_a, keypoints_b, matches = feature._compute_matches(image_a, image_b)
    cost = feature._compute_alignment_cost(keypoints_a, keypoints_b, matches)
    image_a, image_b = asarray(image_a), asarray(image_b)
    match_image = drawMatches(image_a, keypoints_a, image_b, keypoints_b, matches, None)
    imwrite("matches.jpg", match_image)
    print("Cost:", cost)