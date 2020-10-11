# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from pathlib import Path
from pytest import fixture, mark

from plasma.io.group import group_exposures, edge_similarity

def test_group_image ():
    exposure_paths = [
        "test/media/group/1.jpg",
        "test/media/group/2.jpg",
        "test/media/group/3.jpg",
        "test/media/group/4.jpg",
        "test/media/group/5.jpg",
    ]
    groups = group_exposures(exposure_paths, edge_similarity())
    assert len(groups) == 1 and len(groups[0]) == 5

def test_group_raw ():
    exposure_paths = [
        "test/media/raw/1.arw",
        "test/media/raw/2.arw",
        "test/media/raw/3.arw",
        "test/media/raw/4.arw",
        "test/media/raw/5.arw",
    ]
    groups = group_exposures(exposure_paths, edge_similarity())
    assert len(groups) == 1 and len(groups[0]) == 5

def test_edge_group_flash_group_b ():
    exposure_paths = [
        "test/media/group/17.jpg",
        "test/media/group/18.jpg",
    ]
    groups = group_exposures(exposure_paths, edge_similarity())
    assert len(groups) == 1 and len(groups[0]) == 2

def test_edge_group_flash_group_c ():
    exposure_paths = [
        "test/media/group/11.jpg",
        "test/media/group/12.jpg",
        "test/media/group/13.jpg",
    ]
    groups = group_exposures(exposure_paths, edge_similarity())
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
    groups = group_exposures(exposure_paths, edge_similarity())
    assert len(groups) == 2 and all([len(group) == 3 for group in groups])