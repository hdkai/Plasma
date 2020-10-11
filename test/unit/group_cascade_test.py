# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from pathlib import Path
from pytest import fixture, mark

from plasma.io.group import group_exposures, cascade_similarity, feature_similarity, timestamp_similarity

def test_cascade_group_flash_group_a ():
    exposure_paths = [
        "test/media/group/1.jpg",
        "test/media/group/2.jpg",
        "test/media/group/3.jpg",
        "test/media/group/4.jpg",
        "test/media/group/5.jpg",
    ]
    similarity_fn = cascade_similarity(timestamp_similarity(), feature_similarity())
    groups = group_exposures(exposure_paths, similarity_fn)
    assert len(groups) == 1

def test_cascade_group_flash_group_b ():
    exposure_paths = [
        "test/media/group/17.jpg",
        "test/media/group/18.jpg",
    ]
    similarity_fn = cascade_similarity(timestamp_similarity(), feature_similarity())
    groups = group_exposures(exposure_paths, similarity_fn)
    assert len(groups) == 1