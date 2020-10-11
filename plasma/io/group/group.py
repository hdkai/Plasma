# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from typing import Callable, List

from .timestamp import exposure_timestamp
from .utility import load_exposure

def group_exposures (exposure_paths: List[str], similarity_fn: Callable[[Image.Image, Image.Image], bool]) -> List[List[str]]:
    """
    Group a set of exposures using a similarity function.

    Parameters:
        exposure_paths (list): Paths to exposures to group.
        similarity_fn (callable): Pairwise similarity function returning a boolean.
    
    Returns:
        list: Groups of exposure paths.
    """
    # Check
    if not exposure_paths:
        return []
    # Trivial case
    if len(exposure_paths) == 1:
        return [exposure_paths]
    # Load exposures
    exposures = [Image.open(path) for path in exposure_paths]
    exposures_with_paths = zip(exposure_paths, exposures)
    exposures_with_paths = sorted(exposures_with_paths, key=lambda pair: exposure_timestamp(pair[1]))
    # Group
    current_path, current_exposure = exposures_with_paths.pop(0)
    groups = []
    current_group = [current_path]
    while exposures_with_paths:
        path, exposure = exposures_with_paths.pop(0)
        if not similarity_fn(current_exposure, exposure):
            groups.append(current_group)
            current_group = []
        current_group.append(path)
        current_path, current_exposure = path, exposure
    groups.append(current_group)
    return groups