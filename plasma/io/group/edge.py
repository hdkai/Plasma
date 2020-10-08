# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from concurrent.futures import ThreadPoolExecutor
from cv2 import Canny, GaussianBlur
from numpy import ndarray
from typing import Iterable, List, Tuple

from .common import exposure_timestamp, load_exposure, normalize_exposures

def edge_group (exposure_paths: List[str], min_similarity=0.35, workers=8) -> List[List[str]]:
    """
    Group a set of exposures by their mutual edges.

    Parameters:
        exposure_paths (list): Paths to exposures to group.
        min_similarity (float). Minimum edge similarity for images to be considered in the same group. Should be in range [0., 1.].
        workers (int): Number of workers for IO.

    Returns:
        list: Groups of exposure paths.
    """
    # Check
    if not exposure_paths:
        return []
    # Trivial case
    if len(exposure_paths) == 1:
        return [exposure_paths]
    # Sort by EXIF timestamp
    exposure_paths = sorted(exposure_paths, key=lambda path: exposure_timestamp(path))
    # Load all exposures into memory, thread this, `map` preserves order
    with ThreadPoolExecutor(max_workers=workers) as executor:
        exposures = executor.map(lambda path: load_exposure(path, 512), exposure_paths)
    # Group
    groups = _group_exposures(exposure_paths, exposures, min_similarity)
    return groups

def _group_exposures (exposure_paths: List[str], exposures: Iterable[ndarray], min_similarity: float):
    """
    Group a set of exposure paths by their corresponding exposures' mutual edges.

    Parameters:
        exposure_paths (list): Paths to exposures to group.
        exposures (list): Corresponding exposures loaded into memory as `ndarray` instances.
        min_similarity (float): Minimum mutual edge percent.
    
    Returns:
        list: Groups of exposure paths.
    """
    groups = []
    last_exposure = next(exposures)
    current_group = [exposure_paths[0]]
    for path, exposure in zip(exposure_paths[1:], exposures):
        similarity = _mutual_edges(last_exposure, exposure)
        if (similarity < min_similarity):
            groups.append(current_group)
            current_group = []
        current_group.append(path)
        last_exposure = exposure
    groups.append(current_group)
    return groups

def _mutual_edges (image_a: ndarray, image_b: ndarray) -> float:
    """
    Compute the proportion of mutual edges between two images.
    
    Parameters:
        image_a (ndarray): First image.
        image_b (ndarray): Second image.
    
    Returns:
        float: Proportion of mutual edges, in range [0., 1.].
    """
    # Equalize
    image_a, image_b = normalize_exposures(image_a, image_b)
    # Compute edge intersection
    edges_a, edges_b = _extract_edges(image_a), _extract_edges(image_b)
    edges_intersection = edges_a & edges_b
    # Compute ratios
    ratio_a = edges_intersection[edges_a > 0].sum() / edges_a[edges_a > 0].sum()
    ratio_b = edges_intersection[edges_b > 0].sum() / edges_b[edges_b > 0].sum()
    return max(ratio_a, ratio_b)

def _extract_edges (image: ndarray) -> ndarray:
    """
    Extract edges in an image.
    
    Parameters:
        image (ndarray): Input image.
    
    Returns:
        ndarray: Edge image bitmap.
    """
    image = GaussianBlur(image, (5, 5), 0)
    edges = Canny(image, 50, 150)
    return edges