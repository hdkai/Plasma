# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from cv2 import Canny, GaussianBlur, resize, INTER_AREA
from PIL import Image
from numpy import asarray, ndarray
from typing import Callable

from .utility import normalize_exposures

def edge_similarity (min_similarity: float=0.35) -> Callable[[Image.Image, Image.Image], bool]:
    """
    Create an edge-based similarity function.

    Parameters:
        min_similarity (float). Minimum edge similarity for images to be considered in the same group. Should be in range [0., 1.].

    Returns:
        callable: Pairwise similarity function returning a boolean.
    """
    def similarity_fn (image_a: Image.Image, image_b: Image.Image) -> bool:
        # Downsample
        MIN_DIM = 1024
        scale = max(MIN_DIM / image_a.width, MIN_DIM / image_a.height)
        image_a = resize(asarray(image_a), (0, 0), fx=scale, fy=scale, interpolation=INTER_AREA)
        image_b = resize(asarray(image_b), (0, 0), fx=scale, fy=scale, interpolation=INTER_AREA)
        # Compute edge intersection
        image_a, image_b = normalize_exposures(image_a, image_b)
        edges_a, edges_b = _extract_edges(image_a), _extract_edges(image_b)
        edges_intersection = edges_a & edges_b
        # Compute ratios
        ratio_a = edges_intersection[edges_a > 0].sum() / edges_a[edges_a > 0].sum()
        ratio_b = edges_intersection[edges_b > 0].sum() / edges_b[edges_b > 0].sum()
        ratio = max(ratio_a, ratio_b)
        return ratio > min_similarity
    return similarity_fn

def _extract_edges (image: Image.Image) -> ndarray:
    """
    Extract edges in an image.
    
    Parameters:
        image (PIL.Image): Input image.
    
    Returns:
        ndarray: Edge bitmap.
    """
    image = GaussianBlur(image, (5, 5), 0)
    edges = Canny(image, 50, 150)
    return edges