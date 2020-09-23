# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from concurrent.futures import ThreadPoolExecutor
from cv2 import DescriptorMatcher_create, findHomography, ORB_create, DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING, RANSAC
from numpy import array, ndarray, sqrt
from numpy.linalg import eig
from typing import List

from .common import exposure_timestamp, load_exposure, normalize_exposures

def group_exposures_by_features (exposure_paths: List[str], workers=8) -> List[List[str]]:
    """
    Group a set of exposures by their visual features.

    Parameters:
        exposure_paths (list): Paths to exposures to group.
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
        exposures = executor.map(lambda path: load_exposure(path, 2048), exposure_paths)
        exposures = list(exposures) # consume immediately
        bounds = executor.map(lambda pair: _images_aligned(*pair), [(exposures[i], exposures[i+1]) for i in range(len(exposures) - 1)])
    # Group
    groups = []
    current_group = [exposure_paths[0]]
    for i, same_group in enumerate(bounds):
        if not same_group:
            groups.append(current_group)
            current_group = []
        current_group.append(exposure_paths[i+1])
    groups.append(current_group)
    return groups

def _images_aligned (image_a: ndarray, image_b: ndarray) -> bool:
    """
    Check if two exposures are aligned.

    Parameters:
        image_a (ndarray): First image.
        image_b (ndarray): Second image.

    Returns:
        bool: Whether both images are aligned.
    """
    keypoints_a, keypoints_b, matches = _compute_matches(image_a, image_b)
    coefficient = _compute_alignment_coefficient(keypoints_a, keypoints_b, matches)
    return coefficient < 1e-2

def _compute_matches (image_a: ndarray, image_b: ndarray):
    """
    Compute feature matches between two images.

    Parameters:
        image_a (ndarray): First image.
        image_b (ndarray): Second image.

    Returns:
        tuple: Keypoints from first image, keypoints from second image, and matches between them.
    """
    # Normalize
    image_a, image_b = normalize_exposures(image_a, image_b)
    # Detect features
    orb = ORB_create(1000)
    keypoints_a, descriptors_a = orb.detectAndCompute(image_a, None)
    keypoints_b, descriptors_b = orb.detectAndCompute(image_b, None)
    # Match features
    matcher = DescriptorMatcher_create(DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors_a, descriptors_b, None)
    # Extract top k
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:int(len(matches) * 0.15)]
    # Return
    return keypoints_a, keypoints_b, matches

def _compute_alignment_coefficient (keypoints_a: ndarray, keypoints_b: ndarray, matches: ndarray) -> float:
    """
    Compute the alignment coefficient between two keypoints.

    Parameters:
        keypoints_a (ndarray): First set of keypoints.
        keypoints_b (ndarray): Second set of keypoints.
        matches (ndarray): Matches between keypoints.

    Returns:
        float: Alignment coefficient.
    """
    # Compute homography
    points_a = array([keypoints_a[match.queryIdx].pt for match in matches])
    points_b = array([keypoints_b[match.trainIdx].pt for match in matches])
    H, _ = findHomography(points_a, points_b, RANSAC)
    # Compute alignment coefficient
    singular_values, _ = eig(H.T * H)
    induced_norm = sqrt(singular_values.max())
    coefficient = abs(induced_norm - 1.)
    # Return
    return coefficient