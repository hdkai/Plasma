# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from dateutil.parser import parse as parse_datetime
from exifread import process_file
from io import BytesIO
from PIL import Image
from numpy import asarray, float64, interp, ndarray, uint8, unique
from rawpy import imread as rawread, HighlightMode, Params, ThumbFormat
from typing import Tuple

def normalize_exposures (image_a: ndarray, image_b: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Normalize two exposures to have similar histograms.

    This method preserves metadata on the output images.
    
    Parameters:
        image_a (ndarray): First image.
        image_b (ndarray): Second image.
    
    Returns:
        tuple: Normalized exposures.
    """
    # Match histograms
    std_a = image_a.std()
    std_b = image_b.std()
    input, target = (image_a, image_b) if std_a < std_b else (image_b, image_a)
    matched = _match_histogram(input, target)
    return matched, target

def _match_histogram (input: ndarray, target: ndarray) -> ndarray:
    """
    Match the histogram of an input image to that of a target image.
    
    Parameters:
        input (ndarray): Input image.
        target (ndarray): Target image.
    
    Returns:
        ndarray: Histogram-matched input image.
    """
    # Source: https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    s_values, bin_idx, s_counts = unique(input.ravel(), return_inverse=True, return_counts=True)
    t_values, t_counts = unique(target.ravel(), return_counts=True)
    s_quantiles = s_counts.cumsum().astype(float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = t_counts.cumsum().astype(float64)
    t_quantiles /= t_quantiles[-1]
    interp_t_values = interp(s_quantiles, t_quantiles, t_values)
    result = interp_t_values[bin_idx].reshape(input.shape).astype(uint8)
    return result