# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from cv2 import cvtColor, imdecode, imread, resize, COLOR_RGB2GRAY, INTER_AREA
from dateutil.parser import parse as parse_datetime
from exifread import process_file
from numpy import float64, frombuffer, interp, ndarray, uint8, unique
from rawpy import imread as rawread, HighlightMode, Params, ThumbFormat
from typing import Tuple

from ..raster import is_raster_format
from ..raw import is_raw_format

def exposure_timestamp (path: str) -> float:
    """
    Get the exposure timestamp from its EXIF metadata.
    If the required EXIF dictionary or tag is not present, -1 will be returned.
    
    Parameters:
        path (str): Path to exposure.
    
    Returns:
        float: Image timestamp.
    """
    DATETIME_ORIGINAL = "EXIF DateTimeOriginal"
    with open(path, "rb") as file:
        tags = process_file(file, stop_tag=DATETIME_ORIGINAL, details=False)
    time = tags.get(DATETIME_ORIGINAL)
    return parse_datetime(str(time)).timestamp() if time is not None else -1

def load_exposure (image_path: str) -> ndarray:
    """
    Load an exposure into memory.

    For RAW files, this function will try to load the thumbnail.
    But if no thumbnail is available, the RAW is fully demosaiced.
    
    Parameters:
        image_path (str): Path to exposure.
    
    Returns:
        ndarray: Loaded exposure.
    """
    # If raster format, load directly
    if is_raster_format(image_path):
        image = imread(image_path, 0)
    # If RAW, check for thumbnail
    elif is_raw_format(image_path):
        with rawread(image_path) as raw:
            # Load thumbnail
            try:
                thumb = raw.extract_thumb()
                if thumb.format == ThumbFormat.JPEG:
                    image_data = frombuffer(thumb.data, dtype=uint8)
                    image = imdecode(image_data, 0)
                elif thumb.format == ThumbFormat.BITMAP:
                    image = cvtColor(thumb.data, COLOR_RGB2GRAY)
            # Demosaic RAW
            except:
                params = Params(
                    half_size=True,
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=8,
                    user_sat=9000,
                    exp_shift=1.,
                    exp_preserve_highlights=1.,
                    highlight_mode=HighlightMode.Clip,
                )
                image = raw.postprocess(params=params)
                image = cvtColor(image, COLOR_RGB2GRAY)
    # Downsample
    image = resize(image, (512, 512), interpolation=INTER_AREA)
    return image

def normalize_exposures (image_a, image_b) -> Tuple[ndarray, ndarray]:
    """
    Normalize two exposures to have similar histograms.
    
    Parameters:
        image_a (ndarray): First image.
        image_b (ndarray): Second image.
    
    Returns:
        tuple: Normalized exposures.
    """
    std_a, std_b = image_a.std(), image_b.std()
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