# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from concurrent.futures import ThreadPoolExecutor
from cv2 import Canny, GaussianBlur, cvtColor, imdecode, imread, resize, COLOR_RGB2GRAY, INTER_AREA
from dateutil.parser import parse
from exifread import process_file
from numpy import absolute, float64, frombuffer, interp, maximum, ndarray, uint8, unique
from rawpy import imread as rawread, HighlightMode, Params, ThumbFormat
from typing import Iterable, List, Tuple

from .raster import is_raster_format
from .raw import is_raw_format

def group_exposures (exposure_paths: List[str], min_similarity=0.35, workers=4) -> List[List[str]]:
    """
    Group a set of exposures by content.

    Parameters:
        exposure_paths (list): Paths to exposures to group.
        min_similarity (float). Minimum edge similarity for images to be considered in the same group. Should be in range [0., 1.].
    
    Returns:
        list: Groups of exposure paths.
    """
    # Check
    if not exposure_paths:
        return []
    # Sort by EXIF timestamp
    exposure_paths = sorted(exposure_paths, key=lambda path: _get_timestamp(path))
    # Load all exposures into memory, thread this, `map` preserves order
    with ThreadPoolExecutor(max_workers=workers) as executor:
        exposures = executor.map(_load_exposure, exposure_paths)
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
    image_a, image_b = _equalize_exposures(image_a, image_b)
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

def _equalize_exposures (image_a, image_b) -> Tuple[ndarray, ndarray]:
    """
    Equalize two exposures to have similar histograms.
    
    Parameters:
        image_a (ndarray): First image.
        image_b (ndarray): Second image.
    
    Returns:
        tuple: Images with equalized exposures.
    """
    std_a, std_b = image_a.std(), image_b.std()
    input, target = (image_a, image_b) if std_a < std_b else (image_b, image_a)
    matched = _match_histogram(input, target).astype(uint8)
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
    result = interp_t_values[bin_idx].reshape(input.shape)
    return result

def _get_timestamp (path: str) -> float:
    """
    Get the image timestamp from its EXIF metadata.
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
    if time is not None:
        return parse(str(time)).timestamp()
    else:
        return -1

def _load_exposure (image_path: str) -> ndarray:
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
    # Downsample # CHECK # Less downsampling means higher sensitivity to unalignment, good?
    image = resize(image, (512, 512), interpolation=INTER_AREA)
    return image