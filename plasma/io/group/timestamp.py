# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from numpy import array
from sklearn.cluster import DBSCAN
from typing import List

from .common import exposure_timestamp

def group_exposures_by_timestamp (exposure_paths: List[str], max_delta=8.) -> List[List[str]]:
    """
    Group a set of exposures by their visual features.

    Parameters:
        exposure_paths (list): Paths to exposures to group.
        max_delta (float): Maximum exposure time difference for separating exposures, in seconds.
    
    Returns:
        list: Groups of exposure paths.
    """
    # Check
    if not exposure_paths:
        return []
    # Trivial case
    if len(exposure_paths) == 1:
        return [exposure_paths]
    # Get timestamps
    timestamps = [exposure_timestamp(path) for path in exposure_paths]
    timestamps = [time if time >= 0 else -(i + 1) * max_delta * 10 for i, time in enumerate(timestamps)]
    # Cluster
    timestamps = array([timestamps]).T
    timestamps -= timestamps.min()
    dbscan = DBSCAN(eps=max_delta, min_samples=1, p=1)
    labels = dbscan.fit_predict(timestamps)
    groups = [[path for idx, path in enumerate(exposure_paths) if labels[idx] == i] for i in range(labels.max() + 1)]
    return groups