# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from typing import List

def cascade_group (exposure_paths: List[str], max_delta_time=4.) -> List[List[str]]: # INCOMPLETE
    """
    Group a set of exposures using cascading heuristics.

    First, exposures are checked for similarity using timestamp proximity.
    If a demarcation is found, the exposures are then checked for similarity using feature grouping.

    Parameters:
        exposure_paths (list): Paths to exposures to group.
        max_delta_time (float): Maximum exposure time difference for separating exposures, in seconds.
    
    Returns:
        list: Groups of exposure paths.
    """
    # Check based on timestamps, get similarity array. # INCOMPLETE # Each module should export a similarity function.
    # At bounds (sim[i] == False), check for feature similarity. # INCOMPLETE # Each module should also export this.
    pass