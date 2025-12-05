"""
Reimplementation of the 2D Minoria (U3M) mining algorithms.

This package provides a clean, self-contained implementation that follows
the paper “Mining the Minoria: Unknown, Under-represented, and
Under-performing Minority Groups”, while only taking high-level
inspiration from the reference code in `Mining_U3Ms-main/utils`.

The code here is structured for clarity and extensibility and does not
copy any implementation details from the original repository.
"""

from .algorithms.geometry import (
    Point2D,
    Direction2D,
    dual_intersection_2d,
    normalize_direction,
    sort_points_by_polar_angle,
)
from .algorithms.statistics import ProjectionStats2D, skew_from_median_point

__all__ = [
    "Point2D",
    "Direction2D",
    "dual_intersection_2d",
    "normalize_direction",
    "sort_points_by_polar_angle",
    "ProjectionStats2D",
    "skew_from_median_point",
]


