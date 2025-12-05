from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Point2D:
    """Simple 2D point.

    This is just a typed wrapper around a 2D numpy array to make the
    code dealing with primal points more explicit.
    """

    x: float
    y: float

    def as_array(self):
        return np.array([self.x, self.y], dtype=float)


@dataclass(frozen=True)
class Direction2D:
    """Unit direction vector in R^2."""

    dx: float
    dy: float

    def as_array(self):
        return np.array([self.dx, self.dy], dtype=float)


def normalize_direction(v):
    """Return a unit-length direction for a raw 2D numpy vector.

    DIFFERENCE FROM OFFICIAL:
    - Official: Uses L1 normalization (divides by sum of coordinates)
      `normalize_vector(vector) = vector / sum(vector)`
    - This implementation: Uses L2 normalization (Euclidean norm)
      `normalize_direction(v) = v / ||v||_2`
    
    This is more consistent with the theoretical presentation in the paper
    (unit vectors on the sphere), but produces different numerical values
    than the official implementation. The direction is the same, but the
    magnitude differs.
    """

    v = np.asarray(v, dtype=float)
    if v.shape != (2,):
        raise ValueError(f"Expected 2D vector, got shape {v.shape}")
    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise ValueError("Cannot normalize zero vector")
    u = v / norm
    return Direction2D(dx=float(u[0]), dy=float(u[1]))


def dual_intersection_2d(p1, p2):
    """Compute the intersection of two dual hyperplanes in R^2.

    The paper uses the dual transform

        d(t):  t_1 x_1 + t_2 x_2 = 1

    for a primal point t = (t_1, t_2).  The intersection of d(p1) and
    d(p2) is the unique point x such that:

        [p1.x  p1.y] [x1] = [1]
        [p2.x  p2.y] [x2] = [1]

    which we solve as a 2x2 linear system.
    """

    a = np.array([[p1.x, p1.y], [p2.x, p2.y]], dtype=float)
    b = np.ones(2, dtype=float)
    sol = np.linalg.solve(a, b)
    return sol  # shape (2,)


def polar_angle(v):
    """Return the polar angle of a 2D vector in [0, 2π).

    DIFFERENCE FROM OFFICIAL:
    - Official: Uses `np.arctan(y / x)` which can cause division by zero
      when x = 0, and doesn't handle all quadrants correctly
    - This implementation: Uses `np.arctan2(y, x)` which is numerically
      stable and correctly handles all quadrants
    
    The official code in `GeoUtility.sort_points_by_polar` uses:
        `sorted(keys, key=lambda x: np.arctan(x[1] / x[0]))`
    This can fail when x[0] = 0, and doesn't distinguish between
    quadrants II/III vs I/IV.
    """

    v = np.asarray(v, dtype=float)
    if v.shape != (2,):
        raise ValueError(f"Expected 2D vector, got shape {v.shape}")
    ang = float(np.arctan2(v[1], v[0]))
    if ang < 0.0:
        ang += 2.0 * np.pi
    return ang


def sort_points_by_polar_angle(points):
    """Sort 2D points by their polar angle around the origin.

    This is used when we need an ordered traversal of intersection
    points around the origin, as required by the ray-sweeping view of
    the k-level of the arrangement.
    """

    pts = [(float(x), float(y)) for (x, y) in points]
    return sorted(pts, key=lambda p: polar_angle(np.array(p, dtype=float)))



