from __future__ import annotations

from dataclasses import dataclass

import heapq
import numpy as np

from .geometry import (
    Direction2D,
    Point2D,
    dual_intersection_2d,
)
from statistics import ProjectionStats2D, skew_from_median_point


@dataclass
class SkewDirection:
    """Result record storing a candidate high-skew direction."""

    direction: Direction2D
    skew_value: float


def normalize_direction_l1(v):
    """Normalize direction using L1 norm (sum of coordinates), matching official implementation.
    
    Official code: `normalize_vector(vector) = vector / sum(vector)`
    """
    v = np.asarray(v, dtype=float)
    if v.shape != (2,):
        raise ValueError(f"Expected 2D vector, got shape {v.shape}")
    s = float(np.sum(v))
    if abs(s) < 1e-10:
        raise ValueError("Cannot normalize zero vector (sum is zero)")
    u = v / s
    return Direction2D(dx=float(u[0]), dy=float(u[1]))


def polar_angle_atan(x, y):
    """Compute polar angle using atan(y/x), matching official implementation.
    
    Official code: `np.arctan(x[1] / x[0])`
    Note: This can cause division by zero when x = 0, but we handle it.
    """
    if abs(x) < 1e-10:
        # Handle division by zero: when x=0, angle is pi/2 or 3*pi/2
        if y > 0:
            return np.pi / 2.0
        elif y < 0:
            return 3.0 * np.pi / 2.0
        else:
            return 0.0  # (0, 0) case
    return float(np.arctan(y / x))


def sort_points_by_polar_atan(points):
    """Sort points by polar angle using atan(y/x), matching official implementation.
    
    Official code: `sorted(keys, key=lambda x: np.arctan(x[1] / x[0]))`
    """
    pts = [(float(x), float(y)) for (x, y) in points]
    return sorted(pts, key=lambda p: polar_angle_atan(p[0], p[1]))


def _build_projection_stats_official_style(points):
    """Build point list and precomputed statistics, matching official preprocessing.
    
    Official code:
    1. Shifts points: `points[0] -= points[0].min()`, `points[1] -= points[1].min()`
    2. Computes centered points: `q = points - mean(points, axis=0)`
    3. Uses shifted points for all calculations
    """
    # Convert to numpy array for easier manipulation
    arr = np.array([[float(x), float(y)] for (x, y) in points], dtype=float)
    if arr.shape[0] == 0:
        raise ValueError("At least one point is required")
    
    # Official preprocessing: shift to start from 0
    arr[:, 0] = arr[:, 0] - arr[:, 0].min()
    arr[:, 1] = arr[:, 1] - arr[:, 1].min()
    
    # Convert back to Point2D list
    pts = [Point2D(float(arr[i, 0]), float(arr[i, 1])) for i in range(arr.shape[0])]
    
    # Compute statistics from shifted points
    stats = ProjectionStats2D.from_points(pts)
    return pts, stats


def _get_next_median_official_style(intersection, candidate_points, prev_median):
    """Find the next median point, using official's atan-based sorting.
    
    Official code:
        candidate_points = sorted(candidate_points, key=lambda x: np.arctan(x[1] / x[0]))
        index = candidate_points.index(prev_median)
        return candidate_points[len(candidate_points) - index - 1]
    """
    if prev_median not in candidate_points:
        return prev_median
    
    # Sort using atan(y/x) as in official code
    sorted_candidates = sorted(
        candidate_points,
        key=lambda p: polar_angle_atan(p.x, p.y),
    )
    
    try:
        index = sorted_candidates.index(prev_median)
    except ValueError:
        return sorted_candidates[0]
    
    symmetric_index = len(sorted_candidates) - index - 1
    return sorted_candidates[symmetric_index]


def _enumerate_intersections_with_points_official_style(points):
    """Enumerate dual intersections, matching official's three-step process.
    
    Official code (`_get_intersects`):
    1. Enumerate ALL intersections (no pre-filtering)
    2. Sort by polar angle using `np.arctan(y/x)`
    3. Filter: `filter(lambda x: x[1] > 0 and x[0] > 0, ...)` (first quadrant only)
    """
    n = len(points)
    intersections_dict = {}
    
    # Step 1: Enumerate all intersections (no pre-filtering)
    for i in range(n - 1):
        for j in range(i + 1, n):
            p_i = points[i]
            p_j = points[j]
            try:
                x = dual_intersection_2d(p_i, p_j)
                if np.all(np.isfinite(x)):
                    intr_key = (round(float(x[0]), 5), round(float(x[1]), 5))  # Official uses 5 decimals
                    if intr_key not in intersections_dict:
                        intersections_dict[intr_key] = set()
                    intersections_dict[intr_key].add(p_i)
                    intersections_dict[intr_key].add(p_j)
            except np.linalg.LinAlgError:
                continue
    
    # Step 2: Sort by polar angle using atan (matching official)
    intersections_list = sort_points_by_polar_atan(list(intersections_dict.keys()))
    
    # Step 3: Filter to first quadrant only (matching official)
    intersections_list = [
        intr for intr in intersections_list
        if intr[1] > 0 and intr[0] > 0
    ]
    
    # Update dictionary to only include filtered intersections
    filtered_dict = {intr: intersections_dict[intr] for intr in intersections_list if intr in intersections_dict}
    
    return intersections_list, filtered_dict


def _build_point_intersection_map_official_style(points, intersections_dict):
    """Build point-to-intersections mapping, using atan-based sorting."""
    point_intersections = {}
    
    for intr, point_set in intersections_dict.items():
        for point in point_set:
            if point not in point_intersections:
                point_intersections[point] = []
            point_intersections[point].append(intr)
    
    # Sort using atan (matching official)
    for point in point_intersections:
        point_intersections[point] = sort_points_by_polar_atan(point_intersections[point])
    
    return point_intersections


def get_angle_official_style(vec1, vec2):
    """Compute angle between two vectors using arccos, matching official implementation.
    
    Official code:
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        return np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))
    """
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)
    return float(np.arccos(np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0)))


def ray_sweeping_2d_official_style(
    points,
    top_k=10,
    min_angle_step=np.pi / 10.0,  # Official default: pi/10
    vector_transfer=None,
):
    """2D Ray-sweeping with official-style implementation.
    
    This version matches the official implementation's methods:
    - L1 normalization (divide by sum)
    - atan(y/x) for polar angle sorting
    - Data preprocessing with min-shift
    - First-quadrant intersection filtering
    - Initial direction from (1/x_median, 0)
    - Exact termination check (x == 0)
    
    Args:
        points: Iterable of (x, y) coordinate pairs in the primal space.
        top_k: Number of top high-skew directions to return.
        min_angle_step: Minimum angular step between sampled directions (default: pi/10).
        vector_transfer: Optional function to transform direction vectors
            (e.g., for rotated point sets: `lambda x: tuple([-x[1], x[0]])`).
            If None, uses identity: `lambda x: tuple([x[0], x[1]])`.
    
    Returns:
        A list of `SkewDirection` objects, sorted by skew value (highest first).
    """
    
    if vector_transfer is None:
        vector_transfer = lambda x: tuple([x[0], x[1]])
    
    pts, stats = _build_projection_stats_official_style(points)
    if len(pts) < 2:
        return []

    # Initialize median point (x-coordinate median, as in official)
    sorted_by_x = sorted(pts, key=lambda p: p.x)
    median_point = sorted_by_x[len(sorted_by_x) // 2]

    return _ray_sweeping_2d_official_style(
        pts, stats, median_point, top_k, min_angle_step, vector_transfer
    )


def _ray_sweeping_2d_official_style(
    pts,
    stats,
    initial_median,
    top_k,
    min_angle_step,
    vector_transfer,
):
    """Ray-sweeping with official-style implementation details."""
    
    # Enumerate intersections with point relationships (official style)
    intersections, intersections_dict = _enumerate_intersections_with_points_official_style(pts)
    if not intersections:
        return []

    # Build point-to-intersections mapping
    point_intersections = _build_point_intersection_map_official_style(pts, intersections_dict)

    # Official initialization: start from (1/x_median, 0)
    # This represents the direction along the median point's dual line on the X-axis
    current_median = initial_median
    current_point = initial_median

    # Get intersections for the starting point
    if current_point not in point_intersections:
        return []

    current_intersections = point_intersections[current_point]
    
    # Official starts from (1/x_median, 0) - this is the initial direction
    # Matching official: we compute the initial direction but don't push to heap
    # until we meet the angle condition with the next direction
    heap = []
    last_vec = None
    if abs(initial_median.x) >= 1e-10:
        # Compute initial direction from (1/x_median, 0)
        start_dir_vec = np.array([1.0 / initial_median.x, 0.0], dtype=float)
        start_dir_raw = normalize_direction_l1(start_dir_vec)
        # Apply vector_transfer for consistency (though we don't push this direction)
        start_dir_transferred = vector_transfer(start_dir_raw.as_array())
        start_dir_normalized = normalize_direction_l1(np.array(start_dir_transferred))
        # Set last_vec but don't push (matching official: if last_vec is None, just set it)
        last_vec = start_dir_normalized.as_array()
    
    # Start traversal from the first intersection for this point
    intersection_idx = 0

    visited_intersections = set()

    # Traverse the k-level arrangement
    while intersection_idx < len(current_intersections):
        intersection = current_intersections[intersection_idx]

        # Skip if already processed
        if intersection in visited_intersections:
            intersection_idx += 1
            continue

        visited_intersections.add(intersection)

        # Compute direction from intersection (using L1 normalization)
        # Official: computes direction from intersection, normalizes, then uses for skew calculation
        direction_raw = normalize_direction_l1(np.array([intersection[0], intersection[1]], dtype=float))
        skew_vector = direction_raw.as_array()  # Use for angle check and skew calculation
        
        # Apply vector_transfer for storage (matching official: only affects stored direction, not skew calc)
        # Official stores: vector_transfer(tuple(skew_vector_start))
        direction_transferred_tuple = vector_transfer(tuple(skew_vector))
        direction = normalize_direction_l1(np.array(direction_transferred_tuple))  # Final direction for storage

        # Official logic: check angle step and push to heap
        # Matching official: if last_vec is not None and angle > epsilon, push
        # If last_vec is None, set it but don't push (first direction)
        # Note: skew calculation uses original direction (before vector_transfer),
        # but stored direction uses vector_transfer (matching official)
        if last_vec is not None:
            ang = get_angle_official_style(skew_vector, last_vec)
            if ang > min_angle_step:
                # Calculate skew using original direction (before vector_transfer)
                skew_val = skew_from_median_point(stats, direction_raw, current_median)
                # Store direction with vector_transfer applied
                cand = SkewDirection(direction=direction, skew_value=skew_val)
                heapq.heappush(heap, (-skew_val, cand))
                last_vec = skew_vector
        else:
            # First direction: set last_vec but don't push (matching official)
            last_vec = skew_vector

        # Update median point at this intersection
        if intersection in intersections_dict:
            candidate_points = intersections_dict[intersection]
            new_median = _get_next_median_official_style(intersection, candidate_points, current_median)
            
            if new_median != current_median:
                current_median = new_median
                if new_median in point_intersections:
                    new_intersections = point_intersections[new_median]
                    try:
                        intersection_idx = new_intersections.index(intersection) + 1
                    except ValueError:
                        intersection_idx = 0
                    current_point = new_median
                    current_intersections = new_intersections
                    continue

        intersection_idx += 1

        # Official termination: exact check for Y-axis
        if intersection[0] == 0.0:  # Exact equality as in official
            break

    results = []
    while heap and len(results) < top_k:
        _, cand = heapq.heappop(heap)
        results.append(cand)

    return results

