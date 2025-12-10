from __future__ import annotations

from dataclasses import dataclass

import heapq
import numpy as np

from .geometry import (
    Direction2D,
    Point2D,
    dual_intersection_2d,
)
from .statistics import ProjectionStats2D, skew_from_median_point
from .ray_sweeping_2d_official import LinkedList, MedianRegion


@dataclass
class SkewDirection:
    """Result record storing a candidate high-skew direction."""

    direction: Direction2D
    skew_value: float


def normalize_direction_l1(v):
    """Normalize direction using L1 norm (sum of coordinates)."""
    v = np.asarray(v, dtype=float)
    if v.shape != (2,):
        raise ValueError(f"Expected 2D vector, got shape {v.shape}")
    s = float(np.sum(v))
    if abs(s) < 1e-10:
        raise ValueError("Cannot normalize zero vector (sum is zero)")
    u = v / s
    return Direction2D(dx=float(u[0]), dy=float(u[1]))


def normalize_direction_l2(v):
    """Normalize direction using L2 norm."""
    v = np.asarray(v, dtype=float)
    if v.shape != (2,):
        raise ValueError(f"Expected 2D vector, got shape {v.shape}")
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("Cannot normalize zero vector (norm is zero)")
    u = v / n
    return Direction2D(dx=float(u[0]), dy=float(u[1]))


def _is_zero_vector_for_normalization(v, normalize_fn):
    """Check if vector is zero for the given normalization function.
    
    For L1 normalization: checks if sum is near zero.
    For L2 normalization: checks if norm is near zero.
    """
    v = np.asarray(v, dtype=float)
    if normalize_fn == normalize_direction_l1:
        return abs(np.sum(v)) < 1e-10
    else:  # L2 normalization
        return np.linalg.norm(v) < 1e-10


def polar_angle_atan(x, y):
    """Compute polar angle using atan(y/x) with zero-division handling."""
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
    """Sort points by polar angle using atan(y/x)."""
    pts = [(float(x), float(y)) for (x, y) in points]
    return sorted(pts, key=lambda p: polar_angle_atan(p[0], p[1]))

# Aliases for linked-list path
_polar_angle_atan = polar_angle_atan
_sort_points_by_polar_atan = sort_points_by_polar_atan


def _build_projection_stats_official_style(points, use_min_shift: bool = True):
    """Build point list and statistics with optional min-shift preprocessing."""
    # Convert to numpy array for easier manipulation
    arr = np.array([[float(x), float(y)] for (x, y) in points], dtype=float)
    if arr.shape[0] == 0:
        raise ValueError("At least one point is required")
    
    # Official preprocessing: shift to start from 0 (can be disabled)
    if use_min_shift:
        arr[:, 0] = arr[:, 0] - arr[:, 0].min()
        arr[:, 1] = arr[:, 1] - arr[:, 1].min()
    
    # Convert back to Point2D list
    pts = [Point2D(float(arr[i, 0]), float(arr[i, 1])) for i in range(arr.shape[0])]
    
    # Compute statistics from shifted points
    stats = ProjectionStats2D.from_points(pts)
    return pts, stats


def _get_next_median_official_style(intersection, candidate_points, prev_median):
    """Find next median point using atan-based symmetric selection."""
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
    """Enumerate dual intersections: enumerate all, sort by atan, filter first quadrant."""
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
    """Build point-to-intersections mapping with atan-based sorting."""
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
    """Compute angle between two vectors using arccos."""
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)
    return float(np.arccos(np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0)))


def ray_sweeping_2d_official_style(
    points,
    top_k=10,
    min_angle_step=np.pi / 10.0,
    vector_transfer=None,
    use_linkedlist: bool = False,
    use_first_intersection_init: bool = False,
    use_min_shift: bool = True,
    enable_vector_transfer: bool = True,
    use_l1_norm: bool = True,
):
    """
    2D Ray-sweeping with configurable official-style implementation.
    
    Supports multiple variants through parameters:
    - Data structure: LinkedList (official) or dict+list
    - Initialization: Official default (1/x_median, 0) or first-intersection
    - Preprocessing: Optional min-shift, optional vector_transfer
    - Normalization: L1 (official) or L2 (configurable)
    
    Args:
        points: Iterable of (x, y) coordinate pairs.
        top_k: Number of top directions to return.
        min_angle_step: Minimum angular step (default: pi/10).
        vector_transfer: Function to transform direction vectors (default: identity).
        use_linkedlist: Use LinkedList data structure (official).
        use_first_intersection_init: Start from first intersection (ray_sweeping_2d style).
        use_min_shift: Apply min-shift preprocessing.
        enable_vector_transfer: Apply vector_transfer mapping.
        use_l1_norm: Use L1 normalization (official). If False, use L2.
    
    Returns:
        List of SkewDirection objects, sorted by skew (highest first).
    """
    
    if not enable_vector_transfer:
        vector_transfer = lambda x: (x[0], x[1])
    elif vector_transfer is None:
        vector_transfer = lambda x: tuple([x[0], x[1]])

    normalize_fn = normalize_direction_l1 if use_l1_norm else normalize_direction_l2
    
    pts, stats = _build_projection_stats_official_style(points, use_min_shift=use_min_shift)
    if len(pts) < 2:
        return []

    # Initialize median point (x-coordinate median, as in official)
    sorted_by_x = sorted(pts, key=lambda p: p.x)
    median_point = sorted_by_x[len(sorted_by_x) // 2]

    if use_linkedlist:
        return _ray_sweeping_2d_official_style_linkedlist(
            pts, stats, median_point, top_k, min_angle_step, vector_transfer, use_first_intersection_init, normalize_fn
        )
    else:
        return _ray_sweeping_2d_official_style(
            pts, stats, median_point, top_k, min_angle_step, vector_transfer, use_first_intersection_init, normalize_fn
        )

# LinkedList path (official data structure)
def _build_line_intersects_linkedlist(intersect_keys, intersects):
    line_intersects = {}
    for key in intersect_keys:
        occurs = intersects[key]
        links = [LinkedList(key, [], point, None) for point in occurs]
        for link in links:
            link.neighbours = links
        for i, point in enumerate(occurs):
            if point in line_intersects:
                start = line_intersects[point]
                LinkedList.append_to_end(start, links[i])
            else:
                line_intersects[point] = links[i]
    return line_intersects

def _ray_sweeping_2d_official_style_linkedlist(
    pts,
    stats,
    initial_median,
    top_k,
    min_angle_step,
    vector_transfer,
    use_first_intersection_init: bool,
    normalize_fn,
):
    # Enumerate intersections (official-style order and rounding)
    intersects = {}
    n = len(pts)
    for i in range(n - 1):
        for j in range(i + 1, n):
            p_i = pts[i]
            p_j = pts[j]
            try:
                x = dual_intersection_2d(p_i, p_j)
                if np.all(np.isfinite(x)):
                    intr = (round(float(x[0]), 5), round(float(x[1]), 5))
                    if intr[0] <= 0.0 or intr[1] <= 0.0:
                        continue
                    if intr in intersects:
                        intersects[intr].add((p_i.x, p_i.y))
                        intersects[intr].add((p_j.x, p_j.y))
                    else:
                        intersects[intr] = {(p_i.x, p_i.y), (p_j.x, p_j.y)}
            except np.linalg.LinAlgError:
                continue

    intersect_keys = _sort_points_by_polar_atan(list(intersects.keys()))
    if not intersect_keys:
        return []

    line_intersects = _build_line_intersects_linkedlist(intersect_keys, intersects)

    # Median init
    first_median = initial_median
    last_vec: np.ndarray | None = None

    start = LinkedList((1.0 / first_median.x, 0.0), [], (first_median.x, first_median.y), None)
    end = line_intersects.get((first_median.x, first_median.y))
    if end is None:
        if not line_intersects:
            return []
        first_median_tuple, end = next(iter(line_intersects.items()))
        first_median = Point2D(float(first_median_tuple[0]), float(first_median_tuple[1]))
        start = LinkedList((1.0 / first_median.x, 0.0), [], (first_median.x, first_median.y), None)
    median_region = MedianRegion(start, end, (first_median.x, first_median.y))

    # Optional: ray_sweeping_2d-style initialization
    if use_first_intersection_init:
        # choose first intersection from intersect_keys for starting direction
        first_intersection = intersect_keys[0]
        initial_dir = normalize_fn(np.array([first_intersection[0], first_intersection[1]]))
        skew_val = skew_from_median_point(stats, initial_dir, Point2D(*median_region.median))
        dir_transferred_tuple = vector_transfer(tuple(initial_dir.as_array()))
        dir_transferred_arr = np.array(dir_transferred_tuple, dtype=float)
        # Check if transferred vector is zero
        if not _is_zero_vector_for_normalization(dir_transferred_arr, normalize_fn):
            dir_stored = normalize_fn(dir_transferred_arr)
            heapq.heappush(heapq_heap := [], (-skew_val, SkewDirection(direction=dir_stored, skew_value=skew_val)))
        else:
            heapq_heap = []
        last_vec = initial_dir.as_array()
        # override heap reference for consistency
        heap = heapq_heap
    else:
        heap: list[tuple[float, SkewDirection]] = []

    while True:
        skew_vector_start = normalize_fn(np.array(median_region.start.point))

        if last_vec is None:
            skew_val = skew_from_median_point(
                stats, skew_vector_start, Point2D(*median_region.median)
            )
            if not use_first_intersection_init:
                dir_transferred_tuple = vector_transfer(tuple(skew_vector_start.as_array()))
                dir_transferred_arr = np.array(dir_transferred_tuple, dtype=float)
                # Check if transferred vector is zero
                if not _is_zero_vector_for_normalization(dir_transferred_arr, normalize_fn):
                    dir_stored = normalize_fn(dir_transferred_arr)
                    heapq.heappush(heap, (-skew_val, SkewDirection(direction=dir_stored, skew_value=skew_val)))
            last_vec = skew_vector_start.as_array()
        else:
            ang = get_angle_official_style(skew_vector_start.as_array(), last_vec)
            if ang > min_angle_step:
                skew_val = skew_from_median_point(
                    stats, skew_vector_start, Point2D(*median_region.median)
                )
                dir_transferred_tuple = vector_transfer(tuple(skew_vector_start.as_array()))
                dir_transferred_arr = np.array(dir_transferred_tuple, dtype=float)
                # Check if transferred vector is zero
                if not _is_zero_vector_for_normalization(dir_transferred_arr, normalize_fn):
                    dir_stored = normalize_fn(dir_transferred_arr)
                    heapq.heappush(heap, (-skew_val, SkewDirection(direction=dir_stored, skew_value=skew_val)))
                last_vec = skew_vector_start.as_array()

        # termination
        if median_region.end.point[0] == 0:
            break

        try:
            current_points = intersects[median_region.end.point]
        except KeyError:
            break

        # median update with atan ordering
        pts_sorted = sorted(current_points, key=lambda x: polar_angle_atan(x[0], x[1]))
        if median_region.median not in pts_sorted:
            break
        idx = pts_sorted.index(median_region.median)
        line_b = pts_sorted[len(pts_sorted) - idx - 1]

        next_neighbour = [n for n in median_region.end.neighbours if n.line == line_b][0]
        if next_neighbour.next is None:
            new_end = LinkedList((0.0, 1.0 / line_b[1]), [], line_b, None)
        else:
            new_end = next_neighbour.next

        median_region = MedianRegion(median_region.end, new_end, line_b)

    results = []
    while heap and len(results) < top_k:
        _, cand = heapq.heappop(heap)
        results.append(cand)
    return results

def _ray_sweeping_2d_official_style(
    pts, stats, initial_median, top_k, min_angle_step, vector_transfer, use_first_intersection_init, normalize_fn
):
    """Ray-sweeping with dict+list data structure."""
    
    # Enumerate intersections
    intersections, intersections_dict = _enumerate_intersections_with_points_official_style(pts)
    if not intersections:
        return []

    # Build point-to-intersections mapping
    point_intersections = _build_point_intersection_map_official_style(pts, intersections_dict)
    
    # Initialize
    current_median = initial_median
    current_point = initial_median
    if current_point not in point_intersections:
        return []
    
    current_intersections = point_intersections[current_point]
    heap: list[tuple[float, SkewDirection]] = []
    last_vec = None
    
    if use_first_intersection_init and current_intersections:
        # ray_sweeping_2d-style: start from first intersection
        first_intersection = current_intersections[0]
        direction_raw = normalize_fn(np.array([first_intersection[0], first_intersection[1]], dtype=float))
        skew_val = skew_from_median_point(stats, direction_raw, current_median)
        direction_transferred_tuple = vector_transfer(tuple(direction_raw.as_array()))
        direction_transferred_arr = np.array(direction_transferred_tuple, dtype=float)
        # Check if transferred vector is zero
        if not _is_zero_vector_for_normalization(direction_transferred_arr, normalize_fn):
            direction_transferred = normalize_fn(direction_transferred_arr)
            heapq.heappush(heap, (-skew_val, SkewDirection(direction=direction_transferred, skew_value=skew_val)))
        last_vec = direction_raw.as_array()
    else:
        if abs(initial_median.x) >= 1e-10:
            start_dir_vec = np.array([1.0 / initial_median.x, 0.0], dtype=float)
            start_dir_raw = normalize_fn(start_dir_vec)
            start_dir_transferred_tuple = vector_transfer(start_dir_raw.as_array())
            start_dir_transferred_arr = np.array(start_dir_transferred_tuple, dtype=float)
            # Check if transferred vector is zero
            if not _is_zero_vector_for_normalization(start_dir_transferred_arr, normalize_fn):
                start_dir_normalized = normalize_fn(start_dir_transferred_arr)
                last_vec = start_dir_normalized.as_array()
            else:
                last_vec = start_dir_raw.as_array()
    
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
        direction_raw = normalize_fn(np.array([intersection[0], intersection[1]], dtype=float))
        skew_vector = direction_raw.as_array()  # Use for angle check and skew calculation
        
        # Apply vector_transfer for storage
        direction_transferred_tuple = vector_transfer(tuple(skew_vector))
        direction_transferred_arr = np.array(direction_transferred_tuple, dtype=float)
        # Check if transferred vector is zero (cannot be normalized)
        if _is_zero_vector_for_normalization(direction_transferred_arr, normalize_fn):
            intersection_idx += 1
            continue
        direction = normalize_fn(direction_transferred_arr)  # Final direction for storage

        if last_vec is not None:
            ang = get_angle_official_style(skew_vector, last_vec)
            if ang > min_angle_step:
                skew_val = skew_from_median_point(stats, direction_raw, current_median)
                cand = SkewDirection(direction=direction, skew_value=skew_val)
                heapq.heappush(heap, (-skew_val, cand))
                last_vec = skew_vector
        else:
            # First direction: set last_vec but don't push
            last_vec = skew_vector
        
        # Update median point
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
        
        # Termination: exact check for Y-axis
        if intersection[0] == 0.0:
            break

    results = []
    while heap and len(results) < top_k:
        _, cand = heapq.heappop(heap)
        results.append(cand)

    return results


