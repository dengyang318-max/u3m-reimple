from __future__ import annotations

from dataclasses import dataclass

import heapq
import numpy as np

from .geometry import (
    Direction2D,
    Point2D,
    dual_intersection_2d,
    normalize_direction,
    polar_angle,
    sort_points_by_polar_angle,
)
from .statistics import ProjectionStats2D, skew_from_median_point


@dataclass
class SkewDirection:
    """Result record storing a candidate high-skew direction."""

    direction: Direction2D
    skew_value: float


def _build_projection_stats(points):
    """Build point list and precomputed statistics.
    
    ORIGINAL VERSION (before min-shift was added):
    - Uses raw points directly, no explicit min-shift preprocessing.
    - The statistics are computed from raw points (mean is computed but
      points themselves are not shifted). This may cause different absolute
      intersection positions, but relative relationships should be preserved.
    """
    pts = [Point2D(float(x), float(y)) for (x, y) in points]
    stats = ProjectionStats2D.from_points(pts)
    return pts, stats


def _build_point_intersection_map(points, intersections_dict):
    """Build a mapping from each point to its intersections, sorted by polar angle.

    This is similar to the `line_intersects` structure in the official implementation.
    For each point, we collect all intersections it participates in, sorted by polar angle.
    """
    point_intersections = {}
    
    for intr, point_set in intersections_dict.items():
        for point in point_set:
            if point not in point_intersections:
                point_intersections[point] = []
            point_intersections[point].append(intr)
    
    # Sort intersections by polar angle for each point
    for point in point_intersections:
        point_intersections[point] = sort_points_by_polar_angle(point_intersections[point])
    
    return point_intersections


def _get_next_median(intersection, candidate_points, prev_median):
    """Find the next median point after crossing an intersection.

    This implements the logic from the official code: when we cross an intersection,
    the median point changes. We find the new median by:
    1. Sorting candidate points (those that generate this intersection) by polar angle
    2. Finding the position of the previous median in this sorted list
    3. Returning the point at the symmetric position

    DIFFERENCE FROM OFFICIAL:
    - Official (`_get_next_median`):
        `sorted(candidate_points, key=lambda x: np.arctan(x[1] / x[0]))`
      This can cause division by zero when x[0] = 0.
    
    - This implementation:
        Uses `polar_angle` which is based on `np.arctan2`, more numerically stable.
      The symmetric index rule is the same: `len(candidates) - index - 1`.

    Args:
        intersection: The intersection point (x, y) in dual space
        candidate_points: Set of points that generate this intersection
        prev_median: The previous median point

    Returns:
        The next median point
    """
    if prev_median not in candidate_points:
        # If previous median is not in candidates, return it unchanged
        # (this shouldn't happen in normal traversal, but handle gracefully)
        return prev_median
    
    # Sort candidate points by polar angle (around the origin)
    # DIFFERENCE: Official uses np.arctan(y/x), we use atan2-based polar_angle
    sorted_candidates = sorted(
        candidate_points,
        key=lambda p: polar_angle(p.as_array()),
    )
    
    # Find the index of the previous median
    try:
        index = sorted_candidates.index(prev_median)
    except ValueError:
        # Fallback: return first candidate
        return sorted_candidates[0]
    
    # Return the point at symmetric position (same rule as official)
    symmetric_index = len(sorted_candidates) - index - 1
    return sorted_candidates[symmetric_index]


def _enumerate_intersections(points):
    """Naively enumerate all dual intersections for a set of 2D points.

    This corresponds conceptually to building the k-level arrangement.
    For simplicity and clarity, we use the quadratic O(n^2) approach
    here. If needed, this can be replaced by a more advanced output-
    sensitive k-level enumeration algorithm, as discussed in the paper.

    DIFFERENCE FROM OFFICIAL:
    - Official (`_get_intersects`): 
      1. Enumerates ALL intersections (no pre-filtering of points)
      2. Sorts by polar angle using `np.arctan(y/x)` (can fail when x=0)
      3. THEN filters: `filter(lambda x: x[1] > 0 and x[0] > 0, ...)`
      This means official code keeps only first-quadrant intersections.
    
    - This implementation (after modification):
      只保留第一象限 (x>0, y>0) 的有限交点，仿照官方实现。
      结合旋转点集和 vector_transfer 来覆盖全方向空间。
    """

    n = len(points)
    intersections = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            p_i = points[i]
            p_j = points[j]
            try:
                x = dual_intersection_2d(p_i, p_j)
                # 只保留第一象限 (x>0, y>0) 的有限交点，仿照官方实现。
                if np.all(np.isfinite(x)) and x[0] > 0 and x[1] > 0:
                    intersections.append((float(x[0]), float(x[1])))
            except np.linalg.LinAlgError:
                # Parallel lines in dual space; ignore.
                continue
    # 只对第一象限的交点按极角排序，匹配官方在 upper halfspace 的遍历策略。
    return sort_points_by_polar_angle(intersections)


def _enumerate_intersections_with_points(points):
    """Enumerate dual intersections and record which point pairs generate each intersection.

    Returns:
        A tuple of:
        - List of intersection points (x, y) sorted by polar angle
        - Dictionary mapping each intersection point to the set of points that generate it
    """
    n = len(points)
    intersections_dict = {}
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            p_i = points[i]
            p_j = points[j]
            try:
                x = dual_intersection_2d(p_i, p_j)
                # 只保留第一象限 (x>0, y>0) 的有限交点，仿照官方实现。
                if np.all(np.isfinite(x)) and x[0] > 0 and x[1] > 0:
                    intr_key = (round(float(x[0]), 8), round(float(x[1]), 8))
                    if intr_key not in intersections_dict:
                        intersections_dict[intr_key] = set()
                    intersections_dict[intr_key].add(p_i)
                    intersections_dict[intr_key].add(p_j)
            except np.linalg.LinAlgError:
                # Parallel lines in dual space; ignore.
                continue
    
    # Sort intersections by polar angle
    intersections_list = sort_points_by_polar_angle(list(intersections_dict.keys()))
    return intersections_list, intersections_dict


def ray_sweeping_2d(
    points,
    top_k=10,
    min_angle_step=np.pi / 90.0,
    use_incremental=False,
    use_randomized=False,
):
    """Basic 2D Ray-sweeping interface with dynamic median tracking (ORIGINAL VERSION).

    This is the ORIGINAL version before min-shift and vector_transfer were added.
    It uses raw points directly without min-shift preprocessing, and does not
    support vector_transfer for rotated point sets.

    This function implements the full theoretical ray-sweeping algorithm:
    - preprocessing and aggregation for constant-time skew updates;
    - enumeration and ordering of dual intersections;
    - dynamic median point tracking as we traverse the k-level arrangement;
    - a sweep over candidate directions while maintaining a max-heap of high-skew directions.

    This implementation tracks median points dynamically as we traverse
    the k-level arrangement, matching the official implementation.

    Args:
        points: Iterable of (x, y) coordinate pairs in the primal space.
        top_k: Number of top high-skew directions to return.
        min_angle_step: Minimum angular step between sampled directions
            (in radians). Smaller values yield more candidates but increase
            computation time.
        use_incremental: If True, use the incremental divide-and-conquer
            enumeration method (`_enumerate_intersections_incremental`)
            instead of the naive O(n^2) method. This can be faster for
            larger point sets due to better cache locality, though the
            worst-case complexity remains O(n^2).
        use_randomized: If True, use the randomized incremental construction
            method (`_enumerate_intersections_randomized_incremental`).
            This has theoretical complexity O(m + n log n) where m = O(n^{4/3}).
            Takes precedence over `use_incremental` if both are True.
            Note: Currently, dynamic median tracking only works with naive enumeration.

    Returns:
        A list of `SkewDirection` objects, sorted by skew value (highest first).
    """

    pts, stats = _build_projection_stats(points)
    if len(pts) < 2:
        return []

    # Initialize median point (x-coordinate median, as in the paper)
    sorted_by_x = sorted(pts, key=lambda p: p.x)
    median_point = sorted_by_x[len(sorted_by_x) // 2]

    # Always use dynamic median tracking (matching official implementation)
    return _ray_sweeping_2d_with_dynamic_median(
        pts, stats, median_point, top_k, min_angle_step
    )


def _ray_sweeping_2d_with_dynamic_median(
    pts,
    stats,
    initial_median,
    top_k,
    min_angle_step,
):
    """Ray-sweeping with dynamic median point tracking (ORIGINAL VERSION).

    This is the ORIGINAL version before vector_transfer was added.
    It directly stores directions without any transformation.

    This implements the k-level arrangement traversal, updating the median point
    at each intersection as we sweep through the arrangement.

    DIFFERENCES FROM OFFICIAL:
    
    1. **Initial Direction**:
       - Official: Starts from `(1 / first_median[0], 0)` (a point on X-axis)
       - This implementation: Starts from the first intersection in the sorted list
       This may cause different traversal orders.
    
    2. **Data Structure**:
       - Official: Uses LinkedList to organize intersections and points
         `line_intersects = {point -> LinkedList[intersection]}`
       - This implementation: Uses dictionaries and lists
         `point_intersections = {point -> List[intersection]}`
       The traversal logic is equivalent but implementation differs.
    
    3. **Termination Condition**:
       - Official: Checks `median_region.end.point[0] == 0` (exact equality)
       - This implementation: Checks `intersection[0] < 1e-10` (floating-point tolerance)
       More robust to numerical errors.
    
    4. **Direction Transformation (ORIGINAL - NO vector_transfer)**:
       - Original version: Directly stores directions without transformation.
       - This means rotated point sets cannot be correctly mapped back to the
         original coordinate system.
    """
    # Enumerate intersections with point relationships
    intersections, intersections_dict = _enumerate_intersections_with_points(pts)
    if not intersections:
        return []

    # Build point-to-intersections mapping
    point_intersections = _build_point_intersection_map(pts, intersections_dict)

    # Initialize: start from the initial median point
    # DIFFERENCE: Official starts from (1/x_median, 0), we start from first intersection
    current_median = initial_median
    current_point = initial_median

    # Get intersections for the starting point
    if current_point not in point_intersections:
        # If starting point has no intersections, return empty results
        return []

    current_intersections = point_intersections[current_point]
    intersection_idx = 0

    heap = []
    last_dir = None
    visited_intersections = set()

    # Traverse the k-level arrangement
    while intersection_idx < len(current_intersections):
        intersection = current_intersections[intersection_idx]

        # Skip if we've already processed this intersection
        if intersection in visited_intersections:
            intersection_idx += 1
            continue

        visited_intersections.add(intersection)

        # Compute direction from intersection
        # ORIGINAL VERSION: No vector_transfer, directly use the direction
        direction = normalize_direction(np.array([intersection[0], intersection[1]], dtype=float))

        # First direction: always evaluate and push into heap
        if last_dir is None:
            skew_val = skew_from_median_point(stats, direction, current_median)
            cand = SkewDirection(direction=direction, skew_value=skew_val)
            heapq.heappush(heap, (-skew_val, cand))
            last_dir = direction
        else:
            # Enforce minimum angular step for subsequent directions
            # DIFFERENCE: Same method as official (arccos of dot product),
            # but official uses L1-normalized vectors, we use L2-normalized.
            # The angle calculation should be equivalent.
            ang = float(
                np.arccos(
                    np.clip(
                        np.dot(direction.as_array(), last_dir.as_array()),
                        -1.0,
                        1.0,
                    )
                )
            )
            if ang >= min_angle_step:
                # Compute skew with current median
                skew_val = skew_from_median_point(stats, direction, current_median)
                cand = SkewDirection(direction=direction, skew_value=skew_val)
                heapq.heappush(heap, (-skew_val, cand))
                last_dir = direction

        # Update median point at this intersection
        if intersection in intersections_dict:
            candidate_points = intersections_dict[intersection]
            new_median = _get_next_median(intersection, candidate_points, current_median)
            
            # If median changed, update and continue traversal from the new point
            if new_median != current_median:
                current_median = new_median
                # Move to the new point's intersection list
                if new_median in point_intersections:
                    # Find the next intersection after the current one
                    new_intersections = point_intersections[new_median]
                    # Find the intersection in the new list (or start from beginning)
                    try:
                        intersection_idx = new_intersections.index(intersection) + 1
                    except ValueError:
                        intersection_idx = 0
                    current_point = new_median
                    current_intersections = new_intersections
                    continue

        intersection_idx += 1

        # Check if we've reached the end (Y-axis)
        # DIFFERENCE: Official checks `median_region.end.point[0] == 0` (exact),
        # we use floating-point tolerance for robustness
        if intersection[0] < 1e-10:  # Close to Y-axis
            break

    results = []
    while heap and len(results) < top_k:
        _, cand = heapq.heappop(heap)
        results.append(cand)

    return results

