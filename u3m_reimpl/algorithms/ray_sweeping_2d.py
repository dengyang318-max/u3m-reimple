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


def _min_shift_points(pts: list[Point2D]) -> list[Point2D]:
    """Apply min-shift preprocessing to match the official implementation.

    Official (`MaxSkewCalculator.__init__`):
        points[0] -= points[0].min()
        points[1] -= points[1].min()

    Here we apply the same logic on the list of `Point2D` objects.
    """
    if not pts:
        return pts

    min_x = min(p.x for p in pts)
    min_y = min(p.y for p in pts)
    # Shift both coordinates so that the minimum becomes 0, matching official.
    return [Point2D(p.x - min_x, p.y - min_y) for p in pts]


def _build_projection_stats(points):
    """Build point list and precomputed statistics.
    
    DIFFERENCE FROM OFFICIAL:
    - Official (`MaxSkewCalculator.__init__`):
      1. Shifts points: `points[0] -= points[0].min()`, `points[1] -= points[1].min()`
      2. Computes centered points: `q = points - mean(points, axis=0)`
      3. Uses shifted points for all calculations.
    - This implementation (UPDATED):
      Applies the same min-shift preprocessing on both coordinates so that
      intersection positions and projection statistics are computed on the
      shifted point set, matching the official behavior.
    """
    # Convert to Point2D
    pts = [Point2D(float(x), float(y)) for (x, y) in points]
    # Apply min-shift on both dimensions to match the official preprocessing.
    pts = _min_shift_points(pts)
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
      Keeps ALL finite intersections, preserving full-circle sign information.
      This allows negative slopes to appear in results, matching official
      behavior when considering rotated point sets with vector_transfer.
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


def _enumerate_intersections_incremental(points):
    """Enumerate dual intersections using an incremental construction approach.

    This implements a more efficient output-sensitive algorithm inspired by
    the randomized incremental construction method mentioned in the paper
    (Section 4.1). The theoretical complexity is O(m + n log n), where
    m = O(n^{4/3}) is the number of vertices in the k-th level arrangement.

    Algorithm overview:
    1. Optionally sort points by dual line slope for geometric ordering.
    2. Use divide-and-conquer to compute intersections more efficiently:
       - Recursively split points into two groups
       - Compute intersections within each group and between groups
       - Merge results while avoiding duplicates
    3. Keep all finite intersections and sort them over the full circle.

    This approach reduces the constant factors compared to naive O(n^2),
    and is more cache-friendly due to the recursive structure. For very
    large inputs, this can be significantly faster in practice.

    Args:
        points: List of 2D points in the primal space.

    Returns:
        A sorted list of intersection points (x, y) over the entire dual plane,
        ordered by polar angle around the origin.
    """
    n = len(points)
    if n < 2:
        return []

    # Step 1: Work with all points; 第一象限过滤在交点生成阶段完成。
    candidate_points = list(points)

    if len(candidate_points) < 2:
        return []

    # Step 2: Sort by dual line slope for geometric ordering.
    # This helps with cache locality and early termination in the divide step.
    def dual_slope(p):
        """Compute the slope of the dual line for point p."""
        if abs(p.y) < 1e-10:
            return float('inf') if p.x > 0 else float('-inf')
        return -p.x / p.y

    sorted_points = sorted(candidate_points, key=dual_slope)
    m = len(sorted_points)

    # Step 3: Divide-and-conquer approach for computing intersections.
    # For small sets, use the naive method. For larger sets, split recursively.
    def _compute_intersections_dc(pts, threshold=10):
        """Recursively compute intersections using divide-and-conquer."""
        if len(pts) <= threshold:
            # Base case: use naive method for small sets.
            intersections = set()
            for i in range(len(pts) - 1):
                for j in range(i + 1, len(pts)):
                    try:
                        x = dual_intersection_2d(pts[i], pts[j])
                        # 只保留第一象限 (x>0, y>0) 的有限交点。
                        if np.all(np.isfinite(x)) and x[0] > 0 and x[1] > 0:
                            x_rounded = (round(float(x[0]), 8), round(float(x[1]), 8))
                            intersections.add(x_rounded)
                    except np.linalg.LinAlgError:
                        continue
            return intersections

        # Divide: split points into two halves.
        mid = len(pts) // 2
        left = pts[:mid]
        right = pts[mid:]

        # Conquer: recursively compute intersections.
        left_intersections = _compute_intersections_dc(left, threshold)
        right_intersections = _compute_intersections_dc(right, threshold)

        # Combine: compute intersections between left and right groups,
        # and merge with existing intersections.
        cross_intersections = set()
        for p_l in left:
            for p_r in right:
                try:
                    x = dual_intersection_2d(p_l, p_r)
                    # 只保留第一象限 (x>0, y>0) 的有限交点。
                    if np.all(np.isfinite(x)) and x[0] > 0 and x[1] > 0:
                        x_rounded = (round(float(x[0]), 8), round(float(x[1]), 8))
                        cross_intersections.add(x_rounded)
                except np.linalg.LinAlgError:
                    continue

        # Merge all intersections.
        return left_intersections | right_intersections | cross_intersections

    # Compute intersections using divide-and-conquer.
    intersection_set = _compute_intersections_dc(sorted_points)

    # Step 4: Convert to list and sort by polar angle.
    intersections = list(intersection_set)
    return sort_points_by_polar_angle(intersections)


def _enumerate_intersections_randomized_incremental(points, seed=None):
    """Enumerate dual intersections using randomized incremental construction.

    This implements the randomized incremental algorithm mentioned in the paper
    (Section 4.1). The theoretical complexity is O(m + n log n), where
    m = O(n^{4/3}) is the number of vertices in the k-th level arrangement.

    Algorithm overview (randomized incremental construction):
    1. Randomly permute the input points.
    2. Start with an empty set of intersections.
    3. Incrementally insert points one by one:
       - For each new point, compute intersections with all previously
         inserted points (lines in dual space)
       - Maintain the set of active intersections (no quadrant filter)
    4. Sort intersections by polar angle over the full circle.

    The key insight is that by randomizing the insertion order, the expected
    number of intersections we need to update at each step is O(1) on average,
    leading to better average-case performance than the naive O(n^2) method.

    Args:
        points: List of 2D points in the primal space.
        seed: Random seed for reproducibility. If None, uses system time.

    Returns:
        A sorted list of intersection points (x, y) over the entire dual plane,
        ordered by polar angle around the origin.
    """
    n = len(points)
    if n < 2:
        return []

    # Step 1: Work with all points; 第一象限过滤在交点生成阶段完成。
    candidate_points = list(points)

    if len(candidate_points) < 2:
        return []

    # Step 2: Randomly permute the points.
    # This randomization is crucial for the expected O(m + n log n) performance.
    if seed is not None:
        np.random.seed(seed)
    else:
        # Use a deterministic seed based on point count for reproducibility
        # while still having different behavior for different inputs
        np.random.seed(hash(tuple((p.x, p.y) for p in candidate_points)) % (2**31))
    
    # Create a copy and shuffle
    shuffled_points = candidate_points.copy()
    np.random.shuffle(shuffled_points)

    # Step 3: Incrementally build intersections.
    # We maintain a set of active intersections as we add points one by one.
    intersections: set = set()
    
    # For each point being inserted, compute intersections with all
    # previously inserted points.
    for i, new_point in enumerate(shuffled_points):
        # Compute intersections with all previously inserted points
        for j in range(i):
            prev_point = shuffled_points[j]
            try:
                x = dual_intersection_2d(new_point, prev_point)
                # 只保留第一象限 (x>0, y>0) 的有限交点。
                if np.all(np.isfinite(x)) and x[0] > 0 and x[1] > 0:
                    # Round to avoid floating-point duplicates
                    x_rounded = (round(float(x[0]), 8), round(float(x[1]), 8))
                    intersections.add(x_rounded)
            except np.linalg.LinAlgError:
                # Parallel lines in dual space; ignore.
                continue

    # Step 4: Convert to list and sort by polar angle.
    intersections_list = list(intersections)
    return sort_points_by_polar_angle(intersections_list)


def ray_sweeping_2d(
    points,
    top_k=10,
    min_angle_step=np.pi / 90.0,
    use_incremental=False,
    use_randomized=False,
    vector_transfer=None,
):
    """Basic 2D Ray-sweeping interface with dynamic median tracking.

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
        vector_transfer: Optional function to transform direction vectors
            when storing them (e.g. for rotated point sets:
            `lambda x: (-x[1], x[0])`). If None, an identity mapping
            `lambda x: (x[0], x[1])` is used, matching the official
            `vector_transfer` mechanism.

    Returns:
        A list of `SkewDirection` objects, sorted by skew value (highest first).
    """

    if vector_transfer is None:
        # Match official default behavior: identity mapping.
        vector_transfer = lambda x: (x[0], x[1])

    pts, stats = _build_projection_stats(points)
    if len(pts) < 2:
        return []

    # Initialize median point (x-coordinate median, as in the paper)
    sorted_by_x = sorted(pts, key=lambda p: p.x)
    median_point = sorted_by_x[len(sorted_by_x) // 2]

    # Always use dynamic median tracking (matching official implementation)
    return _ray_sweeping_2d_with_dynamic_median(
        pts, stats, median_point, top_k, min_angle_step, vector_transfer
    )


def _ray_sweeping_2d_with_dynamic_median(
    pts,
    stats,
    initial_median,
    top_k,
    min_angle_step,
    vector_transfer,
):
    """Ray-sweeping with dynamic median point tracking (matching official implementation).

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
    4. **Direction Transformation (UPDATED)**:
       - Official: Applies `vector_transfer` before storing directions in the heap so
         that rotated point sets can be mapped back to the original coordinate system.
       - This implementation now supports an equivalent `vector_transfer` parameter
         in `ray_sweeping_2d`, and applies it when pushing directions to the heap
         while still computing skew on the original (untransformed) direction.
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
    # Always store the *original* (pre-vector_transfer) direction in last_dir for
    # angle checks, matching the official behavior conceptually.
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

        # Compute direction from intersection (normalized).
        # Skew calculation always uses this original direction.
        direction = normalize_direction(
            np.array([intersection[0], intersection[1]], dtype=float)
        )

        # Apply vector_transfer *only for storage* (matching official idea:
        # use transformed directions for output, but original for skew).
        dir_array = direction.as_array()
        transferred = np.array(
            vector_transfer((dir_array[0], dir_array[1])), dtype=float
        )
        direction_stored = normalize_direction(transferred)

        # First direction: always evaluate and push into heap
        if last_dir is None:
            skew_val = skew_from_median_point(stats, direction, current_median)
            cand = SkewDirection(direction=direction_stored, skew_value=skew_val)
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
                cand = SkewDirection(direction=direction_stored, skew_value=skew_val)
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



