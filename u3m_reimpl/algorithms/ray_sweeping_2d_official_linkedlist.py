from __future__ import annotations

"""
Official-linked-list-style Ray-sweeping implementation.

This file is a closer port of the original `MaxSkewCalculator` in
`Mining_U3Ms-main/utils/ray_sweep.py`, including:

- LinkedList structure for intersections on each line
- MedianRegion with (start, end, median) linked-list endpoints
- Traversal using `neighbours` and `next` pointers
- L1 normalization and atan-based polar angle sorting
- First-quadrant intersection filtering

It exposes a simple wrapper:

    ray_sweeping_2d_official_linkedlist(points, top_k=10, epsilon=pi/10, vector_transfer=None)

where `points` is an iterable of (x, y) pairs in the primal space.
"""

from dataclasses import dataclass
import heapq
import math
from typing import Iterable, List, Tuple, Dict, Set

import numpy as np


# ---------------------------------------------------------------------------
# Helper structures: LinkedList, MedianRegion, SD (same as official code)
# ---------------------------------------------------------------------------


class LinkedList:
    """Linked list of intersection points along a line, matching official code."""

    def __init__(self, point, neighbours, line, next=None):
        self.point = point          # Intersect point (tuple)
        self.next = next            # Next LinkedList node
        self.neighbours = neighbours  # Other LinkedList nodes at same intersection
        self.line = line            # Original primal point (tuple)

    @staticmethod
    def append_to_end(start, node):
        cur = start
        while cur.next is not None:
            cur = cur.next
        cur.next = node

    def __str__(self):
        return f"value: {self.point} --> {self.next is None}"


@dataclass
class MedianRegion:
    """Region between two intersections where the median line is fixed."""

    start: LinkedList
    end: LinkedList
    median: Tuple[float, float]  # The median point in primal space


class SD:
    """Precomputed statistics for fast SD computation (official-style)."""

    def __init__(self, points, mean):
        # In the original code, `points` is a numpy array; here we accept
        # any iterable and convert to an array for numeric operations.
        pts = np.array(points, dtype=float)
        self.points = pts
        self.n = pts.shape[0]
        self.mean = np.array(mean, dtype=float)
        # Sum over all points (vector)
        self.sum = pts.sum(axis=0)
        x = pts[:, 0]
        y = pts[:, 1]
        self.x_2_sum = float(np.sum(x ** 2))
        self.y_2_sum = float(np.sum(y ** 2))
        self.xy_sum = float(np.sum(x * y))

    def get_sd(self, f):
        mean_f = np.dot(self.mean, f)
        sd2 = (
            self.n * mean_f**2
            - 2 * mean_f * np.dot(self.sum, f)
            + f[0] ** 2 * self.x_2_sum
            + f[1] ** 2 * self.y_2_sum
            + 2 * f[0] * f[1] * self.xy_sum
        )
        return math.sqrt(sd2 / self.n)


class GeoUtility:
    """Geometry utilities, matching the official dual-space helpers."""

    @staticmethod
    def get_intersect_in_dual(point_a, point_b):
        # point_a, point_b: (x, y) in primal; solve for dual intersection
        value = np.linalg.solve(np.array([point_a, point_b], dtype=float), np.ones(2))
        return np.array([round(value[0], 5), round(value[1], 5)])

    @staticmethod
    def sort_points_by_polar(points: Dict[Tuple[float, float], Set[Tuple[float, float]]]):
        keys = points.keys()
        return sorted(keys, key=lambda x: math.atan(x[1] / x[0]))

    @staticmethod
    def normalize_vector(vector: Tuple[float, float]):
        # L1 normalization as in official code
        v = np.array(vector, dtype=float)
        s = float(v.sum())
        if abs(s) < 1e-12:
            return v
        return v / s


# ---------------------------------------------------------------------------
# MaxSkewCalculator with LinkedList-based traversal (official-style)
# ---------------------------------------------------------------------------


class MaxSkewCalculatorLinked:
    """LinkedList-based MaxSkewCalculator, closer to the official implementation."""

    def __init__(self, points: Iterable[Tuple[float, float]], vector_transfer, epsilon):
        # Convert to numpy array and apply min-shift (both dimensions)
        arr = np.array(list(points), dtype=float)
        if arr.shape[0] == 0:
            raise ValueError("At least one point is required")

        # Min-shift to start at 0 (matching official pre-processing)
        arr[:, 0] = arr[:, 0] - arr[:, 0].min()
        arr[:, 1] = arr[:, 1] - arr[:, 1].min()

        # Store as list of tuples for dictionary keys
        self.points = [tuple(arr[i, :]) for i in range(arr.shape[0])]
        self.points_array = arr

        # Precompute statistics for skew
        mean = np.mean(arr, axis=0)
        self.sd = SD(self.points, mean)

        self.intersects: Dict[Tuple[float, float], Set[Tuple[float, float]]] = {}
        self.intersect_keys: List[Tuple[float, float]] = []
        self.line_intersects: Dict[Tuple[float, float], LinkedList] = {}

        self.heap: List[Tuple[float, Tuple[float, float]]] = []
        self.vector_transfer = vector_transfer
        self.epsilon = float(epsilon)

    def get_angel(self, vec1, vec2):
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        return float(np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0)))

    def _get_first_median_on_x(self):
        x_sorted = sorted(self.points, key=lambda x: x[0])
        return tuple(x_sorted[len(self.points) // 2])

    def _calc_skew(self, f, median, verbose=False):
        m_point = np.array(median, dtype=float)
        mean_f = np.dot(self.sd.mean, f)

        if verbose:
            print(np.median(np.dot(self.points_array, f)), np.dot(m_point, f))
        skew = abs((mean_f - np.dot(m_point, f)) / self.sd.get_sd(f))
        return float(skew)

    def _get_next_median(self, intersection, candidate_points, prev_median):
        # Sort candidate points by atan(y/x), as in official code
        candidate_points = sorted(
            candidate_points, key=lambda x: math.atan(x[1] / x[0])
        )
        index = candidate_points.index(prev_median)
        return candidate_points[len(candidate_points) - index - 1]

    def _get_intersects(self):
        self.intersects = {}
        n = len(self.points)
        for i in range(n - 1):
            point_a = self.points[i]
            for j in range(i + 1, n):
                point_b = self.points[j]
                try:
                    intr = tuple(GeoUtility.get_intersect_in_dual(point_a, point_b))
                    if intr in self.intersects:
                        self.intersects[intr].add(point_a)
                        self.intersects[intr].add(point_b)
                    else:
                        self.intersects[intr] = {point_a, point_b}
                except Exception:
                    continue
        self.intersect_keys = GeoUtility.sort_points_by_polar(self.intersects)
        # Keep intersection in the upper halfspace / first quadrant
        self.intersect_keys = list(
            filter(lambda x: x[1] > 0 and x[0] > 0, self.intersect_keys)
        )

    def _get_line_intersects(self):
        self.line_intersects = {}

        for key in self.intersect_keys:
            intersect = key
            occurs = self.intersects[intersect]

            links = [LinkedList(intersect, [], point, None) for point in occurs]
            for link in links:
                link.neighbours = links

            for i, point in enumerate(occurs):
                if point in self.line_intersects:
                    l = self.line_intersects[point]
                    LinkedList.append_to_end(l, links[i])
                else:
                    self.line_intersects[point] = links[i]

    def preprocess(self, verbose=False):
        self._get_intersects()
        self._get_line_intersects()
        if verbose:
            print(f"Number of intersects: {len(self.intersects)}")

    def train(self, verbose=False):
        first_median = self._get_first_median_on_x()
        last_vec = None

        # Initial median region
        median_region = MedianRegion(
            LinkedList((1.0 / first_median[0], 0.0), [], first_median, None),
            self.line_intersects[first_median],
            first_median,
        )
        finish = False

        while not finish:
            if verbose:
                print(median_region)

            skew_vector_start = GeoUtility.normalize_vector(median_region.start.point)

            if (
                last_vec is not None
                and self.get_angel(skew_vector_start, last_vec) > self.epsilon
            ):
                # Compute skew and push to heap
                skew_val = self._calc_skew(skew_vector_start, median_region.median, verbose)
                direction_stored = self.vector_transfer(tuple(skew_vector_start))
                heapq.heappush(self.heap, (-skew_val, direction_stored))
                last_vec = skew_vector_start

            if last_vec is None:
                last_vec = skew_vector_start

            # Termination: reached Y-axis
            if median_region.end.point[0] == 0:
                if verbose:
                    print("Reached Y axis, finish.")
                break

            # Next median region
            try:
                current_points = self.intersects[median_region.end.point]
            except Exception:
                if verbose:
                    print("Didn't find end of median region, quit.")
                break

            line_b = self._get_next_median(
                median_region.end.point, list(current_points), median_region.median
            )
            next_neighbour_list = list(
                filter(lambda n: n.line == line_b, median_region.end.neighbours)
            )
            if not next_neighbour_list:
                if verbose:
                    print("Didn't find next neighbour, quit.")
                break
            next_neighbour = next_neighbour_list[0]

            if verbose:
                print(
                    f"nextneighbour: {next_neighbour.point}, next: {next_neighbour.next}"
                )

            if next_neighbour.next is None:
                new_end = LinkedList((0.0, 1.0 / line_b[1]), [], line_b, None)
            else:
                new_end = next_neighbour.next  # change of line

            median_region = MedianRegion(
                median_region.end, new_end, line_b
            )  # median changes to line_b!


# ---------------------------------------------------------------------------
# Public wrapper: convert heap results to SkewDirection objects
# ---------------------------------------------------------------------------


@dataclass
class SkewDirection:
    direction: np.ndarray  # 2D unit vector
    skew_value: float


def ray_sweeping_2d_official_linkedlist(
    points: Iterable[Tuple[float, float]],
    top_k: int = 10,
    epsilon: float = math.pi / 10.0,
    vector_transfer=None,
) -> List[SkewDirection]:
    """
    Official-linked-list-style Ray-sweeping wrapper.

    Args:
        points: Iterable of (x, y) coordinate pairs in primal space.
        top_k: Number of top directions to return.
        epsilon: Angular threshold (in radians) between successive directions.
        vector_transfer: Optional function to transform direction vectors when
            storing them (e.g. for rotated point sets). If None, identity is used.

    Returns:
        List of SkewDirection (direction as unit vector, skew_value) sorted by skew.
    """
    if vector_transfer is None:
        vector_transfer = lambda x: (x[0], x[1])

    calc = MaxSkewCalculatorLinked(points, vector_transfer=vector_transfer, epsilon=epsilon)
    calc.preprocess(verbose=False)
    calc.train(verbose=False)

    # Extract top-k from heap; heap stores (-skew, direction_tuple)
    heap_copy = list(calc.heap)
    heapq.heapify(heap_copy)

    results: List[SkewDirection] = []
    while heap_copy and len(results) < top_k:
        neg_skew, dir_tuple = heapq.heappop(heap_copy)
        skew_val = -float(neg_skew)
        v = np.array(dir_tuple, dtype=float)
        # Normalize to unit vector for external use (L2 norm)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        results.append(SkewDirection(direction=v, skew_value=skew_val))

    # Sort by skew descending (heap pops smallest first)
    results.sort(key=lambda d: d.skew_value, reverse=True)
    return results


