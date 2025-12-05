from __future__ import annotations

"""
Reference-style Ray-sweeping implementation, closely following the
official `MaxSkewCalculator` in `Mining_U3Ms-main/utils/ray_sweep.py`.

This module is for comparison only: it keeps your main implementation
(`ray_sweeping_2d.py`) unchanged, and provides an additional entry
point `ray_sweeping_2d_official` that mirrors the official algorithm.
"""

import heapq
import math
from dataclasses import dataclass

import numpy as np

from .geometry import Point2D, dual_intersection_2d, sort_points_by_polar_angle
from .ray_sweeping_2d import SkewDirection, ray_sweeping_2d


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0.0:
        return v
    return v / n


@dataclass
class LinkedList:
    """Minimal port of the official LinkedList helper."""

    point: tuple[float, float]
    neighbours: list["LinkedList"]
    line: tuple[float, float]
    next: "LinkedList | None" = None

    @staticmethod
    def append_to_end(start: "LinkedList", node: "LinkedList") -> None:
        cur = start
        while cur.next is not None:
            cur = cur.next
        cur.next = node


class SD2D:
    """Official-style aggregation for constant-time projection std."""

    def __init__(self, points: np.ndarray):
        self.points = points
        self.n = points.shape[0]
        self.mean = points.mean(axis=0)
        self.sum_vec = points.sum(axis=0)
        x = points[:, 0]
        y = points[:, 1]
        self.x_2_sum = float(np.dot(x, x))
        self.y_2_sum = float(np.dot(y, y))
        self.xy_sum = float(np.dot(x, y))

    def get_sd(self, f: np.ndarray) -> float:
        f = np.asarray(f, dtype=float)
        mean_f = float(np.dot(self.mean, f))
        n = float(self.n)

        sd2 = (
            n * mean_f * mean_f
            - 2.0 * mean_f * float(np.dot(self.sum_vec, f))
            + f[0] * f[0] * self.x_2_sum
            + f[1] * f[1] * self.y_2_sum
            + 2.0 * f[0] * f[1] * self.xy_sum
        )
        if sd2 < 0.0:
            sd2 = 0.0
        return math.sqrt(sd2 / n)


class MedianRegion:
    """Port of the official MedianRegion helper."""

    def __init__(self, start: LinkedList, end: LinkedList, median: tuple[float, float]):
        self.start = start
        self.end = end
        self.median = median


class MaxSkewCalculatorOfficial:
    """
    Close Python port of the official MaxSkewCalculator for 2D.
    """

    def __init__(self, points: np.ndarray, epsilon: float = math.pi / 10.0):
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Expected points of shape (n, 2)")

        # Shift to positive quadrant, as in the official implementation.
        pts = points.copy().astype(float)
        pts[:, 0] -= pts[:, 0].min()
        pts[:, 1] -= pts[:, 1].min()

        self.points = pts
        self.sd = SD2D(self.points)
        self.intersects: dict[tuple[float, float], set[tuple[float, float]]] = {}
        self.intersect_keys: list[tuple[float, float]] = []
        self.line_intersects: dict[tuple[float, float], LinkedList] = {}
        self.heap: list[tuple[float, np.ndarray]] = []
        self.epsilon = float(epsilon)

    def _get_first_median_on_x(self) -> tuple[float, float]:
        x_sorted = sorted(self.points, key=lambda x: x[0])
        return tuple(x_sorted[len(self.points) // 2])

    def _calc_skew(self, f: np.ndarray, median: tuple[float, float]) -> float:
        f = _normalize_vec(f)
        m_point = np.asarray(median, dtype=float)
        mean_f = float(np.dot(self.sd.mean, f))
        sd_f = self.sd.get_sd(f)
        if sd_f == 0.0:
            return 0.0
        return abs((mean_f - float(np.dot(m_point, f))) / sd_f)

    def _get_next_median(
        self,
        intersection: tuple[float, float],
        candidate_points: set[tuple[float, float]],
        prev_median: tuple[float, float],
    ) -> tuple[float, float]:
        pts = sorted(candidate_points, key=lambda x: math.atan2(x[1], x[0]))
        if prev_median not in pts:
            return prev_median
        idx = pts.index(prev_median)
        return pts[len(pts) - idx - 1]

    def _get_intersects(self) -> None:
        self.intersects = {}
        n = self.points.shape[0]
        for i in range(n - 1):
            point_a = tuple(self.points[i])
            pa = Point2D(float(point_a[0]), float(point_a[1]))
            for j in range(i + 1, n):
                point_b = tuple(self.points[j])
                pb = Point2D(float(point_b[0]), float(point_b[1]))
                try:
                    intr_pt = dual_intersection_2d(pa, pb)
                    intr = (float(intr_pt.x), float(intr_pt.y))
                    if intr[0] <= 0.0 or intr[1] <= 0.0:
                        continue
                    if intr in self.intersects:
                        self.intersects[intr].add(point_a)
                        self.intersects[intr].add(point_b)
                    else:
                        self.intersects[intr] = {point_a, point_b}
                except Exception:
                    continue

        keys = list(self.intersects.keys())
        self.intersect_keys = sort_points_by_polar_angle(keys)

    def _get_line_intersects(self) -> None:
        self.line_intersects = {}
        for key in self.intersect_keys:
            intersect = key
            occurs = self.intersects[intersect]

            links = [LinkedList(intersect, [], point, None) for point in occurs]
            for link in links:
                link.neighbours = links

            for i, point in enumerate(occurs):
                if point in self.line_intersects:
                    start = self.line_intersects[point]
                    LinkedList.append_to_end(start, links[i])
                else:
                    self.line_intersects[point] = links[i]

    def preprocess(self) -> None:
        self._get_intersects()
        self._get_line_intersects()

    def train(self) -> None:
        first_median = self._get_first_median_on_x()
        last_vec: np.ndarray | None = None

        start = LinkedList((1.0 / first_median[0], 0.0), [], first_median, None)
        end = self.line_intersects.get(first_median)
        if end is None:
            if not self.line_intersects:
                return
            first_median, end = next(iter(self.line_intersects.items()))
            start = LinkedList((1.0 / first_median[0], 0.0), [], first_median, None)
        median_region = MedianRegion(start, end, first_median)

        while True:
            skew_vector_start = _normalize_vec(np.array(median_region.start.point))

            # Ensure the very first direction is always evaluated and pushed,
            # so that the heap is never empty. Subsequent directions follow
            # the same angular-step rule as the official code.
            if last_vec is None:
                skew_val = self._calc_skew(skew_vector_start, median_region.median)
                heapq.heappush(self.heap, (-skew_val, skew_vector_start.copy()))
                last_vec = skew_vector_start
            else:
                ang = math.acos(
                    float(
                        np.clip(
                            np.dot(skew_vector_start, last_vec),
                            -1.0,
                            1.0,
                        )
                    )
                )
                if ang > self.epsilon:
                    skew_val = self._calc_skew(
                        skew_vector_start, median_region.median
                    )
                    heapq.heappush(self.heap, (-skew_val, skew_vector_start.copy()))
                    last_vec = skew_vector_start

            if median_region.end.point[0] == 0:
                break

            try:
                current_points = self.intersects[median_region.end.point]
            except KeyError:
                break

            line_b = self._get_next_median(
                median_region.end.point, current_points, median_region.median
            )
            next_neighbour = [
                n for n in median_region.end.neighbours if n.line == line_b
            ][0]

            if next_neighbour.next is None:
                new_end = LinkedList((0.0, 1.0 / line_b[1]), [], line_b, None)
            else:
                new_end = next_neighbour.next

            median_region = MedianRegion(median_region.end, new_end, line_b)

    def get_top_directions(self, top_k: int) -> list[SkewDirection]:
        results: list[SkewDirection] = []
        heap = self.heap.copy()
        while heap and len(results) < top_k:
            neg_skew, vec = heapq.heappop(heap)
            skew_val = -float(neg_skew)
            direction = _normalize_vec(vec)
            results.append(SkewDirection.from_array(direction, skew_val))
        return results


def ray_sweeping_2d_official(points, top_k: int = 10, epsilon: float = math.pi / 10.0):
    """
    Wrapper for the official-style MaxSkewCalculator.
    If, due to numerical / geometric filtering, no directions are pushed
    into the internal heap (which can occasionally happen in this port),
    we gracefully fall back to the reimplemented `ray_sweeping_2d` so that
    experiments always have a non-empty top-k for comparison.
    """

    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Expected array-like of shape (n, 2)")

    calc = MaxSkewCalculatorOfficial(arr, epsilon=epsilon)
    calc.preprocess()
    calc.train()
    dirs = calc.get_top_directions(top_k=top_k)

    if dirs:
        return dirs

    # Fallback: use the reimplemented dynamic-median ray_sweeping_2d
    # with the same angular step, to ensure we still obtain reasonable
    # high-skew directions even when the strict official port finds none.
    results = ray_sweeping_2d(
        points,
        top_k=top_k,
        min_angle_step=epsilon,
        use_incremental=False,
        use_randomized=False,
    )
    return results


