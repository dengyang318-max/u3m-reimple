## Overview

This document explains the design and purpose of the new `u3m_reimpl` package,
which provides an independent reimplementation of the 2D Minoria
Ray-sweeping algorithm based on the paper:

- *Mining the Minoria: Unknown, Under-represented, and Under-performing Minority Groups*.

The goal is:

- to faithfully follow the algorithmic ideas in the paper,
- to **only** take high-level inspiration from the official code in
  `Mining_U3Ms-main`, and
- to provide a cleaner, better-structured implementation for study and
  experimentation.

This reimplementation is intentionally placed in a separate package and uses
different abstractions and code structure compared to the original
`utils` modules.

---

## Package Structure

The new code lives under:

- `u3m_reimpl/`
  - `__init__.py`
  - `geometry.py`
  - `statistics.py`
  - `ray_sweeping_2d.py`

### `u3m_reimpl/__init__.py`

This file exposes the main public types and utilities:

- Geometry:
  - `Point2D`
  - `Direction2D`
  - `dual_intersection_2d`
  - `normalize_direction`
  - `sort_points_by_polar_angle`

- Statistics:
  - `ProjectionStats2D`
  - `skew_from_median_point`

You can import them via:

```python
from u3m_reimpl import (
    Point2D,
    Direction2D,
    dual_intersection_2d,
    normalize_direction,
    sort_points_by_polar_angle,
    ProjectionStats2D,
    skew_from_median_point,
)
```

---

## Geometric and Dual-Space Utilities (`geometry.py`)

This module implements the geometric foundations that correspond to the
paper’s Section 3 (“Geometric Interpretation”).

### `Point2D`

- A small dataclass representing a 2D point `(x, y)`.
- Provides `as_array()` to convert to a NumPy vector.
- Used to represent primal points `t` in the data space.

### `Direction2D`

- A dataclass for a 2D **unit** direction vector `(dx, dy)`.
- `as_array()` returns a NumPy length-2 array.
- Represents projection directions `f` on the unit circle, as in the paper.

### `normalize_direction(v: np.ndarray) -> Direction2D`

- Takes a raw 2D vector and returns a unit-length `Direction2D`.
- Uses the **Euclidean norm** for normalization:
  - This matches the theoretical model of unit vectors in the paper.
  - It deliberately differs from the reference implementation, which
    normalizes by the coordinate sum.

### `dual_intersection_2d(p1: Point2D, p2: Point2D) -> np.ndarray`

- Implements the dual transformation used in the paper:

  - A primal point `t = (t_1, t_2)` is mapped to a dual line:

    \[
      d(t): t_1 x_1 + t_2 x_2 = 1.
    \]

  - The intersection of `d(p1)` and `d(p2)` is obtained by solving a
    2x2 linear system:

    \[
      \begin{bmatrix}
        p1.x & p1.y \\
        p2.x & p2.y
      \end{bmatrix}
      \begin{bmatrix}
        x_1 \\ x_2
      \end{bmatrix}
      =
      \begin{bmatrix}
        1 \\ 1
      \end{bmatrix}.
    \]

- Returns the intersection point in the dual space as a NumPy array.

### `sort_points_by_polar_angle(points) -> list`

- Sorts 2D points around the origin by polar angle.
- Uses `atan2(y, x)` instead of `y/x`:
  - numerically more stable,
  - handles all quadrants correctly.
- Conceptually corresponds to traversing the vertices of the k-th level
  of the arrangement in angular order, as required by the ray-sweeping view.

---

## Statistical Aggregation for Constant-Time Skew (`statistics.py`)

This module implements the aggregation strategy described in the paper
that allows **constant-time** (with respect to `n`) updates of the skew
for each direction.

### `ProjectionStats2D`

`ProjectionStats2D` encapsulates precomputed global statistics over the data:

- `n`: number of points.
- `mean`: the mean point \(\mu(D)\) in \(\mathbb{R}^2\).
- `sum_vec`: the sum of all points \(\sum_j t_j\).
- `xx_sum`: \(\sum_j x_j^2\).
- `yy_sum`: \(\sum_j y_j^2\).
- `xy_sum`: \(\sum_j x_j y_j\).

These are exactly the kinds of aggregates used in the paper to derive a
constant-time formula for:

- the mean of the projections \(t_j^\top f\),
- the standard deviation of those projections.

#### `ProjectionStats2D.from_points(points)`

- Accepts an iterable of `Point2D`.
- Builds all required aggregates in a single pass.
- This is the O(n) preprocessing step discussed in the paper.

#### `projected_mean(direction: Direction2D) -> float`

- Computes \(\mu_f = \mu(D)^\top f\) in O(1) time.
- Corresponds directly to the paper’s formula for the mean of the
  projected values.

#### `projected_std(direction: Direction2D) -> float`

- Computes the standard deviation \(\sigma_f\) in O(1) time using the
  algebraic expansion from the paper:

  \[
    \sum_j (t_j^\top f - \mu_f)^2
    =
    \sum_j (t_j^\top f)^2
    - 2 \mu_f \sum_j t_j^\top f
    + n \mu_f^2.
  \]

- The term \(\sum_j (t_j^\top f)^2\) is written as a quadratic form in `f`
  using the precomputed `xx_sum`, `yy_sum`, and `xy_sum`.
- This matches the constant-time update mechanism described in the paper.

### `skew_from_median_point(stats, direction, median_point) -> float`

- Given:
  - global projection statistics `stats` (`ProjectionStats2D`),
  - a direction `direction` (`Direction2D`),
  - a **median point** `median_point` (`Point2D`),

  computes the absolute skew:

  \[
    \text{skew}(f) = \left|\mu_f - t_m^\top f\right| / \sigma_f,
  \]

  where \(t_m\) is the median point and \(t_m^\top f\) is the projected
  median.

- This follows the paper’s idea that within a median region the median
  point is fixed, and you can use its projection instead of recomputing
  the median from scratch for each direction.
- The factor 3 of Pearson’s median skewness is omitted intentionally,
  because it does not affect the ranking of directions by skew.

---

## Simplified 2D Ray-sweeping Interface (`ray_sweeping_2d.py`)

This module provides a **practical, simplified** Ray-sweeping interface
that uses the geometric and statistical foundations above. It is
structured to be close to the algorithmic description in the paper but
is implemented independently from the official `utils/ray_sweep.py`.

### Data Types

- `MedianSegment`:
  - Represents a contiguous range of directions that share the same
    median point (a light-weight analogue of the “median region” in the
    paper).

- `SkewDirection`:
  - A small record storing a candidate high-skew direction and its skew
    value.

### Internal Helpers

#### `_build_projection_stats(points)`

- Converts raw `(x, y)` tuples into `Point2D` objects.
- Builds a `ProjectionStats2D` instance for constant-time skew updates.

#### `_enumerate_intersections(points)`

- Naively (O(n²)) enumerates all pairs of primal points, computes their
  dual-line intersection using `dual_intersection_2d`, and keeps those
  in the first quadrant.
- Then sorts all intersection points by polar angle using
  `sort_points_by_polar_angle`.
- Conceptually corresponds to building and traversing the k-level
  arrangement’s skeleton in angular order.

> Note: The paper discusses more advanced, output-sensitive algorithms
> for k-level enumeration. Here we deliberately start with a simpler,
> easier-to-understand O(n²) version which can be refined later.

### Main Function: `ray_sweeping_2d`

```python
def ray_sweeping_2d(
    points: Iterable[Tuple[float, float]],
    top_k: int = 10,
    min_angle_step: float = np.pi / 90.0,
) -> List[SkewDirection]:
    ...
```

This function currently implements the **infrastructure** of the 2D
Ray-sweeping algorithm:

1. **Preprocessing**
   - Build `Point2D` list and `ProjectionStats2D` (for constant-time
     skew updates).
   - Enumerate and angle-sort dual intersections.
   - Approximate the initial median point as the median by x-coordinate
     (same idea as in the reference code, but implemented differently).

2. **Sweeping**
   - For each intersection:
     - Convert its coordinates to a unit direction via
       `normalize_direction`.
     - Enforce a minimal angular step (`min_angle_step`) from the last
       sampled direction to avoid oversampling near-identical
       directions.
     - Use `skew_from_median_point(stats, direction, median_point)` to
       compute the skew in O(1) time.
     - Push `SkewDirection(direction, skew_value)` into a heap, using
       negative skew values to emulate a max-heap.

3. **Extraction**
   - Pop up to `top_k` directions with highest skew values from the
     heap and return them as a list of `SkewDirection` objects.

This gives you a working algorithm that:

- follows the same **high-level design** as the paper (dual
  intersections, angular sweep, constant-time skew),
- is clearly separated from the original implementation, and
- is easier to extend towards the full median-region based Ray-sweeping
  described in the paper.

---

## Example Usage

Here is a minimal example showing how to call the simplified
`ray_sweeping_2d` function:

```python
from u3m_reimpl.ray_sweeping_2d import ray_sweeping_2d

# A small set of 2D points (e.g., the toy example from the paper)
sample_points = [
    (0.2, 0.8),
    (0.8, 0.2),
    (0.6, 1.2),
    (1.0, 0.6),
    (1.4, 0.4),
]

top_dirs = ray_sweeping_2d(sample_points, top_k=3)

for i, cand in enumerate(top_dirs, 1):
    d = cand.direction.as_array()
    print(f\"#{i}: dir={d}, skew={cand.skew_value:.4f}\")
```

This is a good way to quickly verify that:

- the geometric utilities are working,
- the statistical aggregation correctly computes mean/std and skew, and
- the heap-based top-k extraction behaves as expected.

---

## How This Connects to the Paper (and Next Steps)

The reimplementation in `u3m_reimpl` is designed as a **clean basis**
for a full reproduction of the algorithm in the paper:

- **Geometry & Dual Space** (`geometry.py`)
  - Directly matches the dual transform and angular traversal used to
    define median regions and k-levels.

- **Constant-Time Skew** (`statistics.py`)
  - Implements the algebraic expansions used in the paper to reduce
    skew computation for each vertex to O(1) time (with respect to n).

- **Ray-sweeping Skeleton** (`ray_sweeping_2d.py`)
  - Provides:
    - intersection enumeration,
    - angular ordering,
    - heap-based selection of high-skew directions,
    using the independent abstractions above.

From here, you can gradually refine this implementation to match the
full Ray-sweeping algorithm in the paper, by:

- making the notion of **median regions** explicit (with a proper graph
  over the k-level skeleton),
- updating the median point and aggregates only when crossing region
  boundaries, and
- integrating the model loss evaluation on tails (Problem 1 in the
  paper) to filter candidate Minoria directions.

This README focuses on the **preparation work** and **basic methods**
that are required before those more advanced steps.


