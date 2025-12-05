from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .geometry import Direction2D, Point2D


@dataclass
class ProjectionStats2D:
    """Precomputed statistics for constant-time skew updates in 2D.

    This class implements the aggregation strategy described in the
    paper: we precompute a small number of global quantities over the
    dataset so that, for any direction f, we can obtain the mean and
    standard deviation of the projections t^T f in O(1) time (with
    respect to n, the number of points).
    """

    n: int
    mean: np.ndarray
    sum_vec: np.ndarray
    xx_sum: float
    yy_sum: float
    xy_sum: float

    @classmethod
    def from_points(cls, points):
        arr = np.array([[p.x, p.y] for p in points], dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("Expected array of shape (n, 2)")
        n = arr.shape[0]
        if n == 0:
            raise ValueError("At least one point is required")
        mean = arr.mean(axis=0)
        sum_vec = arr.sum(axis=0)
        x = arr[:, 0]
        y = arr[:, 1]
        xx_sum = float(np.dot(x, x))
        yy_sum = float(np.dot(y, y))
        xy_sum = float(np.dot(x, y))
        return cls(
            n=n,
            mean=mean,
            sum_vec=sum_vec,
            xx_sum=xx_sum,
            yy_sum=yy_sum,
            xy_sum=xy_sum,
        )

    def projected_mean(self, direction):
        """Mean of t^T f over all points, in O(1) time."""

        f = direction.as_array()
        return float(self.mean @ f)

    def projected_std(self, direction):
        """Standard deviation of t^T f over all points, in O(1) time.

        We use the algebraic expansion from the paper. Let u_j = t_j^T f
        and μ_f be the mean projection. Then

            Σ_j (u_j - μ_f)^2
                = Σ_j (t_j^T f)^2
                  - 2 μ_f Σ_j t_j^T f
                  + n μ_f^2

        and Σ_j (t_j^T f)^2 can be written as a quadratic form in f
        using precomputed xx_sum, yy_sum, xy_sum.
        """

        f = direction.as_array()
        fx, fy = float(f[0]), float(f[1])

        mu_f = float(self.mean @ f)
        sum_proj = float(self.sum_vec @ f)

        # Σ_j (t_j^T f)^2 = f_x^2 Σ x_j^2 + f_y^2 Σ y_j^2 + 2 f_x f_y Σ x_j y_j
        sum_sq = (
            fx * fx * self.xx_sum
            + fy * fy * self.yy_sum
            + 2.0 * fx * fy * self.xy_sum
        )

        numerator = sum_sq - 2.0 * mu_f * sum_proj + self.n * mu_f * mu_f
        if numerator < 0.0:
            # Numerical guard: small negatives due to rounding.
            numerator = 0.0
        return float(np.sqrt(numerator / self.n))


def skew_from_median_point(stats, direction, median_point):
    """Compute (absolute) Pearson-style skew given a median point.

    The paper uses Pearson's median skewness

        skew(V) = 3 (μ(V) - ν(V)) / σ(V)

    Here V is the set of 1D projections onto a direction f. Instead of
    recomputing the median for every direction, the algorithm tracks
    a median point t_m in the primal space. Given such a point, the
    projected median is just t_m^T f.

    We return the absolute skew value; the caller can attach the sign
    if needed.

    DIFFERENCE FROM OFFICIAL:
    - Official (`_calc_skew`): Uses precomputed SD class with formula:
        `sd2 = n * mean_f^2 - 2 * mean_f * dot(sum, f) + f[0]^2 * x_2_sum + ...`
        `skew = abs((mean_f - dot(m_point, f)) / sd.get_sd(f))`
    - This implementation: Uses ProjectionStats2D with equivalent formula.
      The math is the same, but implementation structure differs.
      Both omit the factor 3 (doesn't affect ranking).
    """

    mu_f = stats.projected_mean(direction)
    sigma_f = stats.projected_std(direction)
    if sigma_f == 0.0:
        return 0.0
    v = median_point.as_array()
    median_proj = float(v @ direction.as_array())
    # We omit the factor 3 because it does not affect ranking by skew.
    return abs((mu_f - median_proj) / sigma_f)



