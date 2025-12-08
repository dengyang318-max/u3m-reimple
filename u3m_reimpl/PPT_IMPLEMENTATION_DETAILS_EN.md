# PPT Implementation Details (English Version)

## Slide: Reimplementation - Three Core Python Files

### Text Content

**Title** (24pt):
```
Reimplementation - Three Core Python Files
```

**Three Core Files** (18-20pt, with icons):
```
1. geometry.py
   Geometric tools (points, directions, dual transformation)

2. statistics.py
   Statistical tools (skewness calculation)

3. ray_sweeping_2d_original.py
   Core algorithm implementation
```

**Modular Design Philosophy** (16-18pt):
```
• Separation of concerns: geometry, statistics, algorithm logic
• Reusable components: each module can be used independently
• Clear interfaces: dataclasses and well-defined functions
```

### Required Visualizations

1. **Module Dependency Diagram** (Main content, center):
   - Central node: `ray_sweeping_2d_original.py` (red circle with magnifying glass icon)
   - Three connected modules:
     - Top: `geometry.py` (blue circle with clock/pie chart icon)
     - Middle: `statistics.py` (red square with document icon)
     - Bottom: `ray_sweeping_2d_original.py` (blue circle with globe icon)
   - Connect with lines showing dependencies
   - Small grey dots along connection lines

2. **File Structure Tree** (Bottom, simplified):
   ```
   u3m_reimpl/algorithms/
   ├── geometry.py
   ├── statistics.py
   └── ray_sweeping_2d_original.py
   ```

### Reference Documents
- `01_project_overview/README_reimpl.md` Package Structure section
- `u3m_reimpl/algorithms/geometry.py`
- `u3m_reimpl/algorithms/statistics.py`
- `u3m_reimpl/algorithms/ray_sweeping_2d_original.py`

---

## Slide: geometry.py - Geometric Foundations

### Text Content

**Title** (24pt):
```
geometry.py - Geometric Foundations
```

**Core Components** (18-20pt):
```
1. Point2D (dataclass)
   • Represents 2D point (x, y)
   • as_array() converts to NumPy vector

2. Direction2D (dataclass)
   • Represents unit direction vector (dx, dy)
   • Used for projection directions

3. Key Functions:
   • normalize_direction(v): L2 normalization (Euclidean norm)
   • dual_intersection_2d(p1, p2): Compute dual space intersection
   • polar_angle(v): Numerically stable angle calculation (arctan2)
   • sort_points_by_polar_angle(points): Sort by polar angle
```

**Key Differences from Official** (16-18pt, emphasized):
```
• Normalization: L2 (ours) vs L1 (official)
• Polar Angle: arctan2 (ours) vs arctan(y/x) (official)
  → More numerically stable, handles all quadrants
```

### Required Visualizations

1. **Code Structure Diagram** (Left side, 50%):
   - Show class hierarchy: Point2D, Direction2D
   - Function list with icons

2. **Dual Space Transformation Illustration** (Right side, 50%):
   - Upper: Original space (2D points)
   - Lower: Dual space (line intersections)
   - Arrow connecting them, labeled "dual_intersection_2d"

### Reference Documents
- `u3m_reimpl/algorithms/geometry.py` Lines 1-119
- `02_algorithm_comparison/COMPARISON_WITH_OFFICIAL.md` Section 3, 9

---

## Slide: statistics.py - Statistical Tools

### Text Content

**Title** (24pt):
```
statistics.py - Statistical Tools
```

**Core Components** (18-20pt):
```
1. ProjectionStats2D (dataclass)
   • Precomputed statistics for O(1) skew updates
   • Stores: n, mean, sum_vec, xx_sum, yy_sum, xy_sum
   • Methods:
     - projected_mean(direction): Mean of projections
     - projected_std(direction): Std of projections

2. skew_from_median_point(stats, direction, median_point)
   • Computes Pearson-style skewness
   • Formula: |(mean - median)| / std
   • Uses precomputed statistics for efficiency
```

**Key Features** (16-18pt):
```
• Constant-time updates: O(1) per direction
• Algebraic expansion: Uses quadratic form in direction vector
• Numerical stability: Guards against negative variance
```

### Required Visualizations

1. **Statistics Flow Diagram** (Main content):
   - Input: Points → ProjectionStats2D.from_points()
   - Process: Precompute sums (xx_sum, yy_sum, xy_sum)
   - Output: O(1) mean and std calculation for any direction

2. **Skewness Calculation Formula** (Bottom):
   ```
   skew = |(μ_f - median_proj)| / σ_f
   where:
   - μ_f = mean of projections
   - median_proj = median_point^T · direction
   - σ_f = std of projections
   ```

### Reference Documents
- `u3m_reimpl/algorithms/statistics.py` Lines 1-128
- `02_algorithm_comparison/COMPARISON_SKEW_CALCULATION.md`

---

## Slide: ray_sweeping_2d_original.py - Core Algorithm

### Text Content

**Title** (24pt):
```
ray_sweeping_2d_original.py - Core Algorithm
```

**Main Function** (18-20pt):
```
ray_sweeping_2d(points, top_k, min_angle_step, ...)
• Entry point for the algorithm
• Returns: List of SkewDirection objects
```

**Key Internal Functions** (18-20pt):
```
1. _build_projection_stats(points)
   • Converts points to Point2D
   • Builds ProjectionStats2D
   • ORIGINAL: No min-shift preprocessing

2. _enumerate_intersections(points)
   • Naive O(n²) enumeration
   • Filters first quadrant (x>0, y>0)
   • Sorts by polar angle

3. _get_next_median(intersection, candidates, prev_median)
   • Updates median point at each intersection
   • Uses symmetric index rule

4. _ray_sweeping_2d_with_dynamic_median(...)
   • Main ray-sweeping loop
   • Tracks median dynamically
   • Maintains max-heap of top-k directions
```

**Algorithm Flow** (16-18pt):
```
1. Preprocess points → Build statistics
2. Enumerate intersections → Sort by polar angle
3. Initialize median point (x-coordinate median)
4. Sweep intersections → Update median → Calculate skew
5. Maintain top-k heap → Return results
```

### Required Visualizations

1. **Algorithm Flowchart** (Main content, horizontal):
   - 5 steps in rounded rectangles
   - Connect with arrows (→)
   - Icons for each step

2. **Ray-Sweeping Visualization** (Right side, optional):
   - Show ray sweeping through intersections
   - Highlight median point updates

### Reference Documents
- `u3m_reimpl/algorithms/ray_sweeping_2d_original.py` Lines 1-382
- `PROJECT_DOCUMENTATION_EN.md` Section 2.1.1

---

## Slide: Version Evolution - Original → Updated

### Text Content

**Title** (24pt):
```
Version Evolution - Original → Updated
```

**Key Changes in Updated Version** (18-20pt):
```
1. Min-Shift Preprocessing
   • Added _min_shift_points() function
   • Matches official: points[0] -= min, points[1] -= min
   • Applied in _build_projection_stats()

2. Vector Transfer Support
   • Added vector_transfer parameter
   • Maps directions from rotated point sets back to original space
   • Enables full direction space coverage

3. Multiple Enumeration Strategies
   • Naive (O(n²)) - default
   • Incremental Divide-and-Conquer
   • Randomized Incremental (O(m+n log n))
   • Selectable via use_incremental, use_randomized flags
```

**Code Location Changes** 
| Component | Original | Updated |
|-----------|----------|---------|
| Preprocessing | No min-shift | _min_shift_points() |
| Direction Transform | None | vector_transfer parameter |
| Enumeration | Naive only | 3 strategies |
| File | ray_sweeping_2d_original.py | ray_sweeping_2d.py |

### Required Visualizations

1. **Version Comparison Diagram** (Main content):
   - Left: Original version (simpler, no preprocessing)
   - Right: Updated version (with min-shift, vector_transfer)
   - Arrow showing evolution direction
   - Highlight key additions

2. **Enumeration Methods Comparison** (Bottom):
   - Three methods side by side
   - Complexity notation (O(n²), O(m+n log n))

### Reference Documents
- `u3m_reimpl/algorithms/ray_sweeping_2d.py` Lines 27-43, 45-63
- `u3m_reimpl/algorithms/ray_sweeping_2d_original.py` Lines 27-38
- `02_algorithm_comparison/INTERSECTION_ENUMERATION_BENCHMARK.md`

---

## Slide: Version Evolution - Updated → Official Style

### Text Content

**Title** (24pt):
```
Version Evolution - Updated → Official Style
```

**Key Changes in Official Style Version** (18-20pt):
```
1. L1 Normalization (Matching Official)
   • Changed from L2 to L1 normalization
   • normalize_direction: v / sum(v) instead of v / ||v||_2
   • Matches official GeoUtility.normalize_vector

2. arctan(y/x) for Polar Angle (Matching Official)
   • Changed from arctan2 to arctan(y/x)
   • Note: Can cause division by zero (x=0)
   • Matches official GeoUtility.sort_points_by_polar

3. Exact Termination Check
   • Changed from tolerance (x < 1e-10) to exact (x == 0)
   • Matches official termination condition

4. Initial Direction from (1/x_median, 0)
   • Changed from first intersection to (1/x_median, 0)
   • Matches official starting point
```

**Code Location Changes**
| Component | Updated | Official Style |
|-----------|---------|----------------|
| Normalization | L2 (Euclidean) | L1 (sum) |
| Polar Angle | arctan2(y, x) | arctan(y/x) |
| Termination | x < 1e-10 | x == 0 |
| Initial Dir | First intersection | (1/x_median, 0) |
| File | ray_sweeping_2d.py | ray_sweeping_2d_official_style.py |

### Required Visualizations

1. **Comparison Table** (Main content):
   - Three columns: Updated | Official Style | Official
   - Show alignment between Official Style and Official

2. **Normalization Comparison** (Bottom):
   - L2: v / ||v||_2 (geometric)
   - L1: v / sum(v) (official)
   - Visual representation of difference

### Reference Documents
- `u3m_reimpl/algorithms/ray_sweeping_2d_official_style.py`
- `u3m_reimpl/algorithms/geometry.py` Lines 35-57, 82-103
- `02_algorithm_comparison/COMPARISON_WITH_OFFICIAL.md` Sections 2, 3, 9

---

## Slide: Three Versions Summary

### Text Content

**Title** (24pt):
```
Three Versions Summary
```

**Version Comparison Table** (18-20pt, table):
```
| Feature | Original | Updated | Official Style |
|---------|----------|---------|----------------|
| Min-Shift | ❌ No | ✅ Yes | ✅ Yes |
| Vector Transfer | ❌ No | ✅ Yes | ✅ Yes |
| Enumeration | Naive only | 3 methods | Naive only |
| Normalization | L2 | L2 | L1 (official) |
| Polar Angle | arctan2 | arctan2 | arctan(y/x) |
| Termination | Tolerance | Tolerance | Exact (x==0) |
| Initial Dir | First intersection | First intersection | (1/x_median, 0) |
```

**Use Cases** (16-18pt):
```
• Original: Initial reimplementation, learning algorithm
• Updated: Recommended for experiments, supports multiple strategies
• Official Style: For comparison with official results
```

### Required Visualizations

1. **Feature Comparison Matrix** (Main content):
   - Three columns for three versions
   - Use checkmarks (✅) and crosses (❌)
   - Color code: green for features, red for missing

2. **Version Timeline** (Bottom, optional):
   - Show evolution: Original → Updated → Official Style
   - Arrow showing progression

### Reference Documents
- `PROJECT_DOCUMENTATION_EN.md` Section 2.1.2
- `u3m_reimpl/algorithms/ray_sweeping_2d_original.py`
- `u3m_reimpl/algorithms/ray_sweeping_2d.py`
- `u3m_reimpl/algorithms/ray_sweeping_2d_official_style.py`

---

## 📊 Required Code Snippets for Visualization

### geometry.py Key Functions

1. **normalize_direction** (L2 normalization):
   ```python
   def normalize_direction(v):
       norm = np.linalg.norm(v)  # L2 norm
       return Direction2D(dx=float(u[0]), dy=float(u[1]))
   ```

2. **polar_angle** (arctan2-based):
   ```python
   def polar_angle(v):
       ang = float(np.arctan2(v[1], v[0]))  # Numerically stable
       if ang < 0.0:
           ang += 2.0 * np.pi
       return ang
   ```

### statistics.py Key Class

1. **ProjectionStats2D**:
   ```python
   @dataclass
   class ProjectionStats2D:
       n: int
       mean: np.ndarray
       sum_vec: np.ndarray
       xx_sum: float
       yy_sum: float
       xy_sum: float
   ```

### ray_sweeping_2d_original.py Key Function

1. **_build_projection_stats** (Original - no min-shift):
   ```python
   def _build_projection_stats(points):
       pts = [Point2D(float(x), float(y)) for (x, y) in points]
       stats = ProjectionStats2D.from_points(pts)
       return pts, stats
   ```

### ray_sweeping_2d.py Key Changes

1. **_min_shift_points** (New in Updated):
   ```python
   def _min_shift_points(pts: list[Point2D]) -> list[Point2D]:
       min_x = min(p.x for p in pts)
       min_y = min(p.y for p in pts)
       return [Point2D(p.x - min_x, p.y - min_y) for p in pts]
   ```

2. **_build_projection_stats** (Updated - with min-shift):
   ```python
   def _build_projection_stats(points):
       pts = [Point2D(float(x), float(y)) for (x, y) in points]
       pts = _min_shift_points(pts)  # NEW: Min-shift preprocessing
       stats = ProjectionStats2D.from_points(pts)
       return pts, stats
   ```

---

## 🎨 Design Specifications

### Color Scheme

- **Primary Color**: Blue (#2E86AB)
- **Emphasis Color**: Orange (#F18F01)
- **Code Highlight**: Light Gray (#F5F5F5)
- **Version Colors**: 
  - Original: Blue
  - Updated: Green (#06A77D)
  - Official Style: Orange (#F18F01)

### Font Specifications

- **Title**: 24pt, Bold
- **Body Text**: 18-20pt
- **Code**: 14-16pt, Monospace font (Courier New / Consolas)
- **Small Text**: 14-16pt

### Layout Specifications

- **Code Snippets**: Use syntax highlighting if possible
- **Diagrams**: Clear arrows and labels
- **Tables**: Use alternating row colors for readability

---

## ✅ Production Checklist

- [ ] All three Python files introduced clearly
- [ ] Module dependency diagram generated
- [ ] Version comparison table complete
- [ ] Key code differences highlighted
- [ ] Reference documents accessible
- [ ] Code snippets formatted correctly
- [ ] Visualizations match PPT style (handwritten-style fonts for titles)

---

**Last Updated**: 2024-12-05

