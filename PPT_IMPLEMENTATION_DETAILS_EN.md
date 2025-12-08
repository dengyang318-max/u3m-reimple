# PPT Implementation Details (English Version)

## Slide: Reimplementation - Core Files (Configurable)

### Text Content

**Title** (24pt):
```
Reimplementation - Three Core Python Files
```

**Core Files** (18-20pt, with icons):
```
1. geometry.py
   Geometric tools (points, directions, dual/angle helpers)

2. statistics.py
   Projection stats + skewness calculation

3. ray_sweeping_2d_official_style.py
   Parameterized ray-sweeping (original/official behaviors via flags)
   • LinkedList optional, vector_transfer on/off, min-shift on/off, init switch, L1/L2

4. ray_sweeping_2d_official.py
   Official port (LinkedList, L1, atan, no vector_transfer)
```

**Design Philosophy** (16-18pt):
```
• Separation of concerns: geometry, statistics, algorithm logic
• Single configurable implementation: switch behaviors via parameters
• Reuse: geometry/statistics shared by official-style and official port
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
├── ray_sweeping_2d_official_style.py   # configurable
└── ray_sweeping_2d_official.py         # official port
```

### Reference Documents
- `01_project_overview/README_reimpl.md` Package Structure section
- `u3m_reimpl/algorithms/geometry.py`
- `u3m_reimpl/algorithms/statistics.py`
- `u3m_reimpl/algorithms/ray_sweeping_2d_official_style.py`
- `u3m_reimpl/algorithms/ray_sweeping_2d_official.py`

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

**Key Differences vs Official** (16-18pt):
```
• Normalization: L2 in geometry helpers; official uses L1. (Official-style supports L1/L2 via flag.)
• Polar Angle: arctan2 here (stable, all quadrants); official uses atan(y/x).
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

## Slide: ray_sweeping_2d_official_style.py - Configurable Algorithm

### Text Content

**Main Entry** (18-20pt):
```
ray_sweeping_2d_official_style(
    points, top_k, min_angle_step,
    use_linkedlist=False,
    use_first_intersection_init=False,
    use_min_shift=True,
    enable_vector_transfer=True,
    use_l1_norm=True,
)
• Single implementation; parameters emulate original/official behaviors
• Returns: List[SkewDirection]
```

**Key Toggles (18-20pt)**:
```
1) Data structure: LinkedList (official) vs dict+list
2) Init strategy: official (1/x_median,0) vs first-intersection
3) Min-shift: on/off
4) Vector transfer: on/off (mapping rotated dirs back)
5) Normalization: L1 (official) vs L2
```

**Flow** (16-18pt):
```
1. Optional min-shift → build ProjectionStats2D
2. Enumerate intersections (official-style order)
3. Initialize median (x-median) and starting direction (configurable)
4. Sweep intersections, update median, compute skew, maintain top-k
5. Apply vector_transfer (optional) when storing directions
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

## Slide: Parameterization Overview (one codebase, many behaviors)

### Text Content

**Title** (24pt):
```
One Codebase, Parameterized Behaviors
```

**Key Switches (18-20pt)**:
```
• use_linkedlist: LinkedList (official) vs dict+list
• use_first_intersection_init: first-intersection vs (1/x_median,0)
• use_min_shift: on/off
• enable_vector_transfer: on/off (mapping rotated dirs back)
• use_l1_norm: L1 (official) vs L2
```

**Presets (for narration)**:
```
Baseline style: dict+list, min-shift on, vector_transfer on, init=(1/x_median,0), L1
Official port: LinkedList, min-shift on, vector_transfer off, init=(1/x_median,0), L1
Original-like: dict+list, min-shift off, vector_transfer off, first-intersection init, L2
```

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

## Slide: Official Style vs Official Port (what still differs)

### Text Content

**Title** (24pt):
```
Version Evolution - Updated → Official Style
```

**Key Alignment / Differences** (18-20pt):
```
• Normalization: official = L1; official-style supports L1/L2 (flag).
• Polar angle: official = atan(y/x); official-style uses atan (matches), original used arctan2.
• Data structure: official port = LinkedList; official-style can switch (use_linkedlist).
• Vector transfer: official port = off; official-style can toggle.
• Init: official = (1/x_median,0); official-style can also use first-intersection init.
```

**What to show on slide**:
• Small matrix (Normalization, Polar angle, Init, Data structure, Vector transfer)  
  - Official port: L1 / atan(y/x) / (1/x_median,0) / LinkedList / mapping OFF  
  - Official-style (flags): L1 or L2 / atan(y/x) / (1/x_median,0) or first-intersection / LinkedList or dict+list / mapping ON|OFF

**Note on LinkedList** (16-18pt):
```
• ray_sweeping_2d_official.py: LinkedList (canonical port)
• ray_sweeping_2d_official_style.py: LinkedList optional via use_linkedlist flag
```

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
- `u3m_reimpl/algorithms/ray_sweeping_2d_official.py`
- `02_algorithm_comparison/PARAMETERIZED_IMPLEMENTATION_GUIDE.md`

---

## Slide: Important Note - LinkedList Data Structure

### Text Content

**Title** (24pt):
```
Important Note - LinkedList Data Structure
```

**Official Implementation Uses LinkedList** (18-20pt):
```
Official code structure:
• line_intersects = {point -> LinkedList[intersection]}
• Each LinkedList node contains:
  - point: intersection coordinates
  - next: pointer to next intersection on same line
  - neighbours: other LinkedList nodes at same intersection
  - line: original primal point
• Traversal: via next pointers (not list indices)
```

**Official Style (configurable)** (18-20pt):
```
ray_sweeping_2d_official_style.py:
• Default: dict+list; optional LinkedList via use_linkedlist flag
```

**Files with LinkedList** (16-18pt):
```
• ray_sweeping_2d_official.py: LinkedList port
• ray_sweeping_2d_official_style.py: LinkedList optional (flag)
```

**Why This Matters** (16-18pt):
```
• Traversal order → median updates → skew → top-k differences
• Turn on use_linkedlist when matching official traversal is required
```

### Required Visualizations

1. **LinkedList Structure Diagram** (Left side, 50%):
   - Show LinkedList nodes with next pointers
   - Show neighbours relationship
   - Label: point, next, neighbours, line

2. **Comparison Diagram** (Right side, 50%):
   - Left: Official (LinkedList with pointers)
   - Right: Official Style (List with indices)
   - Highlight the difference in traversal method

### Reference Documents
- `Mining_U3Ms-main/Mining_U3Ms-main/utils/linkedlist.py`
- `u3m_reimpl/algorithms/ray_sweeping_2d_official.py` Lines 30-45, 159-174
- `u3m_reimpl/algorithms/ray_sweeping_2d_official_linkedlist.py`
- `02_algorithm_comparison/WHY_RESULTS_DIFFERENT.md` Section 1

---

## Slide: Modes Summary (Preset combinations)

### Text Content

**Title** (24pt):
```
Three Versions Summary
```

**Preset Table** (18-20pt):
```
| Mode                | LinkedList | Init             | Min-Shift | Vector Transfer | Norm | Notes |
|---------------------|-----------|------------------|-----------|-----------------|------|-------|
| Baseline style      | dict+list | (1/x_median,0)   | ON        | ON              | L1   | Default experiments |
| Style + LinkedList  | ON        | (1/x_median,0)   | ON        | ON              | L1   | Match traversal |
| No mapping          | dict+list | (1/x_median,0)   | ON        | OFF             | L1   | Mimic official port mapping |
| No min-shift        | dict+list | (1/x_median,0)   | OFF       | ON              | L1   | Test preprocessing impact |
| L2 variant          | dict+list | (1/x_median,0)   | ON        | ON              | L2   | Match original norm |
| Official port       | ON        | (1/x_median,0)   | ON        | OFF             | L1   | ray_sweeping_2d_official |
```

**Use Cases** (16-18pt):
```
• Reproduce official: Official port, or style with LinkedList + L1 + mapping OFF.
• Isolate differences: toggle vector_transfer, min_shift, init, norm.
• Stability vs exploration: mapping ON for stable Top3; mapping OFF to explore other tails.
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
- `u3m_reimpl/algorithms/ray_sweeping_2d_official_style.py`
- `u3m_reimpl/algorithms/ray_sweeping_2d_official.py`
- `02_algorithm_comparison/PARAMETERIZED_IMPLEMENTATION_GUIDE.md`

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

**Last Updated**: 2025-12-08

