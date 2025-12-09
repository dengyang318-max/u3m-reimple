# Official Implementation Differences and Parameter Control Guide

This document provides a detailed summary of the differences between `ray_sweeping_2d_original.py` and `ray_sweeping_2d_official_style.py` compared to the official implementation, and explains how each parameter in `ray_sweeping_2d_official_style.py` controls these differences.

---

## 1. Core Differences Overview Table

| Difference Category | Official Implementation | ray_sweeping_2d_original.py | ray_sweeping_2d_official_style.py | Parameter Control |
|---------------------|------------------------|----------------------------|----------------------------------|-------------------|
| **Normalization Method** | L1 normalization | L2 normalization  | Configurable L1/L2 | `use_l1_norm` |
| **Data Preprocessing** | min-shift (`points -= min`) | No min-shift | Configurable min-shift | `use_min_shift` |
| **Data Structure** | LinkedList | dict + list | Configurable LinkedList/dict+list | `use_linkedlist` |
| **Initialization Method** | `(1/x_median, 0)` | First intersection | Configurable | `use_first_intersection_init` |
| **vector_transfer** | Supported (for rotated point sets) | Not supported | Configurable | `enable_vector_transfer` |

---

## 2. Detailed Difference Explanations

### 2.1 Normalization Method Differences

#### Official Implementation
```python
def normalize_vector(vector: tuple):
    return np.array(vector) / sum(vector)  # L1 normalization
```

#### ray_sweeping_2d_original.py
```python
def normalize_direction(v):
    norm = np.linalg.norm(v)  # L2 norm
    u = v / norm
    return Direction2D(dx=float(u[0]), dy=float(u[1]))
```

#### ray_sweeping_2d_official_style.py
```python
def normalize_direction_l1(v):
    s = float(np.sum(v))
    u = v / s
    return Direction2D(dx=float(u[0]), dy=float(u[1]))

def normalize_direction_l2(v):
    n = float(np.linalg.norm(v))
    u = v / n
    return Direction2D(dx=float(u[0]), dy=float(u[1]))

# Parameter control
normalize_fn = normalize_direction_l1 if use_l1_norm else normalize_direction_l2
```

**Impact**:
- L1 normalization: `||f||_1 = f[0] + f[1] = 1`, preserves direction ratio
- L2 normalization: `||f||_2 = sqrt(f[0]^2 + f[1]^2) = 1`, standard unit vector
- Different normalization methods affect direction vector representation, but relative skew value ordering should be consistent

**Parameter Control**:
- `use_l1_norm=True`: Use L1 normalization (matches official)
- `use_l1_norm=False`: Use L2 normalization (original version style)

---

### 2.2 Data Preprocessing Differences

#### Official Implementation
```python
points[0] = points[0] - points[0].min()
points[1] = points[1] - points[1].min()
```

#### ray_sweeping_2d_original.py
```python
def _build_projection_stats(points):
    pts = [Point2D(float(x), float(y)) for (x, y) in points]
    stats = ProjectionStats2D.from_points(pts)
    return pts, stats
# No min-shift preprocessing
```

#### ray_sweeping_2d_official_style.py
```python
def _build_projection_stats_official_style(points, use_min_shift: bool = True):
    arr = np.array([[float(x), float(y)] for (x, y) in points], dtype=float)
    if use_min_shift:
        arr[:, 0] = arr[:, 0] - arr[:, 0].min()
        arr[:, 1] = arr[:, 1] - arr[:, 1].min()
    pts = [Point2D(float(arr[i, 0]), float(arr[i, 1])) for i in range(arr.shape[0])]
    stats = ProjectionStats2D.from_points(pts)
    return pts, stats
```

**Impact**:
- min-shift makes coordinates start from 0, may change absolute positions of intersections
- But relative relationships should remain consistent
- For some datasets, min-shift may affect discovered high-skew directions

**Parameter Control**:
- `use_min_shift=True`: Apply min-shift preprocessing (matches official)
- `use_min_shift=False`: No min-shift (original version style)

---

### 2.3 Data Structure Differences

#### Official Implementation
```python
line_intersects = {point -> LinkedList[intersection]}
# Uses LinkedList to organize intersections and points
# Traverses via next pointers and neighbours
```

#### ray_sweeping_2d_original.py
```python
point_intersections = {point -> List[intersection]}
# Uses dictionary + list
# Traverses via list indices
```

#### ray_sweeping_2d_official_style.py
```python
# Supports two data structures
if use_linkedlist:
    # Use LinkedList (matches official)
    line_intersects = _build_line_intersects_linkedlist(...)
else:
    # Use dict + list
    point_intersections = _build_point_intersection_map_official_style(...)
```

**Impact**:
- Traversal order may differ between LinkedList and dict+list
- Different traversal orders may lead to different median update sequences
- May ultimately find different top-k directions

**Parameter Control**:
- `use_linkedlist=True`: Use LinkedList data structure (matches official)
- `use_linkedlist=False`: Use dict + list (original version style)

---

### 2.4 Initialization Method Differences

#### Official Implementation
```python
start = LinkedList((1.0 / first_median[0], 0.0), [], first_median, None)
# Starts from (1/x_median, 0)
```

#### ray_sweeping_2d_original.py
```python
# Starts from first intersection
first_intersection = current_intersections[0]
direction = normalize_direction(np.array([first_intersection[0], first_intersection[1]]))
```

#### ray_sweeping_2d_official_style.py
```python
if use_first_intersection_init:
    # Start from first intersection (ray_sweeping_2d_original style)
    first_intersection = intersect_keys[0]
    initial_dir = normalize_fn(np.array([first_intersection[0], first_intersection[1]]))
else:
    # Start from (1/x_median, 0) (official style)
    start_dir_vec = np.array([1.0 / initial_median.x, 0.0], dtype=float)
    start_dir_raw = normalize_fn(start_dir_vec)
```

**Impact**:
- Different starting directions may lead to different traversal paths
- May affect the set of discovered high-skew directions

**Parameter Control**:
- `use_first_intersection_init=False`: Start from `(1/x_median, 0)` (matches official)
- `use_first_intersection_init=True`: Start from first intersection (original version style)

---

### 2.5 vector_transfer Support Differences

#### Official Implementation
```python
# Supports vector_transfer for mapping directions from rotated point sets back to original coordinate system
dir_transferred = vector_transfer(direction)
```

#### ray_sweeping_2d_original.py
```python
# Does not support vector_transfer
# Directly stores directions, cannot handle rotated point sets
```

#### ray_sweeping_2d_official_style.py
```python
if not enable_vector_transfer:
    vector_transfer = lambda x: (x[0], x[1])  # Identity mapping
elif vector_transfer is None:
    vector_transfer = lambda x: tuple([x[0], x[1]])

# Apply vector_transfer
dir_transferred_tuple = vector_transfer(tuple(direction.as_array()))
dir_transferred_arr = np.array(dir_transferred_tuple, dtype=float)
direction = normalize_fn(dir_transferred_arr)
```

**Impact**:
- vector_transfer is used to handle rotated point sets, mapping directions back to original coordinate system
- Disabling vector_transfer may cause directions from rotated point sets to be incorrectly mapped

**Parameter Control**:
- `enable_vector_transfer=True`: Apply vector_transfer mapping (matches official)
- `enable_vector_transfer=False`: Do not use vector_transfer (identity mapping)


---

## 3. Parameter Control Mapping Table

### 3.1 ray_sweeping_2d_official_style Parameter Overview

| Parameter | Type | Default Value | Controls Difference | Official Match Value |
|-----------|------|---------------|---------------------|---------------------|
| `use_l1_norm` | bool | `True` | Normalization method (L1 vs L2) | `True` |
| `use_min_shift` | bool | `True` | Data preprocessing (min-shift) | `True` |
| `use_linkedlist` | bool | `False` | Data structure (LinkedList vs dict+list) | `True` |
| `use_first_intersection_init` | bool | `False` | Initialization method | `False` |
| `enable_vector_transfer` | bool | `True` | vector_transfer support | `True` |

### 3.2 Parameter Combinations and Difference Mapping

| Parameter Combination | Normalization | Preprocessing | Data Structure | Initialization | vector_transfer | Description |
|----------------------|---------------|---------------|-----------------|----------------|-----------------|-------------|
| **Full Official Match** | L1 | min-shift | LinkedList | (1/x_median, 0) | Enabled | Closest to official implementation |
| `use_l1_norm=True`<br>`use_min_shift=True`<br>`use_linkedlist=True`<br>`use_first_intersection_init=False`<br>`enable_vector_transfer=True` | ✅ | ✅ | ✅ | ✅ | ✅ | |
| **Official Style (dict+list)** | L1 | min-shift | dict+list | (1/x_median, 0) | Enabled | Matches official except data structure |
| `use_l1_norm=True`<br>`use_min_shift=True`<br>`use_linkedlist=False`<br>`use_first_intersection_init=False`<br>`enable_vector_transfer=True` | ✅ | ✅ | ❌ | ✅ | ✅ | |
| **Original Version Style** | L2 | No min-shift | dict+list | First intersection | Disabled | Similar to ray_sweeping_2d_original |
| `use_l1_norm=False`<br>`use_min_shift=False`<br>`use_linkedlist=False`<br>`use_first_intersection_init=True`<br>`enable_vector_transfer=False` | ❌ | ❌ | ❌ | ❌ | ❌ | |

---

## 4. Fixed Differences (No Parameter Control)

The following differences in `ray_sweeping_2d_official_style.py` are **fixed to match official** and cannot be controlled by parameters:

1. **Polar Angle Calculation**: Fixed to use `atan(y/x)` (matches official)
2. **Termination Condition**: Fixed to use exact check `x == 0` (matches official)
3. **Intersection Enumeration Strategy**: Fixed to use "enumerate all → sort → filter first quadrant" (matches official)
4. **Median Update Sorting**: Fixed to use `atan(y/x)` (matches official)

These fixed differences ensure that `ray_sweeping_2d_official_style.py` maintains consistency with the official implementation in core algorithm logic.

---

## 5. Usage Recommendations

### 5.1 Full Official Match
```python
results = ray_sweeping_2d_official_style(
    points,
    use_l1_norm=True,              # L1 normalization
    use_min_shift=True,             # min-shift preprocessing
    use_linkedlist=True,            # LinkedList data structure
    use_first_intersection_init=False,  # Official initialization method
    enable_vector_transfer=True,    # Enable vector_transfer
)
```

### 5.2 Original Version Style (For Comparison)
```python
results = ray_sweeping_2d_official_style(
    points,
    use_l1_norm=False,             # L2 normalization
    use_min_shift=False,            # No min-shift
    use_linkedlist=False,           # dict+list data structure
    use_first_intersection_init=True,  # First intersection initialization
    enable_vector_transfer=False,   # Disable vector_transfer
)
```

### 5.3 Parameter Ablation Experiments
You can test the impact of each difference by disabling parameters one by one:
- Disable `use_min_shift`: Test impact of min-shift
- Disable `enable_vector_transfer`: Test impact of vector_transfer
- Toggle `use_l1_norm`: Test impact of normalization method
- Toggle `use_linkedlist`: Test impact of data structure

---

## 6. Summary

`ray_sweeping_2d_official_style.py` controls differences with the official implementation through 5 main parameters:

1. **`use_l1_norm`**: Controls normalization method (L1 vs L2)
2. **`use_min_shift`**: Controls data preprocessing (min-shift)
3. **`use_linkedlist`**: Controls data structure (LinkedList vs dict+list)
4. **`use_first_intersection_init`**: Controls initialization method
5. **`enable_vector_transfer`**: Controls vector_transfer support

Other differences (polar angle calculation, termination condition, intersection enumeration strategy, median update sorting) are **fixed to match official**, ensuring consistency in core algorithm logic.

By properly configuring these parameters, you can flexibly test the impact of different implementation details on results while maintaining algorithm correctness.

---

**Document Generated**: 2025-12-08

