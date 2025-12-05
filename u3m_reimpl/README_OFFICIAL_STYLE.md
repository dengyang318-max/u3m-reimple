# Official-Style Ray Sweeping Implementation

## 文件说明

`ray_sweeping_2d_official_style.py` 是一个与官方代码实现方法完全统一的版本。

## 主要统一点

### 1. **归一化方法：L1 归一化**
- 官方：`normalize_vector(vector) = vector / sum(vector)`
- 本文件：`normalize_direction_l1(v) = v / sum(v)`

### 2. **极角排序：使用 atan(y/x)**
- 官方：`sorted(keys, key=lambda x: np.arctan(x[1] / x[0]))`
- 本文件：`sort_points_by_polar_atan` 使用 `np.arctan(y/x)`
- 注意：处理了除零情况

### 3. **数据预处理：min-shift**
- 官方：`points[0] -= points[0].min()`, `points[1] -= points[1].min()`
- 本文件：`_build_projection_stats_official_style` 实现相同的预处理

### 4. **交点过滤：先排序后过滤第一象限**
- 官方：
  1. 枚举所有交点
  2. 使用 `atan` 排序
  3. 过滤：`filter(lambda x: x[1] > 0 and x[0] > 0, ...)`
- 本文件：`_enumerate_intersections_with_points_official_style` 实现相同的三步过程

### 5. **初始方向：从 (1/x_median, 0) 开始**
- 官方：`LinkedList((1 / first_median[0], 0), [], first_median, None)`
- 本文件：从 `(1.0 / initial_median.x, 0.0)` 开始计算初始方向

### 6. **中位数更新：使用 atan(y/x) 排序**
- 官方：`sorted(candidate_points, key=lambda x: np.arctan(x[1] / x[0]))`
- 本文件：`_get_next_median_official_style` 使用 `polar_angle_atan`

### 7. **终止条件：精确检查 x == 0**
- 官方：`if median_region.end.point[0] == 0:`
- 本文件：`if intersection[0] == 0.0:`

### 8. **方向转换：支持 vector_transfer**
- 官方：在 push 到 heap 前应用 `vector_transfer`
- 本文件：支持 `vector_transfer` 参数，默认使用 identity

## 使用方法

```python
from ray_sweeping_2d_official_style import ray_sweeping_2d_official_style

# Primary point set
results_primary = ray_sweeping_2d_official_style(
    points=primary_points,
    top_k=10,
    min_angle_step=np.pi / 10.0,
    vector_transfer=lambda x: tuple([x[0], x[1]])  # identity for primary
)

# Rotated point set
results_rotated = ray_sweeping_2d_official_style(
    points=rotated_points,
    top_k=10,
    min_angle_step=np.pi / 10.0,
    vector_transfer=lambda x: tuple([-x[1], x[0]])  # rotation for rotated set
)
```

## 与原始复现代码的对比

| 特性 | 原始复现 (`ray_sweeping_2d.py`) | 官方风格 (`ray_sweeping_2d_official_style.py`) |
|------|--------------------------------|-----------------------------------------------|
| 归一化 | L2 (Euclidean) | L1 (sum) |
| 极角排序 | `atan2(y, x)` | `atan(y/x)` |
| 数据预处理 | 无 min-shift | 有 min-shift |
| 交点过滤 | 保留全圆 | 只保留第一象限 |
| 初始方向 | 第一个交点 | `(1/x_median, 0)` |
| 终止条件 | `x < 1e-10` | `x == 0.0` |
| vector_transfer | 不支持 | 支持 |

## 注意事项

1. **除零处理**：`atan(y/x)` 在 `x=0` 时会除零，代码中已添加处理
2. **浮点精度**：官方使用 `round(value, 5)`，本文件也使用 5 位小数
3. **初始方向**：如果 `x_median` 太小，会回退到第一个交点
4. **角度检查**：第一个方向（从 `(1/x_median, 0)`）不会立即 push 到 heap，需要等待后续方向满足角度条件

