# 复现代码与官方代码的详细对比

本文档详细对比了复现代码（`ray_sweeping_2d.py`）与官方代码（`Mining_U3Ms-main/utils/ray_sweep.py`）在算法实现上的关键差异。

## 1. 数据预处理差异

### 官方代码（`MaxSkewCalculator.__init__`）:
```python
points[0] = points[0] - points[0].min()
points[1] = points[1] - points[1].min()
self.points = np.array(points)
self.q = self.points - np.mean(self.points, axis=0)  # 中心化
```

### 复现代码:
- **差异**: 复现代码没有显式地进行 `min` 归一化，直接使用原始点坐标
- **位置**: `_build_projection_stats` 函数中，直接使用原始点构建统计量
- **影响**: 可能导致交点的绝对位置不同，但相对关系应该一致

## 2. 交点枚举和过滤策略

### 官方代码（`_get_intersects`）:
```python
# 1. 枚举所有交点（不预过滤点）
for i in range(len(self.points) - 1):
    for point_b in self.points[i + 1:]:
        intr = tuple(GeoUtility.get_intersect_in_dual(point_a, point_b))
        # 存储所有交点

# 2. 按极角排序
self.intersect_keys = GeoUtility.sort_points_by_polar(self.intersects)

# 3. 最后过滤：只保留第一象限
self.intersect_keys = list(
    filter(lambda x: x[1] > 0 and x[0] > 0, self.intersect_keys)
)
```

### 复现代码（`_enumerate_intersections_with_points`）:
```python
# 枚举所有交点，不进行象限过滤（已修改）
for i in range(n - 1):
    for j in range(i + 1, n):
        x = dual_intersection_2d(p_i, p_j)
        if np.all(np.isfinite(x)):  # 只检查有限性
            intersections_dict[intr_key] = set([p_i, p_j])
```

- **差异**: 
  - 官方：先枚举所有交点，排序后再过滤第一象限
  - 复现：枚举所有有限交点，不进行象限过滤（保留全圆符号信息）
- **影响**: 复现代码现在可以产生负斜率的方向，与官方行为更接近

## 3. 归一化方法差异

### 官方代码（`GeoUtility.normalize_vector`）:
```python
def normalize_vector(vector: tuple):
    return np.array(vector) / sum(vector)  # L1 归一化
```

### 复现代码（`normalize_direction`）:
```python
def normalize_direction(v):
    norm = np.linalg.norm(v)  # L2 范数
    u = v / norm
    return Direction2D(dx=float(u[0]), dy=float(u[1]))
```

- **差异**: 
  - 官方：使用 **L1 归一化**（除以坐标和）
  - 复现：使用 **L2 归一化**（除以欧几里得范数）
- **影响**: 归一化后的方向向量长度不同，但方向相同。这会影响 skew 计算的数值，但排序应该一致

## 4. 极角排序方法差异

### 官方代码（`GeoUtility.sort_points_by_polar`）:
```python
def sort_points_by_polar(points):
    keys = points.keys()
    return sorted(keys, key=lambda x: np.arctan(x[1] / x[0]))
```

### 复现代码（`sort_points_by_polar_angle`）:
```python
def sort_points_by_polar_angle(points):
    return sorted(pts, key=lambda p: polar_angle(np.array(p, dtype=float)))

def polar_angle(v):
    ang = float(np.arctan2(v[1], v[0]))  # 使用 atan2
    if ang < 0.0:
        ang += 2.0 * np.pi
    return ang
```

- **差异**: 
  - 官方：使用 `np.arctan(y/x)`，当 `x=0` 时会除零错误
  - 复现：使用 `np.arctan2(y, x)`，数值更稳定，能正确处理所有象限
- **影响**: 复现代码更稳健，但排序结果在大部分情况下应该一致

## 5. 标准差计算差异

### 官方代码（`SD.get_sd`）:
```python
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
```

### 复现代码（`ProjectionStats2D.projected_std`）:
```python
def projected_std(self, direction):
    mu_f = float(self.mean @ f)
    sum_proj = float(self.sum_vec @ f)
    sum_sq = (
        fx * fx * self.xx_sum
        + fy * fy * self.yy_sum
        + 2.0 * fx * fy * self.xy_sum
    )
    numerator = sum_sq - 2.0 * mu_f * sum_proj + self.n * mu_f * mu_f
    return float(np.sqrt(numerator / self.n))
```

- **差异**: 数学公式等价，但实现细节略有不同
- **影响**: 应该产生相同的结果

## 6. 中位数更新方法差异

### 官方代码（`_get_next_median`）:
```python
def _get_next_median(self, intersection, candidate_points, prev_median):
    candidate_points = sorted(
        candidate_points, key=lambda x: np.arctan(x[1] / x[0])  # 可能除零
    )
    index = candidate_points.index(prev_median)
    return candidate_points[len(candidate_points) - index - 1]
```

### 复现代码（`_get_next_median`）:
```python
def _get_next_median(intersection, candidate_points, prev_median):
    sorted_candidates = sorted(
        candidate_points,
        key=lambda p: polar_angle(p.as_array()),  # 使用 atan2
    )
    index = sorted_candidates.index(prev_median)
    symmetric_index = len(sorted_candidates) - index - 1
    return sorted_candidates[symmetric_index]
```

- **差异**: 
  - 官方：使用 `np.arctan(y/x)` 排序
  - 复现：使用 `polar_angle`（基于 `atan2`）排序
- **影响**: 复现代码更稳健，但逻辑相同（对称索引规则）

## 7. 遍历结构差异

### 官方代码:
- 使用 **LinkedList** 结构组织交点和点的关系
- `line_intersects`: `{point -> LinkedList[intersection]}`
- 通过 `neighbours` 和 `next` 指针遍历

### 复现代码:
- 使用 **字典和列表**结构
- `point_intersections`: `{point -> List[intersection]}`
- 通过索引遍历列表

- **差异**: 数据结构不同，但遍历逻辑等价
- **影响**: 性能可能略有不同，但算法逻辑应该一致

## 8. 初始方向设置差异

### 官方代码（`train` 方法）:
```python
median_region = MedianRegion(
    LinkedList((1 / first_median[0], 0), [], first_median, None),  # 从 (1/x_median, 0) 开始
    self.line_intersects[first_median],
    first_median,
)
```

### 复现代码（`_ray_sweeping_2d_with_dynamic_median`）:
```python
# 从第一个交点开始
current_intersections = point_intersections[current_point]
intersection_idx = 0
```

- **差异**: 
  - 官方：从 `(1/x_median, 0)` 这个特殊点开始（X轴上的点）
  - 复现：从第一个交点开始
- **影响**: 起始方向不同，可能导致遍历顺序略有差异

## 9. 角度计算差异

### 官方代码（`get_angel`）:
```python
def get_angel(self, vec1, vec2):
    vec1 = vec1 / np.linalg.norm(vec1)  # L2 归一化
    vec2 = vec2 / np.linalg.norm(vec2)
    return np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))
```

### 复现代码（`_ray_sweeping_2d_with_dynamic_median`）:
```python
ang = float(
    np.arccos(
        np.clip(
            np.dot(direction.as_array(), last_dir.as_array()),
            -1.0, 1.0,
        )
    )
)
```

- **差异**: 基本相同，都使用 `arccos(dot product)`
- **影响**: 应该产生相同的结果

## 10. 终止条件差异

### 官方代码:
```python
if median_region.end.point[0] == 0:  # 精确等于 0
    print("Reached Y axis, finish.")
    break
```

### 复现代码:
```python
if intersection[0] < 1e-10:  # 接近 0（浮点容差）
    break
```

- **差异**: 
  - 官方：精确检查 `x == 0`
  - 复现：使用浮点容差 `x < 1e-10`
- **影响**: 复现代码更稳健，避免浮点误差问题

## 11. 方向向量转换（vector_transfer）

### 官方代码:
```python
# 在创建 MaxSkewCalculator 时传入
max_skew_1 = MaxSkewCalculator(points, skew_heap, 
    lambda x: tuple([x[0], x[1]]), math.pi / 10)  # primary set
max_skew_2 = MaxSkewCalculator(points_prime, skew_heap,
    lambda x: tuple([-x[1], x[0]]), math.pi / 10)  # rotated set

# 在 train 中应用
heapq.heappush(self.heap, (
    -self._calc_skew(...),
    self.vector_transfer(tuple(skew_vector_start)),  # 转换方向
))
```

### 复现代码:
- **差异**: 复现代码没有 `vector_transfer` 机制
- **影响**: 对于 rotated point set，官方会进行坐标转换，复现直接使用原始方向

## 12. 点集旋转操作的缺失

### 官方代码的完整流程：

```python
# 步骤1: 构建原始点集和旋转点集
x_train_new = np.array(final_df[["Lon", "Lat"]])
max_y = np.max(x_train_new[:, 1])
x_train_new_prime = np.array(list(map(
    lambda row: [max_y - row[1], row[0]], 
    x_train_new
)))

# 步骤2: 对两个点集运行算法，共享同一个 heap
skew_heap = []
max_skew_1 = MaxSkewCalculator(
    points, 
    skew_heap,  # 共享 heap
    lambda x: tuple([x[0], x[1]]),  # 原始点集：恒等变换
    math.pi / 10
)
max_skew_2 = MaxSkewCalculator(
    points_prime, 
    skew_heap,  # 共享同一个 heap
    lambda x: tuple([-x[1], x[0]]),  # 旋转点集：90度旋转
    math.pi / 10
)

# 步骤3: 两个实例共享 heap，结果自动合并
max_skew_1.preprocess()
max_skew_2.preprocess()
max_skew_1.train()
max_skew_2.train()
# 最终结果在共享的 skew_heap 中
```

### 复现代码（基础版本 `ray_sweeping_2d.py`）:

```python
# 实验脚本中虽然构建了旋转点集
x_train_new_prime = np.array(list(map(
    lambda row: [max_y - row[1], row[0]], 
    x_train_new
)))

# 但运行算法时没有 vector_transfer 机制
primary_dirs = ray_sweeping_2d(points_primary, ...)  # 没有 vector_transfer 参数
rotated_dirs = ray_sweeping_2d(points_rotated, ...)  # 没有 vector_transfer 参数
```

- **差异**: 
  - 官方：对旋转点集使用 `vector_transfer=lambda x: (-x[1], x[0])` 将方向映射回原始坐标系
  - 复现（基础版本）：没有 `vector_transfer` 机制，旋转点集上的方向无法正确映射
- **影响**: 
  - 旋转点集上找到的方向无法正确映射回原始坐标系
  - 两个点集的结果无法正确合并
  - 可能遗漏某些高偏斜方向

### 复现代码（官方风格版本 `ray_sweeping_2d_official_linkedlist.py`）:

- **✅ 已实现**: 支持 `vector_transfer` 参数
- **✅ 已实现**: 实验脚本 `experiment_ray_sweeping_2d_chicago_crimes_official_style.py` 正确实现了旋转操作

## 总结

主要差异点：
1. **归一化方法**: L1 vs L2
2. **极角排序**: `atan(y/x)` vs `atan2(y, x)`
3. **交点过滤**: 官方最后过滤第一象限，复现保留全圆
4. **初始方向**: 官方从 `(1/x_median, 0)` 开始，复现从第一个交点开始
5. **数据结构**: LinkedList vs 字典+列表
6. **方向转换**: 官方有 `vector_transfer`，基础复现没有，官方风格复现已实现
7. **点集旋转**: 官方完整实现，基础复现部分实现（缺少 vector_transfer），官方风格复现已完整实现

这些差异可能导致：
- 方向向量的数值不同（但方向等价）
- 遍历顺序可能略有不同
- 最终结果的排序可能略有差异
- **基础复现可能遗漏某些高偏斜方向（因为缺少旋转操作的完整支持）**

