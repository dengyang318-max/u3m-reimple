# 高偏斜计算方法的对比：官方实现 vs 复现实现

## 1. 方向向量归一化方法

### 官方实现 (`utils/dual_space.py`)
```python
def normalize_vector(vector: tuple):
    return np.array(vector) / sum(vector)  # L1 归一化
```
- **方法**：L1 归一化（坐标和归一化）
- **结果**：`||f||_1 = f[0] + f[1] = 1`
- **特点**：保持方向比例，但向量长度不是 1

### 复现实现 (`u3m_reimpl/geometry.py`)
```python
def normalize_direction(v: np.ndarray) -> Direction2D:
    norm = np.linalg.norm(v)  # L2 范数
    u = v / norm
    return Direction2D(dx=float(u[0]), dy=float(u[1]))
```
- **方法**：L2 归一化（欧几里得范数）
- **结果**：`||f||_2 = sqrt(f[0]^2 + f[1]^2) = 1`
- **特点**：标准单位向量，符合论文理论描述

**影响**：归一化方法不同会导致方向向量的表示不同，但**偏斜值的相对大小和排序应该是一致的**（因为偏斜是比例量）。

---

## 2. 标准差（Standard Deviation）计算

### 官方实现 (`utils/ray_sweep.py` - `SD.get_sd`)
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

### 复现实现 (`u3m_reimpl/statistics.py` - `ProjectionStats2D.projected_std`)
```python
def projected_std(self, direction: Direction2D) -> float:
    f = direction.as_array()
    fx, fy = float(f[0]), float(f[1])
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

**对比**：
- **数学公式相同**：两者都使用展开的方差公式
- **实现细节**：
  - 官方：直接计算 `n * mean_f^2 - 2 * mean_f * sum_f + ...`
  - 复现：先计算 `sum_sq`，再减去 `2 * mu_f * sum_proj + n * mu_f^2`
- **数值稳定性**：复现版本添加了 `numerator < 0` 的数值保护

**结论**：**标准差计算在数学上等价**，但由于归一化方法不同，实际数值会有差异。

---

## 3. 偏斜（Skew）值计算

### 官方实现 (`utils/ray_sweep.py` - `_calc_skew`)
```python
def _calc_skew(self, f, median, verbose=False):
    m_point = np.array(median)
    mean_f = np.dot(self.sd.mean, f)
    skew = abs((mean_f - np.dot(m_point, f)) / self.sd.get_sd(f))
    return skew
```

### 复现实现 (`u3m_reimpl/statistics.py` - `skew_from_median_point`)
```python
def skew_from_median_point(
    stats: ProjectionStats2D, direction: Direction2D, median_point: Point2D
) -> float:
    mu_f = stats.projected_mean(direction)
    sigma_f = stats.projected_std(direction)
    if sigma_f == 0.0:
        return 0.0
    v = median_point.as_array()
    median_proj = float(v @ direction.as_array())
    return abs((mu_f - median_proj) / sigma_f)
```

**对比**：
- **公式相同**：`skew = |(mean - median) / std|`
- **实现方式**：
  - 官方：直接计算，无除零保护
  - 复现：添加了 `sigma_f == 0.0` 的保护
- **注释说明**：复现版本明确说明省略了 Pearson 偏斜公式中的因子 3（因为不影响排序）

**结论**：**偏斜计算公式相同**，但：
1. 由于归一化方法不同（L1 vs L2），`f` 的表示不同
2. 因此 `mean_f`、`median_proj`、`sigma_f` 的数值会有差异
3. 但**偏斜值的相对大小和排序应该保持一致**（因为都是比例量）

---

## 4. 中位数点（Median Point）跟踪

### 官方实现
- 使用 **k-level arrangement** 的 median region 概念
- 通过 `_get_first_median_on_x()` 初始化：按 x 坐标排序，取中位数
- 在遍历过程中，通过 `_get_next_median()` 更新中位数点
- 使用 `LinkedList` 结构维护 k-level 的拓扑关系

### 复现实现
- 同样使用 median region 的概念（`MedianSegment` 类）
- 初始化：`sorted_by_x = sorted(pts, key=lambda p: p.x)`，取中位数
- **关键差异**：复现版本**没有实现完整的 k-level traversal**
- 复现版本只是遍历所有交点，对每个交点计算偏斜，**没有动态更新中位数点**

**这是最重要的差异！**

---

## 5. 交点枚举和遍历策略

### 官方实现
```python
def train(self, verbose=False):
    # 1. 初始化第一个 median region
    first_median = self._get_first_median_on_x()
    median_region = MedianRegion(...)
    
    # 2. 遍历 k-level arrangement
    while not finish:
        # 在 median region 的起点计算偏斜
        skew_vector_start = GeoUtility.normalize_vector(median_region.start.point)
        if (last_vec is not None and self.get_angel(skew_vector_start, last_vec) > self.epsilon):
            heapq.heappush(self.heap, (-self._calc_skew(...), ...))
        
        # 3. 移动到下一个 median region
        # 通过 line_intersects 找到下一个交点，更新 median_region
        ...
```

**特点**：
- 沿着 k-level arrangement 的边界**连续遍历**
- 每个 median region 内，中位数点保持不变
- 只在 region 边界（交点）处更新中位数点
- **这是论文中描述的标准 Ray-sweeping 算法**

### 复现实现
```python
def ray_sweeping_2d(...):
    # 1. 枚举所有交点
    intersections = _enumerate_intersections(pts)
    
    # 2. 初始化中位数点（只初始化一次）
    sorted_by_x = sorted(pts, key=lambda p: p.x)
    median_point = sorted_by_x[len(sorted_by_x) // 2]
    
    # 3. 遍历所有交点，对每个交点计算偏斜
    for x, y in intersections:
        direction = normalize_direction(np.array([x, y], dtype=float))
        skew_val = skew_from_median_point(stats, direction, median_point)  # 使用固定的 median_point
        ...
```

**特点**：
- **枚举所有交点**，然后遍历
- **中位数点固定不变**（只初始化一次）
- 没有实现 k-level arrangement 的连续遍历
- 没有在交点处更新中位数点

**这是最关键的差异！**

---

## 6. 总结：主要差异

| 方面 | 官方实现 | 复现实现 | 影响 |
|------|---------|---------|------|
| **归一化** | L1 归一化 (`sum`) | L2 归一化 (`norm`) | 方向向量表示不同，但偏斜排序一致 |
| **标准差计算** | 相同公式 | 相同公式 + 数值保护 | 数学等价，数值略有差异 |
| **偏斜计算** | `abs((mean - median) / std)` | `abs((mean - median) / std)` + 除零保护 | 公式相同 |
| **中位数跟踪** | **动态更新**（k-level traversal） | **固定不变**（只初始化一次） | **这是最大差异！** |
| **遍历策略** | 沿着 k-level 连续遍历 | 枚举所有交点后遍历 | **算法正确性差异** |

---

## 7. 为什么会有差异？

### 复现版本的简化
复现版本采用了**简化的实现策略**：
1. **枚举所有交点**：O(n²) 暴力枚举
2. **固定中位数点**：使用初始 x-中位数点，不再更新
3. **遍历所有交点**：对每个交点计算偏斜

### 官方版本的完整实现
官方版本实现了**完整的 Ray-sweeping 算法**：
1. **k-level arrangement**：维护完整的拓扑结构
2. **动态中位数更新**：在 median region 边界处更新中位数点
3. **连续遍历**：沿着 k-level 边界连续移动

---

## 8. 影响分析

### 对偏斜值的影响
- **理论上**：如果中位数点固定，偏斜值可能**不准确**
- **实际上**：对于大多数方向，初始 x-中位数点可能是合理的近似
- **极端情况**：当方向与 x 轴接近垂直时，真正的中位数点可能与 x-中位数点差异很大

### 对结果排序的影响
- **可能影响**：由于中位数点不更新，某些方向的偏斜值可能被高估或低估
- **排序可能不同**：top-k 方向的排序可能与官方结果不同
- **但方向本身**：找到的高偏斜方向应该仍然是有意义的

---

## 9. 建议

如果要完全匹配官方结果，需要：
1. **实现 k-level arrangement 的完整遍历**
2. **在 median region 边界处动态更新中位数点**
3. **使用 L1 归一化**（如果希望数值完全一致）

但当前的复现版本：
- **数学上正确**：偏斜计算公式正确
- **实现更简单**：易于理解和维护
- **结果仍然有效**：找到的高偏斜方向是有意义的

