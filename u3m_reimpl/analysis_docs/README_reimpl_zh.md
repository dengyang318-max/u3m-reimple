## 总体说明

本说明文档用于解释新建包 `u3m_reimpl` 的设计目的和结构。  
该包是对论文：

- *Mining the Minoria: Unknown, Under-represented, and Under-performing Minority Groups*

中 **2D Ray-sweeping 算法** 的一次独立复现实现：

- 在算法思想上严格参考论文；
- 只在**高层思路**上参考官方仓库 `Mining_U3Ms-main`，但不复制其实现细节；
- 采用更清晰、模块化的结构，方便阅读、实验和后续扩展。

`u3m_reimpl` 与原仓库中的 `utils` 代码完全分离，可以并存、对照使用。

---

## 代码结构概览

新代码位于：

- `u3m_reimpl/`
  - `__init__.py`
  - `geometry.py`
  - `statistics.py`
  - `ray_sweeping_2d.py`

### `u3m_reimpl/__init__.py`

对外导出本次复现中用到的核心类型与函数：

- 几何相关：
  - `Point2D`
  - `Direction2D`
  - `dual_intersection_2d`
  - `normalize_direction`
  - `sort_points_by_polar_angle`

- 统计相关：
  - `ProjectionStats2D`
  - `skew_from_median_point`

示例导入：

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

## 几何与对偶空间工具（`geometry.py`）

该模块对应论文第 3 节的几何解释，主要完成：

- primal 空间中点与方向的封装；
- 对偶空间中超平面交点计算；
- 交点按极角排序，支持 Ray-sweeping。

### `Point2D`

- 用 dataclass 表示二维点 `(x, y)`；
- `as_array()` 将其转为长度为 2 的 NumPy 数组；
- 用于表示数据空间中的样本点 \(t\)。

### `Direction2D`

- 表示二维**单位方向向量** `(dx, dy)`；
- `as_array()` 转为 NumPy 数组；
- 对应论文中用于投影的单位向量 \(f\)。

### `normalize_direction(v)`

- 将一个原始 2D 向量归一化为单位向量，返回 `Direction2D`；
- 使用 **欧式范数** 进行归一化：
  - 与论文中“单位向量在单位圆上”的表述一致；
  - 刻意区别于官方实现中“除以坐标和”的做法。

### `dual_intersection_2d(p1, p2)`

- 实现论文中的对偶变换：

  对于原始点 \(t = (t_1, t_2)\)，其对偶表示为超平面：
  \[
    d(t):\quad t_1 x_1 + t_2 x_2 = 1
  \]

- 两个 primal 点 `p1, p2` 的对偶直线交点即为解 2×2 线性方程组：
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
    \end{bmatrix}
  \]

- 返回值为对偶空间中的交点（NumPy 数组）。

### `sort_points_by_polar_angle(points)`

- 按原点极角对一组 2D 点排序；
- 使用 `atan2(y, x)` 而非简单的 `y/x`：
  - 数值更稳定；
  - 能正确区分四个象限；
- 概念上对应论文中对 k-level 顶点按角度顺序遍历的需要，为 Ray-sweeping 提供“有序遍历”。

---

## 常数时间偏度统计聚合（`statistics.py`）

该模块实现论文中“常数时间更新偏度”的统计聚合思路：

- 通过预先计算少量全局量；
- 对任意方向 \(f\) 在 O(1) 时间（相对于样本数 \(n\)）得到：
  - 投影均值 \(\mu_f\)；
  - 投影标准差 \(\sigma_f\)；
  - 再结合中位数点计算偏度。

### `ProjectionStats2D`

内部存储以下聚合量：

- `n`：样本数；
- `mean`：数据均值点 \(\mu(D)\in\mathbb{R}^2\)；
- `sum_vec`：所有点之和 \(\sum_j t_j\)；
- `xx_sum`：\(\sum_j x_j^2\)；
- `yy_sum`：\(\sum_j y_j^2\)；
- `xy_sum`：\(\sum_j x_j y_j\)。

这些量正是论文中将
\[
  \sum_j (t_j^\top f_i - \mu_i)^2
\]
展开为若干可预计算项时所用到的那一类。

#### `ProjectionStats2D.from_points(points)`

- 输入一批 `Point2D`；
- 一次遍历 O(n) 完成所有聚合量的构建；
- 对应论文中“预处理阶段”的工作。

#### `projected_mean(direction)`

- 计算投影均值：
  \[
    \mu_f = \mu(D)^\top f
  \]
- 时间复杂度 O(1)；
- 直接对应论文中对均值的公式。

#### `projected_std(direction)`

- 计算投影标准差 \(\sigma_f\)，时间复杂度 O(1)；
- 使用论文中的代数展开：
  \[
    \sum_j (t_j^\top f - \mu_f)^2
      =
      \sum_j (t_j^\top f)^2
      - 2 \mu_f \sum_j t_j^\top f
      + n \mu_f^2.
  \]
- 其中：
  - \(\sum_j (t_j^\top f)^2\) 通过 `xx_sum, yy_sum, xy_sum` 写成关于 \(f\) 的二次型；
  - 其它项使用 `mean` 与 `sum_vec` 直接计算。

这一实现对应论文中“常数时间更新标准差/偏度”的核心推导。

### `skew_from_median_point(stats, direction, median_point)`

- 输入：
  - 全局统计量 `stats`（`ProjectionStats2D`）；
  - 某个方向 `direction`（`Direction2D`）；
  - 某个中位数点 `median_point`（`Point2D`）；

- 利用中位数点计算投影偏度：
  \[
    \text{skew}(f)
    =
    \left|\mu_f - t_m^\top f\right| / \sigma_f,
  \]
  其中 \(t_m\) 为中位数点，\(t_m^\top f\) 为在该方向上的投影中位数。

- 完全对应论文中的思路：在一个 median region 内，中位数点固定，因此可以用该点代替每次重新求中位数。
- 有意省略 Pearson 偏度中的系数 3，因为它并不影响“按偏度排序方向”的结果。

---

## 简化版 2D Ray-sweeping 接口（`ray_sweeping_2d.py`）

该模块基于上述几何与统计基础，给出了一个**可运行的、简化版的** Ray-sweeping 算法接口：

- 思路上和论文一致：对偶交点 → 角度排序 → Ray-sweeping → 常数时间偏度 → 堆选 top-k；
- 实现上与官方 `utils/ray_sweep.py` 完全独立，数据结构与代码风格不同；
- 重点是为后续完整复现论文中的“median region + skeleton graph”做铺垫。

### 数据类型

- `MedianSegment`：
  - 表示一段方向范围及其对应的同一中位数点；
  - 可以看作论文中“median region”的简化版本。

- `SkewDirection`：
  - 封装一个候选方向及其偏度值；
  - 便于统一处理和返回结果。

### 内部辅助函数

#### `_build_projection_stats(points)`

- 将原始 `(x, y)` 转成 `Point2D` 列表；
- 构建 `ProjectionStats2D` 实例，为后续常数时间偏度计算做准备。

#### `_enumerate_intersections(points)`

- 朴素的 O(n²) 实现：
  - 对所有点对 `(p_i, p_j)` 调用 `dual_intersection_2d`，得到对偶空间中的交点；
  - 仅保留第一象限中的交点（与论文/官方实现一致）；
  - 调用 `sort_points_by_polar_angle` 按极角排序。

- 概念上等价于构造并按角度遍历 k-level 安排的骨架图（skeleton）。  
  论文中给出了更高效的输出敏感算法，这里先用 O(n²) 版本保证实现简单易懂，后续可以替换为更高效实现。

### 主函数：`ray_sweeping_2d`

```python
def ray_sweeping_2d(
    points: Iterable[Tuple[float, float]],
    top_k: int = 10,
    min_angle_step: float = np.pi / 90.0,
) -> List[SkewDirection]:
    ...
```

当前版本实现的是 Ray-sweeping 算法的“基础骨架”：

1. **预处理阶段**
   - 构造 `Point2D` 列表和 `ProjectionStats2D`，对应论文中的常数时间偏度预处理；
   - 枚举所有对偶交点并按极角排序；
   - 使用按 x 排序的中位数点作为初始中位数（思路与官方代码一致，但实现不同）。

2. **Ray-sweeping 扫描**
   - 遍历每个交点 `(x, y)`：
     - 通过 `normalize_direction` 得到单位方向；
     - 与上一方向对比夹角，小于 `min_angle_step` 的跳过，以控制方向采样密度；
     - 调用 `skew_from_median_point(stats, direction, median_point)` 以 O(1) 时间计算偏度；
     - 将 `(方向, 偏度)` 封装为 `SkewDirection`，并以负偏度值入堆（用最小堆模拟最大堆）。

3. **top-k 提取**
   - 从堆中弹出 `top_k` 个偏度最大的方向，返回 `List[SkewDirection]`。

该版本已经具备：

- 与论文一致的总体结构（对偶 + 角度遍历 + 常数时间偏度 + 堆选 top-k）；
- 清晰的代码组织，方便后续按论文逐步替换为“严格的 median region + skeleton graph”实现。

---

## 使用示例

以下是一个最小示例，展示如何调用当前的简化版 `ray_sweeping_2d`：

```python
from u3m_reimpl.ray_sweeping_2d import ray_sweeping_2d

# 一组二维点（可以参考论文的 toy 示例）
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
    print(f"#{i}: dir={d}, skew={cand.skew_value:.4f}")
```

通过该示例你可以快速检查：

- 几何工具是否正常工作（对偶交点、极角排序）；
- 统计聚合是否正确计算均值 / 标准差 / 偏度；
- 堆选 top-k 的逻辑是否符合预期。

---

## 与论文的对应关系及后续工作方向

当前 `u3m_reimpl` 中的实现已经完成了论文中 Ray-sweeping 算法的**准备工作和基础方法**：

- **几何与对偶空间部分**（`geometry.py`）：
  - 对偶变换、交点计算、角度排序，直接对应论文第 3 节的几何解释。

- **常数时间偏度计算部分**（`statistics.py`）：
  - 通过预计算 `xx_sum, yy_sum, xy_sum, sum_vec, mean` 等，实现了论文中对标准差/偏度 O(1) 更新的推导。

- **Ray-sweeping 骨架部分**（`ray_sweeping_2d.py`）：
  - 提供了从原始点集 → 对偶交点 → 方向 → 偏度 → top-k 的完整数据流；
  - 与官方实现解耦，便于在此基础上按照论文逐步补足：
    - 明确的 median region 图结构（k-level skeleton 图）；
    - 只在跨越中位数区域边界时更新中位数点和聚合值；
    - 集成 tail 上模型损失的计算和阈值过滤（对应 Problem 1 中的 \(L_{D_g}(\theta) - L_D(\theta) \ge \tau\)）。

本 README 聚焦于**复现算法前需要完成的准备工作和基础方法**。在此之上，你可以继续对照论文，把完整的 Ray-sweeping 中位数区域遍历和 Minoria 发现逻辑一步步补齐。


