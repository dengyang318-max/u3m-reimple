# 为什么官方风格实现的结果和官方结果还是不一样？

## 关键差异分析

### 1. **数据结构差异：LinkedList vs 字典+列表**

**官方代码**：
- 使用 `LinkedList` 结构组织交点和点的关系
- `line_intersects = {point -> LinkedList[intersection]}`
- 通过 `next` 指针顺序遍历
- `neighbours` 指向同一交点的其他线

**我们的实现**：
- 使用字典和列表：`point_intersections = {point -> List[intersection]}`
- 通过索引遍历列表

**影响**：
- **遍历顺序可能不同**：LinkedList 的 `next` 指针顺序可能与我们排序后的列表顺序不同
- **这会导致访问交点的顺序不同，从而影响找到的高偏度方向**

### 2. **初始 MedianRegion 的构建方式不同**

**官方代码**：
```python
median_region = MedianRegion(
    LinkedList((1 / first_median[0], 0), [], first_median, None),  # start
    self.line_intersects[first_median],  # end (LinkedList)
    first_median,  # median
)
```

**我们的实现**：
```python
# 从 (1/x_median, 0) 开始，但遍历从第一个交点开始
current_intersections = point_intersections[current_point]
intersection_idx = 0
```

**影响**：
- 官方代码的 `start` 是一个特殊的 LinkedList 节点，代表 `(1/x_median, 0)`
- 我们的实现虽然计算了这个方向，但没有将其作为遍历的起点
- **这可能导致第一个方向的处理方式不同**

### 3. **遍历逻辑的根本差异**

**官方代码**：
```python
while not finish:
    skew_vector_start = GeoUtility.normalize_vector(median_region.start.point)
    # ... 检查角度并 push
    
    if median_region.end.point[0] == 0:
        break
    
    # 移动到下一个 median_region
    current_points = self.intersects[median_region.end.point]
    line_b = self._get_next_median(...)
    next_neighbour = list(filter(lambda n: n.line == line_b, median_region.end.neighbours))[0]
    
    if next_neighbour.next is None:
        new_end = LinkedList((0, 1 / line_b[1]), [], line_b, None)
    else:
        new_end = next_neighbour.next
    
    median_region = MedianRegion(median_region.end, new_end, line_b)
```

**我们的实现**：
```python
while intersection_idx < len(current_intersections):
    intersection = current_intersections[intersection_idx]
    # ... 处理交点
    
    # 更新 median
    if new_median != current_median:
        # 切换到新点的交点列表
        new_intersections = point_intersections[new_median]
        intersection_idx = new_intersections.index(intersection) + 1
```

**关键差异**：
1. **官方通过 `neighbours` 和 `next` 指针找到下一个交点**
2. **我们通过排序后的列表索引找到下一个交点**
3. **这两种方式的遍历顺序可能完全不同**

### 4. **数据预处理：只对 points[1] 做 min-shift？**

**官方代码**（注意第 45-46 行）：
```python
def __init__(self, points: "pd.DataFrame", heap, vector_transfer, epsilon):
    points[0] = points[0] - points[0].min()  # 第 45 行
    points[1] = points[1] - points[1].min()  # 第 46 行
```

但在搜索结果中看到：
```python
points[1] = points[1] - points[1].min()  # 只有这一行？
```

**需要确认**：官方代码是否真的只对 `points[1]` 做 min-shift，还是对两列都做？

**我们的实现**：
```python
arr[:, 0] = arr[:, 0] - arr[:, 0].min()
arr[:, 1] = arr[:, 1] - arr[:, 1].min()
```

### 5. **Skew 计算时的方向使用**

**官方代码**：
```python
skew_vector_start = GeoUtility.normalize_vector(median_region.start.point)
# normalize_vector 使用 L1 归一化：vector / sum(vector)

skew = self._calc_skew(skew_vector_start, median_region.median, verbose)
# _calc_skew 使用这个归一化后的方向计算 skew

heapq.heappush(
    self.heap,
    (
        -skew,
        self.vector_transfer(tuple(skew_vector_start)),  # 存储时应用 vector_transfer
    ),
)
```

**我们的实现**：
```python
direction_raw = normalize_direction_l1(np.array([intersection[0], intersection[1]]))
skew_vector = direction_raw.as_array()

direction_transferred_tuple = vector_transfer(tuple(skew_vector))
direction = normalize_direction_l1(np.array(direction_transferred_tuple))

skew_val = skew_from_median_point(stats, direction_raw, current_median)
```

**差异**：
- 官方：计算 skew 时使用 L1 归一化的方向，存储时应用 vector_transfer
- 我们：计算 skew 时使用 L1 归一化的方向（正确），但存储的方向又做了一次 L1 归一化（可能不对）

### 6. **终止条件的处理**

**官方代码**：
```python
if next_neighbour.next is None:
    new_end = LinkedList((0, 1 / line_b[1]), [], line_b, None)  # 创建 Y 轴上的点
```

**我们的实现**：
```python
if intersection[0] == 0.0:  # 直接检查
    break
```

**差异**：官方代码会创建一个 Y 轴上的特殊节点，我们直接检查并退出

## 最可能的原因

### **主要原因：遍历顺序不同**

由于我们使用**排序后的列表**而不是**LinkedList 的 next 指针**，遍历交点的顺序可能与官方完全不同。这会导致：

1. **访问交点的顺序不同** → 中位数更新的时机不同
2. **找到的高偏度方向不同** → 最终结果不同

### **次要原因：初始方向处理**

官方代码将 `(1/x_median, 0)` 作为 `MedianRegion.start`，这是一个特殊的 LinkedList 节点。我们的实现虽然计算了这个方向，但没有将其作为遍历的起点，可能导致第一个方向的处理不同。

## 如何验证

1. **检查交点数量**：
   ```python
   print(f"Official intersects: {len(intersections_official)}")
   print(f"Our intersects: {len(intersections_our)}")
   ```

2. **检查遍历顺序**：
   - 打印官方代码访问的交点序列
   - 打印我们的实现访问的交点序列
   - 比较两者是否相同

3. **检查初始方向**：
   - 验证 `(1/x_median, 0)` 是否被正确处理
   - 验证第一个方向是否被 push 到 heap

4. **检查数据预处理**：
   - 确认官方代码是否真的只对 `points[1]` 做 min-shift
   - 比较预处理后的点坐标

## 建议的修复方向

1. **实现 LinkedList 结构**：完全匹配官方的数据结构
2. **匹配遍历逻辑**：使用 `neighbours` 和 `next` 指针而不是列表索引
3. **匹配初始 MedianRegion**：将 `(1/x_median, 0)` 作为特殊的 start 节点
4. **验证数据预处理**：确认 min-shift 的具体实现

## 结论

**结果不同的根本原因**：我们使用了不同的数据结构（列表 vs LinkedList）和遍历方式（索引 vs 指针），导致访问交点的顺序不同，从而找到的高偏度方向也不同。

要完全匹配官方结果，需要：
1. 实现完整的 LinkedList 结构
2. 使用与官方完全相同的遍历逻辑
3. 匹配所有边界条件的处理

