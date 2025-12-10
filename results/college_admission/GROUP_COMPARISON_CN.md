# College Admission — Group A vs Group B vs Group C：参数设置与结果差异对比

## 参数配置总结

### Group A — 基线配置
**关键设置：** `disable_vector_transfer: False`（vector_transfer **开启**）

**运行：** 6个实验
- `20251208_205408`: linkedlist=False, first_init=False, min_shift=开启
- `20251208_205427`: linkedlist=False, first_init=True, min_shift=开启
- `20251208_205445`: linkedlist=False, first_init=False, min_shift=关闭
- `20251208_205538`: linkedlist=True, first_init=False, min_shift=开启
- `20251208_205557`: linkedlist=True, first_init=True, min_shift=开启
- `20251208_205619`: linkedlist=True, first_init=False, min_shift=关闭

**共同参数：**
- `use_official_style`: True
- `disable_vector_transfer`: **False**（vector_transfer 启用）
- `use_l2_norm`: False（L1归一化）
- `top_k`: 10
- `min_angle_step`: 0.314 (π/10)
- `n_samples`: None（使用全部 400 条记录）

**参数变化（Group A 内部）：**
- `use_linkedlist`: False（3次运行）/ True（3次运行）
- `use_first_intersection_init`: False（4次运行）/ True（2次运行）
- `disable_min_shift`: False（4次运行）/ True（2次运行）

**观察：** 尽管存在这些变化，所有6次运行的百分位表几乎完全相同，证实了当 vector_transfer 开启时，`linkedlist`、`first_init` 和 `min_shift` 的影响可忽略。

---

### Group B — Vector-transfer 关闭
**关键设置：** `disable_vector_transfer: True`（vector_transfer **关闭**）

**运行：** 5个实验
- `20251208_205502`: linkedlist=False, first_init=False, min_shift=开启
- `20251208_205521`: linkedlist=False, first_init=True, min_shift=关闭
- `20251208_205636`: linkedlist=True, first_init=False, min_shift=开启
- `20251208_205653`: linkedlist=True, first_init=True, min_shift=关闭
- `20251208_205816`: 官方移植（use_official_style=False），vector_transfer 实际关闭

**共同参数：**
- `disable_vector_transfer`: **True**（vector_transfer 禁用）
- `use_l2_norm`: False（L1归一化）
- `top_k`: 10
- `min_angle_step`: 0.314 (π/10)
- `n_samples`: None（使用全部 400 条记录）

**参数变化（Group B 内部）：**
- `use_linkedlist`: False（2次运行）/ True（2次运行）/ N/A（1次官方移植）
- `use_first_intersection_init`: False（2次运行）/ True（2次运行）/ N/A（1次官方移植）
- `disable_min_shift`: False（2次运行）/ True（2次运行）/ N/A（1次官方移植）

**观察：** Group B 的运行与 Group A 存在一致的差异，组内因相同次要参数存在微小变化。

---

### Group C — L2 归一化（vector_transfer 开启）
**关键设置：** `use_l2_norm: True`（L2 归一化），`disable_vector_transfer: False`（vector_transfer **开启**）

**运行：** 4个实验
- `20251208_205710`: linkedlist=False, first_init=False, min_shift=开启
- `20251208_205727`: linkedlist=True, first_init=False, min_shift=开启
- `20251208_205744`: linkedlist=False, first_init=True, min_shift=开启
- `20251208_205759`: linkedlist=True, first_init=True, min_shift=关闭

**共同参数：**
- `use_official_style`: True
- `disable_vector_transfer`: **False**（vector_transfer 启用）
- `use_l2_norm`: **True**（L2 归一化）
- `top_k`: 10
- `min_angle_step`: 0.314 (π/10)
- `n_samples`: None（使用全部 400 条记录）

**参数变化（Group C 内部）：**
- `use_linkedlist`: False（2次运行）/ True（2次运行）
- `use_first_intersection_init`: False（2次运行）/ True（2次运行）
- `disable_min_shift`: False（3次运行）/ True（1次运行）

**观察：** Group C 的结果与 Group A 非常相似，只有微小、一致的漂移。这证实了 L2 vs L1 归一化的影响远小于关闭 vector_transfer。

---

## 核心参数差异

| 参数 | Group A | Group B | Group C |
|------|---------|---------|---------|
| **`disable_vector_transfer`** | **False**（开启） | **True**（关闭） | **False**（开启） |
| **`use_l2_norm`** | **False**（L1） | **False**（L1） | **True**（L2） |
| `use_linkedlist` | 混合 | 混合 | 混合 |
| `use_first_intersection_init` | 混合 | 混合 | 混合 |
| `disable_min_shift` | 混合 | 混合 | 混合 |

**系统性差异：**
- **Group A vs Group B：** `disable_vector_transfer`（开启 vs 关闭）— **主要影响**
- **Group A vs Group C：** `use_l2_norm`（L1 vs L2）— **次要影响**
- **Group B vs Group C：** `disable_vector_transfer`（关闭 vs 开启）和 `use_l2_norm`（L1 vs L2）均不同

---

## 结果差异 — 百分位分析

**使用的百分位：** q = 1.0, 0.5, 0.2, 0.1, 0.08, 0.04

### 1. F1-score

**Group A（代表：205408/205427/205445）：**

**Top-1：**
- q=1.0: 0.320
- q=0.5: 0.254
- q=0.2: 0.308
- q=0.1: **0.000**（F1 崩溃）
- q=0.08: **0.000**
- q=0.04: **0.000**

**Top-2：**
- q=1.0: 0.321
- q=0.5: 0.375
- q=0.2: 0.409
- q=0.1: 0.364
- q=0.08: 0.444
- q=0.04: 0.400

**Top-3：**
- q=1.0: 0.320
- q=0.5: 0.350
- q=0.2: 0.364
- q=0.1: 0.286
- q=0.08: 0.250
- q=0.04: 0.500

**Group B（代表：205502, 205521, 205636, 205653）：**

**Top-1：**
- q=1.0: 0.320
- q=0.5: 0.407–0.400
- q=0.2: 0.492–0.481
- q=0.1: **0.562–0.452**（F1 保持）
- q=0.08: **0.571–0.538**（F1 保持）
- q=0.04: **0.533–0.625**（F1 保持）

**Top-2：**
- q=1.0: 0.320–0.307
- q=0.5: 0.400–0.407
- q=0.2: 0.481–0.483
- q=0.1: 0.452–0.562
- q=0.08: 0.538–0.571
- q=0.04: 0.533–0.545

**Top-3：**
- q=1.0: 0.307–0.320
- q=0.5: 0.128–0.400
- q=0.2: **0.000–0.517**（因运行而异；部分崩溃，部分保持）
- q=0.1: **0.000–0.529**（变化）
- q=0.08: **0.000–0.500**（变化）
- q=0.04: **0.000–0.462**（变化）

**Group C（代表：205710, 205744）：**

**Top-1：**
- q=1.0: 0.320
- q=0.5: 0.254
- q=0.2: 0.250–0.308
- q=0.1: **0.000**（F1 崩溃，与 Group A 相同）
- q=0.08: **0.000**
- q=0.04: **0.000**

**Top-2：**
- q=1.0: 0.320–0.321
- q=0.5: 0.375
- q=0.2: 0.409
- q=0.1: 0.364
- q=0.08: 0.444
- q=0.04: 0.400

**Top-3：**
- q=1.0: 0.320
- q=0.5: 0.350
- q=0.2: 0.364
- q=0.1: 0.286–0.381
- q=0.08: 0.250–0.353
- q=0.04: 0.400–0.500

**关键发现：**
- **Top-1：** Group B 在 q=0.1–0.04 保持 F1（0.45–0.63），而 Group A 和 Group C 都崩溃至 0。这是最显著的差异。
- **Top-2：** 三组都保持 F1，Group B 在紧百分位（q=0.1–0.04）显示略高的值。Group C 与 Group A 几乎完全相同。
- **Top-3：** Group B 显示更多变异性；部分运行保持 F1，部分崩溃（与 Group A/C 模式相似）。Group C 与 Group A 相比有微小变化（例如，205727/205759 有略微不同的值）。

---

### 2. Accuracy（准确率）

**Group A：**

**Top-1：**
- q=0.1: 0.775
- q=0.08: 0.844
- q=0.04: 0.875（高，因为尾部主要是负例）

**Top-2/3：**
- q=0.1: 0.641–0.750
- q=0.08: 0.677–0.812
- q=0.04: 0.800–0.875

**Group B：**

**Top-1：**
- q=0.1: 0.575–0.650（低于 Group A）
- q=0.08: 0.625（较低）
- q=0.04: 0.562–0.625（远低于 Group A）

**Top-2：**
- q=0.1: 0.575–0.650
- q=0.08: 0.625
- q=0.04: 0.562–0.583

**Top-3：**
- q=0.1: 0.590–0.825（变化；部分运行高如 Group A）
- q=0.08: 0.562–0.812（变化）
- q=0.04: 0.533–0.812（变化）

**Group C：**

**Top-1：**
- q=0.1: 0.775
- q=0.08: 0.844
- q=0.04: 0.875（与 Group A 相同）

**Top-2/3：**
- q=0.1: 0.641–0.750
- q=0.08: 0.677–0.812
- q=0.04: 0.800–0.875

**关键发现：**
- Group B 在紧百分位（q=0.1–0.04）显示**较低的准确率**，特别是 Top-1/2，在 q=0.04 处 Group A 和 Group C 达到 0.84–0.88，而 Group B 降至 0.56–0.63。这反映了 Group B 捕获了更多正例（更高的 F1），可能包含更多假正例，从而降低准确率。
- **Group C 的准确率与 Group A 几乎完全相同**，证实了 L2 vs L1 归一化对准确率的影响极小。

---

### 3. F/M Ratio（女性/男性比例）

**Group A：**
- Top-1: F/M ≈ 1.0–1.2（跨百分位）
- Top-2: F/M ≈ 0.8–2.0（更易变）
- Top-3: F/M ≈ 1.0–1.3

**Group B：**
- Top-1: F/M ≈ 0.9–1.3（相似范围）
- Top-2: F/M ≈ 0.7–1.4（更易变，包含较低值）
- Top-3: F/M ≈ 0.6–1.5（更易变）

**Group C：**
- Top-1: F/M ≈ 1.0–1.2（与 Group A 相同）
- Top-2: F/M ≈ 0.8–2.0（与 Group A 相似）
- Top-3: F/M ≈ 0.6–1.3（比 Group A 略易变）

**关键发现：**
- Group B 显示**更易变的 F/M 比例**，特别是在紧百分位，部分运行显示比例 < 1.0（男性主导）或 > 1.4（女性主导），表明发现的尾部存在更强的人口统计学偏移。
- **Group C 的 F/M 比例与 Group A 非常相似**，只有微小变化，证实了归一化方法对人口统计学组成的影响极小。

---

## 可视化差异（3×3 九宫格）

**Group A（例如，`20251208_205408/official_utils_top_grid.png`）：**
- **Full 面板：** 统一的方向足迹；标准方向
- **Tail 面板：** 稀疏的尾部区域；很少的集中簇
- **Density 面板：** 低正例密度；分散的点

**Group B（例如，`20251208_205502/official_utils_top_grid.png`，`20251208_205636/official_utils_top_grid.png`）：**
- **Full 面板：** 不同的方向方向（无 vector_transfer 映射回原空间）
- **Tail 面板：** 更集中、更紧密的尾部区域；更高的点密度
- **Density 面板：** 更密集的正例斑块；更明显的簇

**Group C（例如，`20251208_205710/official_utils_top_grid.png`，`20251208_205744/official_utils_top_grid.png`）：**
- **Full 面板：** 与 Group A 相似（相同的 vector_transfer 映射）
- **Tail 面板：** 与 Group A 非常相似；稀疏、分散
- **Density 面板：** 与 Group A 相似；低正例密度

**可视化观察：** Group C 的九宫格与 Group A 几乎无法区分，证实了 L2 vs L1 归一化对可视化的影响极小。

---

## 总结：实际变化

### 参数影响层次

1. **`disable_vector_transfer`**（开启 vs 关闭）：
   - **主要影响：** 改变方向集；改变尾部组成
   - **结果：** Group B 找到在紧百分位（q=0.1–0.04）保持 Top-1 F1 的方向，而 Group A 和 Group C 崩溃至 F1=0
   - **证据：** Top-1 F1 在 q=0.04：Group B = 0.53–0.63 vs Group A = 0.00 vs Group C = 0.00

2. **`use_l2_norm`**（L1 vs L2）：
   - **次要影响：** 微小、一致的漂移；远小于 vector_transfer
   - **结果：** Group C 显示与 Group A 几乎相同的结果，只有微小变化（例如，Top-1 F1 在 q=0.2：Group C = 0.250 vs Group A = 0.308）
   - **证据：** Group C 的百分位表与 Group A 几乎完全相同；可视化九宫格无法区分

3. **`use_linkedlist`、`use_first_intersection_init`、`disable_min_shift`：**
   - **可忽略影响：** 仅在每组内产生微小数值漂移
   - **证据：** 每组内的运行几乎完全相同，尽管存在这些变化

### 核心结果差异

| 指标 | Group A | Group B | Group C | 关键差异 |
|------|---------|---------|---------|----------|
| **F1-score Top-1 (q=0.04)** | 0.000 | 0.53–0.63 | 0.000 | **Group B 保持 F1；A & C 崩溃** |
| **F1-score Top-1 (q=0.1)** | 0.000 | 0.45–0.56 | 0.000 | **Group B 保持 F1；A & C 崩溃** |
| **Accuracy Top-1 (q=0.04)** | 0.84–0.88 | 0.56–0.63 | 0.84–0.88 | Group B 较低；A & C 相似 |
| **F/M 比例变异性** | 中等 | 高 | 中等 | Group B 显示更强的偏移 |
| **尾部集中度** | 稀疏、分散 | 集中、密集 | 稀疏、分散 | Group B 更结构化；A & C 相似 |

### 解释

- **Group A（vector_transfer 开启，L1）：** 方向被映射回原空间，产生稀疏尾部。在紧百分位（q≤0.1），Top-1 F1 崩溃至 0，因为正例消失。Top-2/3 在某些情况下可能保持 F1，但 Top-1 始终失败。

- **Group B（vector_transfer 关闭，L1）：** 方向不映射回原空间；算法直接探索旋转空间。这找到不同的方向，捕获更密集的正例区域，即使在 q=0.04 也保持 Top-1 F1（0.53–0.63）。然而，这以较低的准确率为代价（0.56–0.63 vs 0.84–0.88），因为捕获了更多正例，可能包含更多假正例。

- **Group C（vector_transfer 开启，L2）：** 使用 L2 归一化而非 L1，但 vector_transfer 仍开启。结果与 Group A 几乎完全相同，只有微小变化（例如，Top-1 F1 在 q=0.2：0.250 vs 0.308）。这证实了归一化方法（L1 vs L2）的影响远小于 vector_transfer。

**结论：**
- `disable_vector_transfer` 参数是**主要系统性驱动**有意义结果差异的因素（Group A/C vs Group B）。
- `use_l2_norm` 参数具有**次要影响**（Group A vs Group C），仅导致微小、一致的漂移。
- 所有其他参数（`linkedlist`、`first_init`、`min_shift`）仅导致可忽略的数值漂移。
- **关键权衡：** Group B 在紧百分位保持 F1 但牺牲准确率，而 Group A 和 Group C 在紧百分位达到高准确率但完全失去 F1。

