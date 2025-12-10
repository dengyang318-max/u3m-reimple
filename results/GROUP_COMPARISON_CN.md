# Chicago Crimes — Group A vs Group B：参数设置与结果差异对比

## 参数配置总结

### Group A — 基线配置
**关键设置：** `disable_vector_transfer: False`（vector_transfer **开启**）

**运行：** 6个实验
- `20251208_173917`: linkedlist=False, first_init=True, min_shift=开启
- `20251208_174132`: linkedlist=False, first_init=False, min_shift=开启
- `20251208_174346`: linkedlist=False, first_init=False, min_shift=关闭
- `20251208_174903`: linkedlist=True, first_init=False, min_shift=开启
- `20251208_175104`: linkedlist=True, first_init=True, min_shift=开启
- `20251208_175253`: linkedlist=True, first_init=False, min_shift=关闭

**共同参数：**
- `use_official_style`: True
- `disable_vector_transfer`: **False**（vector_transfer 启用）
- `use_l2_norm`: None/False（L1归一化）
- `top_k`: 10
- `min_angle_step`: 0.314 (π/10)
- `n_samples`: 500

**参数变化（Group A 内部）：**
- `use_linkedlist`: False（3次运行）/ True（3次运行）
- `use_first_intersection_init`: False（4次运行）/ True（2次运行）
- `disable_min_shift`: False（4次运行）/ True（2次运行）

**观察：** 尽管存在这些变化，所有6次运行的百分位表几乎完全相同，证实了当 vector_transfer 开启时，`linkedlist`、`first_init` 和 `min_shift` 的影响可忽略。

---

### Group B — Vector-transfer 关闭
**关键设置：** `disable_vector_transfer: True`（vector_transfer **关闭**）

**运行：** 4个实验
- `20251208_174522`: linkedlist=False, first_init=False, min_shift=开启
- `20251208_174705`: linkedlist=False, first_init=True, min_shift=关闭
- `20251208_175635`: linkedlist=True, first_init=False, min_shift=开启
- `20251208_175920`: linkedlist=True, first_init=True, min_shift=关闭

**共同参数：**
- `use_official_style`: True
- `disable_vector_transfer`: **True**（vector_transfer 禁用）
- `use_l2_norm`: None/False（L1归一化）
- `top_k`: 10
- `min_angle_step`: 0.314 (π/10)
- `n_samples`: 500

**参数变化（Group B 内部）：**
- `use_linkedlist`: False（2次运行）/ True（2次运行）
- `use_first_intersection_init`: False（2次运行）/ True（2次运行）
- `disable_min_shift`: False（2次运行）/ True（2次运行）

**观察：** Group B 的运行与 Group A 存在一致的差异，组内因相同次要参数存在微小变化。

---

## 核心参数差异

| 参数 | Group A | Group B |
|------|---------|---------|
| **`disable_vector_transfer`** | **False**（开启） | **True**（关闭） |
| `use_l2_norm` | None/False（L1） | None/False（L1） |
| `use_linkedlist` | 混合 | 混合 |
| `use_first_intersection_init` | 混合 | 混合 |
| `disable_min_shift` | 混合 | 混合 |

**唯一系统性差异：** `disable_vector_transfer` 设置。

---

## 结果差异 — 百分位分析

### 1. PosRate_tail（尾部正例率）

**Group A（代表：173917/174132）：**
- q=1.0: 0.228（全局基线）
- q=0.1: 0.171–0.180
- q=0.01: 0.188–0.203
- q=0.001: 0.134–0.226
- q=0.0001: 0.019–0.154（非常稀疏）

**Group B（代表：174522, 175635, 175920）：**
- q=1.0: 0.228（相同全局基线）
- q=0.1: 0.200–0.214（略高）
- q=0.01: 0.207–0.252（**高于** Group A）
- q=0.001: 0.114–0.233（相似或更高）
- q=0.0001: 0.058–0.286（**远高于** Group A）

**关键发现：** Group B 在中低百分位（q≤0.01）保持更高的正例率，特别是在 q=0.0001 处，Group A 降至 0.019–0.154，而 Group B 达到 0.058–0.286。

---

### 2. F1-score

**Group A：**
- q=0.1: F1 ≈ 0.23–0.25
- q=0.01: F1 ≈ 0.15–0.51（因 top 方向而异）
- q=0.001: F1 ≈ 0.26–0.64（部分运行在此达到峰值）
- q=0.0001: **F1 = 0.000**（正例消失，F1 崩溃）

**Group B：**
- q=0.1: F1 ≈ 0.18–0.33（相似或略低）
- q=0.01: F1 ≈ 0.15–0.24（相似范围）
- q=0.001: F1 ≈ 0.20–0.30（保持）
- q=0.0001: **F1 = 0.25–0.46**（例如，174522 top-1: 0.381, top-3: 0.462；175920 top-2: 0.381）

**关键发现：** 在 q=0.0001 处，Group B **保持非零 F1**（0.25–0.46），而 Group A 崩溃至 0。这表明 Group B 的方向即使在极端尾部也能捕获更密集的正例区域。

---

### 3. Accuracy（准确率）

**Group A：**
- q=0.1: 0.82–0.83
- q=0.01: 0.80–0.84
- q=0.001: 0.78–0.89（当正例稀少时飙升）
- q=0.0001: 0.80–0.90（高，因为尾部主要是负例）

**Group B：**
- q=0.1: 0.80–0.81（略低）
- q=0.01: 0.76–0.79（较低，反映更高的正例率）
- q=0.001: 0.77–0.87（相似范围）
- q=0.0001: 0.74–0.89（极端处较低，因为保留了正例）

**关键发现：** Group B 在中百分位（q=0.01–0.001）显示略低的准确率，因为它捕获了更多正例（更高的 PosRate_tail），可能包含更多假正例。在 q=0.0001 处，Group B 的准确率（0.74–0.75）低于 Group A（0.80–0.90），因为 Group B 保留了正例，而 Group A 的尾部几乎全是负例。

---

## 可视化差异（3×3 九宫格）

**Group A（例如，`20251208_173917/official_utils_top_grid.png`）：**
- **Full 面板：** 统一的方向足迹；标准方向
- **Tail 面板：** 稀疏、分散的尾部区域；很少的集中簇
- **Density 面板：** 低正例密度；分散的点

**Group B（例如，`20251208_174522/official_utils_top_grid.png`，`20251208_175920/official_utils_top_grid.png`）：**
- **Full 面板：** 不同的方向方向（无 vector_transfer 映射回原空间）
- **Tail 面板：** 更集中、更紧密的尾部区域；更高的点密度
- **Density 面板：** 更密集的正例斑块；更明显的簇

---

## 总结：实际变化

### 参数影响层次

1. **`disable_vector_transfer`**（开启 vs 关闭）：
   - **主要影响：** 改变方向集；改变尾部组成
   - **结果：** Group B 在中/极端百分位找到具有更密集正例区域的方向
   - **证据：** Group B 在 q=0.0001 处保持非零 F1，而 Group A 为 F1=0

2. **`use_linkedlist`、`use_first_intersection_init`、`disable_min_shift`：**
   - **可忽略影响：** 仅在每组内产生微小数值漂移
   - **证据：** Group A 的运行几乎完全相同，尽管存在这些变化；Group B 的运行显示微小差异但整体模式相同

### 核心结果差异

| 指标 | Group A | Group B | 差异 |
|------|---------|---------|------|
| **PosRate_tail (q=0.0001)** | 0.019–0.154 | 0.058–0.286 | **Group B 高 2–15 倍** |
| **F1-score (q=0.0001)** | 0.000 | 0.25–0.46 | **Group B 保持 F1；Group A 崩溃** |
| **Accuracy (q=0.0001)** | 0.80–0.90 | 0.74–0.89 | Group B 略低（因保留正例） |
| **尾部集中度** | 稀疏、分散 | 集中、密集 | Group B 更结构化 |

### 解释

- **Group A（vector_transfer 开启）：** 方向被映射回原空间，产生由负例主导的稀疏尾部。在极端百分位，正例消失 → F1=0。
- **Group B（vector_transfer 关闭）：** 方向不映射回原空间；算法直接探索旋转空间。这找到不同的方向，捕获更密集的正例区域，即使在 q=0.0001 处也保持 F1。

**结论：** `disable_vector_transfer` 参数是**唯一系统性驱动**有意义结果差异的因素。所有其他参数（`linkedlist`、`first_init`、`min_shift`）仅导致可忽略的数值漂移。

