# College Admission Official-Utils 实验结果分组总结

本文基于 `ALL_RUNS_GRIDS.md` 的百分位结果，对各次运行进行分组，并指出哪些参数会显著影响结果。

## 结果分组

- **A 组 — 基线（vector_transfer 开、L1）**  
  运行：`20251208_205408`、`20251208_205427`、`20251208_205445`（关闭 min_shift）、`20251208_205538`、`20251208_205557`、`20251208_205619`  
  表现：百分位表基本一致；`use_linkedlist` 仅带来极小数值漂移。  
  参数：`disable_vector_transfer=False`，`use_l2_norm=False`；`use_first_intersection_init`、`use_linkedlist`、`disable_min_shift` 单独切换不引起可见变化。

- **B 组 — 关闭 vector_transfer（变化最大）**  
  运行：`20251208_205502`、`20251208_205521`、`20251208_205636`、`20251208_205653`、`20251208_205816`  
  表现：百分位表出现明显差异（accuracy/F1/F/M 比例等均有变化），这是主要的结果变化来源。  
  参数：`disable_vector_transfer=True`（或使用无 vector-transfer 映射的官方移植）。

- **C 组 — L2 归一化（vector_transfer 开）**  
  运行：`20251208_205710`、`20251208_205727`、`20251208_205744`、`20251208_205759`  
  表现：相对基线有小幅、可预期的漂移，影响远小于关闭 vector_transfer。  
  参数：`use_l2_norm=True`，`disable_vector_transfer=False`。

## 参数影响（真正起作用的）

- **vector_transfer**（`disable_vector_transfer`）：关闭后带来最明显、最可重复的变化，改变了找到的方向/尾部。
- **归一化**（`use_l2_norm`）：次要敏感性，影响幅度小。
- **min-shift**（`disable_min_shift`）：单独切换几乎无影响；仅在已关闭 vector_transfer 时有轻微交互。
- **数据结构**（`use_linkedlist`）：影响可忽略，仅有微小数值漂移。
- **初始化**（`use_first_intersection_init`）：单独切换影响可忽略，无质变。

## 建议

- 若追求稳定的“官方风格”结果：保持 `disable_vector_transfer=False` 且 `use_l2_norm=False`；其余开关可自由切换影响不大。
- 若要研究主要差异：对比基线 A 组与关闭 vector_transfer 的 B 组。
- 若要考察次要效应：对比基线 A 组与 L2 的 C 组。

