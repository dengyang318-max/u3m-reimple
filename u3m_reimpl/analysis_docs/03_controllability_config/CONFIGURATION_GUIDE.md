# 实验配置指南：如何规范化 Ray Sweeping 实验

## 1. 快速开始

### 1.1 使用配置模板

1. **复制配置模板**：
   ```bash
   cp u3m_reimpl/experiments/experiment_config_template.yaml my_experiment_config.yaml
   ```

2. **修改配置**：
   ```yaml
   experiment_name: "my_experiment"
   preset: "updated"  # 选择预设配置
   ```

3. **运行实验**：
   ```python
   from u3m_reimpl.experiments.config_loader import load_config
   config = load_config("my_experiment_config.yaml")
   results = run_experiment_with_config(config)
   ```

### 1.2 预设配置说明

#### **original**：原始算法
- `min_shift`: false
- `vector_transfer`: None
- **特点**：不使用 min-shift 和 vector_transfer
- **适用场景**：对比实验，验证 min-shift 和 vector_transfer 的影响

#### **updated**：更新算法（推荐）
- `min_shift`: true
- `vector_transfer`: 90° rotation for rotated set
- **特点**：使用 min-shift 和 vector_transfer，与官方实现更接近
- **适用场景**：标准实验，发现更多方向的隐含信息

#### **official**：官方实现风格
- `min_shift`: true
- `vector_transfer`: 90° rotation for rotated set
- `filter_quadrant`: first
- **特点**：完全对齐官方实现
- **适用场景**：与官方结果对比

## 2. 关键配置项说明

### 2.1 数据预处理

#### **normalization**
- **选项**：`min_max` | `z_score` | `none`
- **影响**：影响点的分布，进而影响偏度计算
- **建议**：使用 `min_max`，与论文一致

#### **min_shift**
- **选项**：`enabled: true/false`
- **影响**：将数据平移到第一象限，改变极角排序起点
- **建议**：使用 `true`，与官方实现一致

#### **sampling**
- **影响**：不同的采样会导致不同的点集 → 不同的交点 → 不同的方向
- **建议**：固定 `random_state`，确保可复现

### 2.2 算法参数

#### **min_angle_step**
- **默认**：`π/10` (约 18°)
- **影响**：控制方向采样密度，更小的步长 → 更多候选方向
- **建议**：使用默认值，除非需要更精细的方向探索

#### **top_k**
- **默认**：`10`
- **影响**：返回多少个高偏度方向
- **建议**：根据需求调整，通常 3-10 个方向足够

### 2.3 方向合并策略

#### **vector_transfer_primary**
- **选项**：`identity` | `custom`
- **说明**：主点集的方向映射，通常是恒等映射
- **建议**：使用 `identity`

#### **vector_transfer_rotated**
- **选项**：`90_degree_rotation` | `custom`
- **说明**：旋转点集的方向映射，通常是 90° 旋转
- **建议**：使用 `90_degree_rotation`

## 3. 如何记录和报告配置

### 3.1 实验报告模板

```markdown
# 实验报告：{experiment_name}

## 1. 实验配置

### 1.1 数据预处理
- **归一化方式**: Min-max to [0, 1]
- **Min-shift**: Enabled (both coordinates)
- **采样方式**: Random sampling, n=500, random_state=1
- **数据打乱**: Enabled, random_state=0, frac=1.0

### 1.2 算法参数
- **min_angle_step**: π/10 (0.314 rad)
- **top_k**: 10
- **预设配置**: updated

### 1.3 方向合并策略
- **主点集映射**: Identity
- **旋转点集映射**: 90° rotation
- **合并策略**: Union and sort by skew

### 1.4 数值稳定性
- **零值容差**: 1e-10
- **排序稳定性**: Enabled
- **终止条件**: Exact equality

## 2. 实验结果

### 2.1 Top-3 方向
| Rank | Direction (x, y) | Skew |
|------|------------------|------|
| 1 | (-0.342964, 0.939348) | 0.192549 |
| 2 | (-0.033306, 0.999445) | 0.188779 |
| 3 | (-0.616631, 0.787252) | 0.152957 |

### 2.2 Tail 区域分析
- **Top-1 方向**:
  - Arrest rate: 0.190 (vs global 0.262, -27% relative change)
  - F1 score: 0.586 (vs global 0.512, +14% relative change)
  - **隐含信息**: Low-arrest-rate crime patterns

## 3. 配置影响分析

### 3.1 与原始配置的差异
- **Min-shift 的影响**: 改变了极角排序起点，发现了不同的高偏度方向
- **Vector transfer 的影响**: 通过旋转点集覆盖了更多方向空间

### 3.2 结果稳定性
- **方向一致性**: 多次运行（n=10）的方向相似度 > 0.8
- **Tail 重叠率**: 多次运行的 tail 区域重叠率 > 0.7
```

### 3.2 配置对比表

```markdown
## 配置对比

| 配置项 | Original | Updated | Official |
|--------|----------|---------|----------|
| Min-shift | ❌ | ✅ | ✅ |
| Vector transfer | ❌ | ✅ | ✅ |
| Filter quadrant | All | All | First |
| 发现的方向数 | 较少 | 较多 | 较多 |
| 与官方一致性 | 低 | 中 | 高 |
```

## 4. 常见问题

### Q1: 为什么不同配置发现不同的方向？

**A**: 这是算法的**探索性特性**：
- 不同的配置会探索不同的方向空间
- Min-shift 改变极角排序起点
- Vector transfer 覆盖不同的方向范围
- 这是**算法的特性**，不是错误

### Q2: 如何选择配置？

**A**: 根据实验目的：
- **对比实验**：使用 `original` 配置
- **标准实验**：使用 `updated` 配置（推荐）
- **官方对比**：使用 `official` 配置

### Q3: 如何确保可复现性？

**A**: 遵循以下原则：
1. **固定随机种子**：`random_state=1` for sampling, `random_state=0` for shuffle
2. **明确记录配置**：使用配置文件，不要硬编码
3. **版本控制**：记录算法版本、数据版本、配置版本
4. **稳定性分析**：多次运行，分析结果稳定性

### Q4: 如何解释不同配置的差异？

**A**: 使用以下框架：
1. **算法层面**：不同配置探索不同的方向空间
2. **数据层面**：不同配置从不同角度挖掘数据
3. **实现层面**：数值精度、排序稳定性等细节的影响

## 5. 最佳实践

### 5.1 实验设计

1. **系统化对比**：
   - 先运行 `original` 配置作为基线
   - 再运行 `updated` 配置看改进
   - 最后运行 `official` 配置与官方对比

2. **稳定性验证**：
   - 多次运行（n=10），分析方向一致性
   - 分析 tail 区域重叠率
   - 报告稳定性指标

3. **结果解释**：
   - 明确说明使用的配置
   - 解释不同配置的影响
   - 分析发现的隐含信息

### 5.2 代码规范

1. **配置外部化**：
   ```python
   # ✅ 好的做法
   config = load_config("experiment_config.yaml")
   results = run_experiment(**config)
   
   # ❌ 不好的做法
   results = run_experiment(top_k=10, min_angle_step=np.pi/10)
   ```

2. **日志记录**：
   ```python
   # ✅ 好的做法
   logger.info(f"Using config: {config}")
   logger.info(f"Found {len(results)} directions")
   
   # ❌ 不好的做法
   results = run_experiment()
   ```

3. **结果保存**：
   ```python
   # ✅ 好的做法
   save_results(results, config, output_path)
   
   # ❌ 不好的做法
   print(results)
   ```

## 6. 总结

### 关键要点

1. **算法不是"不可控"的**，而是需要**明确规范**
2. **不同配置发现不同的隐含信息**是算法的**特性**，不是错误
3. **关键是要明确记录**所有配置，并**解释**不同设置的影响
4. **使用配置文件**和**预设配置**可以提高可复现性
5. **进行稳定性分析**可以验证结果的可靠性

### 下一步

1. 使用配置模板创建你的实验配置
2. 运行实验并记录所有配置
3. 进行稳定性分析
4. 撰写实验报告，明确说明配置和结果

