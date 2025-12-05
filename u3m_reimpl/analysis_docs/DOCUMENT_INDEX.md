# Analysis Documents Index - 分析文档索引

本文档按照主题对 `analysis_docs/` 目录中的所有文档进行分类索引，方便快速查找相关内容。

---

## 📚 文档分类索引

### 1. 项目概述和介绍类 (Project Overview and Introduction)

**文件夹**：`01_project_overview/`

**目的**：介绍项目设计、实现思路和使用方法

| 文档 | 语言 | 主要内容 |
|------|------|----------|
| `01_project_overview/README_reimpl.md` | English | 项目设计和实现说明，包结构和使用方法，几何和统计工具说明 |
| `01_project_overview/README_reimpl_zh.md` | 中文 | 项目设计和实现说明（中文版） |
| `01_project_overview/README_experiments_2d.md` | English | 2D 实验的详细说明，如何运行 College Admission 和 Chicago Crimes 实验 |

**使用场景**：
- 初次接触项目，了解整体设计
- 学习如何使用算法和工具
- 运行实验前的准备

---

### 2. 算法实现对比类 (Algorithm Implementation Comparison)

**文件夹**：`02_algorithm_comparison/`

**目的**：对比复现实现与官方实现的差异，分析结果不同的原因

| 文档 | 语言 | 主要内容 |
|------|------|----------|
| `02_algorithm_comparison/COMPARISON_WITH_OFFICIAL.md` | 中文 | 复现代码与官方代码的详细对比，12 个关键差异点的详细说明 |
| `02_algorithm_comparison/WHY_RESULTS_DIFFERENT.md` | 中文 | 为什么官方风格实现的结果和官方结果还是不一样？数据结构差异、遍历逻辑差异分析 |
| `02_algorithm_comparison/IMPLEMENTATION_REVIEW.md` | 中文 | 复现实现检查报告，代码质量评估，发现的问题和修复建议，旋转操作的完整流程说明 |
| `02_algorithm_comparison/COMPARISON_SKEW_CALCULATION.md` | 中文 | 偏度计算方法的对比，方向向量归一化方法（L1 vs L2），标准差计算差异 |
| `02_algorithm_comparison/ANALYSIS_FIXED_VS_DYNAMIC_MEDIAN.md` | 中文 | 固定中位数 vs 动态中位数的对比分析，为什么结果差异如此巨大？ |

**使用场景**：
- 理解复现实现与官方实现的差异
- 调试算法，找出结果不一致的原因
- 学习算法的不同实现方式

**相关文档关系**：
- `COMPARISON_WITH_OFFICIAL.md` → 全面的差异对比
- `WHY_RESULTS_DIFFERENT.md` → 深入分析为什么还有差异
- `IMPLEMENTATION_REVIEW.md` → 代码质量检查
- `COMPARISON_SKEW_CALCULATION.md` → 偏度计算细节
- `ANALYSIS_FIXED_VS_DYNAMIC_MEDIAN.md` → 中位数策略影响

---

### 3. 算法可控性和配置类 (Algorithm Controllability and Configuration)

**文件夹**：`03_controllability_config/`

**目的**：分析算法的可控性、可复现性，提供配置管理方案

| 文档 | 语言 | 主要内容 |
|------|------|----------|
| `03_controllability_config/ALGORITHM_CONTROLLABILITY_ANALYSIS.md` | 中文 | 算法可控性与可复现性分析，细节变化分类，规范化建议，如何说明不同设置能发掘哪个方向的少数群体 |
| `03_controllability_config/CONFIGURATION_GUIDE.md` | 中文 | 实验配置指南，如何使用配置模板，关键配置项说明，实验报告模板和最佳实践 |
| `03_controllability_config/college_admission_data_pipeline_alignment.md` | 中文 | College Admission 数据管道对齐，数据打乱与采样方式的统一，与官方 notebook 的对齐方式 |

**使用场景**：
- 理解算法对实现细节的敏感性
- 设计可复现的实验
- 记录和管理实验配置
- 解释不同配置的差异

**相关文档关系**：
- `ALGORITHM_CONTROLLABILITY_ANALYSIS.md` → 理论分析和框架
- `CONFIGURATION_GUIDE.md` → 实践指南和模板
- `college_admission_data_pipeline_alignment.md` → 具体数据对齐案例

---

### 4. 实验结果分析类 (Experimental Results Analysis)

**文件夹**：`04_results_analysis/`

**目的**：分析实验结果，识别 tail 区域的隐含信息

| 文档 | 语言 | 主要内容 |
|------|------|----------|
| `04_results_analysis/EXPERIMENT_RESULTS_ANALYSIS.md` | 中文 | 早期实验结果异常分析，Chicago Crimes 和 College Admission 的异常模式识别，不同方法的结果差异说明 |
| `04_results_analysis/EXPERIMENT_RESULTS_ANALYSIS_NEW.md` | 中文 | 新实验结果异常分析（使用完整数据集和多百分位评估），Original vs Updated 版本对比，Official Utils 方法分析 |
| `04_results_analysis/EXPERIMENT_7_1_SUMMARY.md` | 中文 | 论文 7.1 节 2D 实验检查与补充总结，已实现的实验内容，实验流程（完整版） |

**使用场景**：
- 分析实验结果，理解 tail 区域的隐含信息
- 对比不同方法的效果
- 验证实验是否符合论文要求

**相关文档关系**：
- `EXPERIMENT_RESULTS_ANALYSIS.md` → 早期实验结果
- `EXPERIMENT_RESULTS_ANALYSIS_NEW.md` → 最新实验结果（推荐阅读）
- `EXPERIMENT_7_1_SUMMARY.md` → 实验完整性检查

---

### 5. 实验方法和概念解释类 (Experimental Methods and Concept Explanation)

**文件夹**：`05_methods_concepts/`

**目的**：解释实验方法、概念和检查清单

| 文档 | 语言 | 主要内容 |
|------|------|----------|
| `05_methods_concepts/PERCENTILE_EXPLANATION.md` | 中文 | Percentile（百分位）在项目中的含义和处理方式，为什么使用不同的 Percentile？代码示例和重要注意事项 |
| `05_methods_concepts/INTERPRETATION_OF_MINORITIES_2D.md` | English | 如何解释 Ray-sweeping 识别的少数群体，算法的工作原理，为什么"少数群体"不是固定的个体列表 |
| `05_methods_concepts/EXPERIMENT_CHECKLIST_2D.md` | 中文 | 2D 实验检查清单（基于论文 7.1 节），已实现的实验内容，已补充的实验内容 |
| `05_methods_concepts/MISSING_VISUALIZATIONS_7_1.md` | 中文 | 论文 7.1 节缺失的可视化和分析，需要补充的内容（投影密度图、多百分位评估表等），优先级说明 |

**使用场景**：
- 理解实验中的概念（如 Percentile）
- 解释算法发现的少数群体
- 检查实验完整性
- 补充缺失的可视化

**相关文档关系**：
- `PERCENTILE_EXPLANATION.md` → 概念解释
- `INTERPRETATION_OF_MINORITIES_2D.md` → 结果解释
- `EXPERIMENT_CHECKLIST_2D.md` → 实验检查
- `MISSING_VISUALIZATIONS_7_1.md` → 可视化补充

---

## 🔍 快速查找指南

### 按问题类型查找

**Q: 我想了解项目的整体设计？**
→ 查看 **1. 项目概述和介绍类**

**Q: 为什么我的结果和官方结果不一样？**
→ 查看 **2. 算法实现对比类** (`02_algorithm_comparison/`)，特别是：
- `02_algorithm_comparison/COMPARISON_WITH_OFFICIAL.md`
- `02_algorithm_comparison/WHY_RESULTS_DIFFERENT.md`

**Q: 如何确保实验可复现？**
→ 查看 **3. 算法可控性和配置类** (`03_controllability_config/`)，特别是：
- `03_controllability_config/ALGORITHM_CONTROLLABILITY_ANALYSIS.md`
- `03_controllability_config/CONFIGURATION_GUIDE.md`

**Q: 如何分析实验结果？**
→ 查看 **4. 实验结果分析类** (`04_results_analysis/`)，特别是：
- `04_results_analysis/EXPERIMENT_RESULTS_ANALYSIS_NEW.md`

**Q: Percentile 是什么意思？**
→ 查看 **5. 实验方法和概念解释类** (`05_methods_concepts/`)，特别是：
- `05_methods_concepts/PERCENTILE_EXPLANATION.md`

### 按任务类型查找

**任务：运行实验**
1. `01_project_overview/README_experiments_2d.md` - 了解如何运行
2. `05_methods_concepts/EXPERIMENT_CHECKLIST_2D.md` - 检查实验完整性

**任务：理解算法差异**
1. `02_algorithm_comparison/COMPARISON_WITH_OFFICIAL.md` - 全面对比
2. `02_algorithm_comparison/WHY_RESULTS_DIFFERENT.md` - 深入分析
3. `02_algorithm_comparison/IMPLEMENTATION_REVIEW.md` - 代码检查

**任务：分析结果**
1. `04_results_analysis/EXPERIMENT_RESULTS_ANALYSIS_NEW.md` - 最新结果分析
2. `05_methods_concepts/PERCENTILE_EXPLANATION.md` - 理解 Percentile
3. `05_methods_concepts/INTERPRETATION_OF_MINORITIES_2D.md` - 解释少数群体

**任务：配置管理**
1. `03_controllability_config/CONFIGURATION_GUIDE.md` - 配置指南
2. `03_controllability_config/ALGORITHM_CONTROLLABILITY_ANALYSIS.md` - 可控性分析

---

## 📊 文档统计

- **总文档数**：17 个
- **中文文档**：14 个
- **英文文档**：3 个
- **分类数**：5 个

---

## 🔗 相关资源

- **项目主 README**：`../../README.md`
- **项目说明文档（中文）**：`../../PROJECT_DOCUMENTATION_CN.md`
- **项目说明文档（英文）**：`../../PROJECT_DOCUMENTATION_EN.md`
- **GitHub 上传指南**：`../../GITHUB_UPLOAD_GUIDE.md`

---

## 📝 更新记录

- **2024-12-04**：创建文档索引，完成文档分类
- **2024-12-05**：文档按分类移动到对应文件夹，更新路径引用

