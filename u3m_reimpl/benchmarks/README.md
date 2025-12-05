# Ray-Sweeping 算法性能基准测试

本目录包含 Ray-sweeping 算法中三种交点枚举方法的性能对比测试。

## 文件说明

- **`benchmark_comparison.py`**: 性能基准测试脚本
- **`benchmark_comparison.png`**: 性能对比图（对数刻度，0-2000 点）
- **`benchmark_comparison_more_points.png`**: 性能对比图（线性刻度，500-4000 点）

## 三种交点枚举方法

1. **Naive Enumeration (O(n²))**：暴力枚举所有点对
2. **Incremental Divide-and-Conquer**：增量分治方法
3. **Randomized Incremental (O(m+n log n))**：随机增量方法

## 运行基准测试

```bash
cd u3m_reimpl/benchmarks
python benchmark_comparison.py
```

## 测试结果

详细的分析报告请参考：
- `../analysis_docs/02_algorithm_comparison/INTERSECTION_ENUMERATION_BENCHMARK.md`

## 关键发现

- **理论 vs 实际**：理论上 Randomized 方法应该最优，但实际测试中 Naive 方法在大规模数据上表现最好
- **常数因子重要性**：实际应用中，常数因子往往比渐近复杂度更重要
- **方法选择**：推荐使用 Naive 方法（简单、高效、稳定）

