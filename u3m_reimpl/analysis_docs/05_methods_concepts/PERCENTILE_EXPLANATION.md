# Percentile（百分位）在项目中的含义和处理方式

## 1. Percentile 的含义

在项目中，**percentile（百分位）** 用于定义 **tail（尾部）区域的大小**，即从数据集中选择多少比例的极端数据点作为 tail 子集。

### 具体含义

- **Percentile = 1.00**：选择 100% 的数据（即整个数据集）
- **Percentile = 0.50**：选择 50% 的数据（即一半的数据）
- **Percentile = 0.20**：选择 20% 的数据
- **Percentile = 0.10**：选择 10% 的数据
- **Percentile = 0.08**：选择 8% 的数据
- **Percentile = 0.04**：选择 4% 的数据

**注意**：这里的 percentile 指的是**极端尾部区域的比例**，不是统计学中常见的分位数概念。

## 2. 处理流程

### 2.1 基本步骤

对于每个高偏度方向（high-skew direction）`f`，处理流程如下：

1. **投影所有数据点到方向 f**：
   ```python
   proj = coords @ f  # coords 是 (gre, gpa) 或 (Lon, Lat) 等特征
   ```

2. **计算偏度（skew）**：
   ```python
   sk = (np.mean(proj) - np.median(proj)) / np.std(proj)
   ```

3. **根据偏度方向确定 tail 区域**：
   - 如果 `sk < 0`（负偏度）：tail 在投影值的**左侧**（较小值）
   - 如果 `sk > 0`（正偏度）：tail 在投影值的**右侧**（较大值）

4. **根据 percentile 计算阈值**：
   ```python
   q = 0.10  # 例如 10% percentile
   if -sk < 0:  # 负偏度，tail 在左侧
       thresh = np.quantile(proj, q)  # 取第 q 分位数（左侧）
   else:  # 正偏度，tail 在右侧
       thresh = np.quantile(proj, 1.0 - q)  # 取第 (1-q) 分位数（右侧）
   ```

5. **选择 tail 子集**：
   ```python
   if -sk < 0:
       mask = proj < thresh  # 选择投影值小于阈值的点
   else:
       mask = proj > thresh  # 选择投影值大于阈值的点
   tail = df.loc[mask]
   ```

### 2.2 多 Percentile 评估

在 College Admission 实验中，会对**多个 percentile 值**进行评估，以观察不同 tail 大小下的模型性能：

```python
percentiles = [1.00, 0.50, 0.20, 0.10, 0.08, 0.04]

for q in percentiles:
    # 1. 计算该 percentile 下的阈值
    thresh_q = np.quantile(proj, q if -sk < 0 else (1.0 - q))
    
    # 2. 选择 tail 子集
    if -sk < 0:
        mask_q = proj < thresh_q
    else:
        mask_q = proj > thresh_q
    tail_q = df.loc[mask_q]
    
    # 3. 在 tail 子集上评估模型性能
    y_tail_q = tail_q["admit"].to_numpy(dtype=int)
    X_tail_q = tail_q[features].to_numpy(dtype=float)
    
    # 使用全局训练的模型在 tail 上预测
    y_pred_tail_q = clf.predict(X_tail_q)
    
    # 4. 计算指标
    acc_q = accuracy_score(y_tail_q, y_pred_tail_q)
    f1_q = f1_score(y_tail_q, y_pred_tail_q)
    
    # 5. 计算 tail 中的 Female/Male 比例（College Admission 特有）
    if "Gender_Male" in tail_q.columns:
        gender_counts_tail = tail_q["Gender_Male"].value_counts()
        tail_female_ratio = gender_counts_tail[0] / gender_counts_tail[1]
```

## 3. 为什么使用不同的 Percentile？

### 3.1 研究目的

使用多个 percentile 的目的是：

1. **观察模型在不同 tail 大小下的性能变化**：
   - 当 tail 很大（percentile = 1.00）时，相当于在整个数据集上评估
   - 当 tail 很小（percentile = 0.04）时，只评估最极端的 4% 数据
   - 可以观察模型在极端情况下的表现

2. **发现潜在的公平性问题**：
   - 在 College Admission 数据集中，观察不同 tail 大小下的 Female/Male 比例
   - 如果 tail 越小，Female/Male 比例变化越大，可能说明存在性别相关的偏差

3. **验证高偏度方向的有效性**：
   - 如果某个方向确实捕获了重要的偏差，那么 tail 越小，模型性能或人口统计比例的变化应该越明显

### 3.2 实验设计

在官方 notebook 和我们的实现中：

- **College Admission**：使用 `[1.00, 0.50, 0.20, 0.10, 0.08, 0.04]`
  - 对应论文中的 Table 3
  - 评估指标：Accuracy, F1-score, Female/Male ratio

- **Chicago Crimes**：通常使用 `[1, 0.1, 0.01, 0.001, 0.0001]`
  - 对应论文中的 Table 1 & 2
  - 评估指标：Accuracy, F1-score

## 4. 代码示例

### 4.1 单 Percentile 评估（用于可视化）

```python
# 在 validate_and_visualize_tail 函数中
q = 0.1  # 10% tail，用于可视化
thresh = np.quantile(proj, q if -sk < 0 else (1.0 - q))
if -sk < 0:
    mask = proj < thresh
else:
    mask = proj > thresh
tail = df.loc[mask]
```

### 4.2 多 Percentile 评估（用于表格）

```python
# 在 validate_and_visualize_tail 函数中
percentiles = [1.00, 0.50, 0.20, 0.10, 0.08, 0.04]

print("| Percentile | Accuracy | F1-score | F/M ratio |")
print("|-----------:|---------:|---------:|----------:|")

for q in percentiles:
    thresh_q = np.quantile(proj, q if -sk < 0 else (1.0 - q))
    if -sk < 0:
        mask_q = proj < thresh_q
    else:
        mask_q = proj > thresh_q
    tail_q = df.loc[mask_q]
    
    # 评估模型性能
    y_tail_q = tail_q["admit"].to_numpy(dtype=int)
    X_tail_q = tail_q[features].to_numpy(dtype=float)
    y_pred_tail_q = clf.predict(X_tail_q)
    
    acc_q = accuracy_score(y_tail_q, y_pred_tail_q)
    f1_q = f1_score(y_tail_q, y_pred_tail_q)
    
    # 计算 Female/Male 比例
    if "Gender_Male" in tail_q.columns:
        gender_counts_tail = tail_q["Gender_Male"].value_counts()
        tail_female_ratio = gender_counts_tail[0] / gender_counts_tail[1]
    
    print(f"| {q:>10.2f} | {acc_q:>8.3f} | {f1_q:>8.3f} | {tail_female_ratio:>8.3f} |")
```

## 5. 重要注意事项

### 5.1 模型训练方式

**重要**：在多 percentile 评估中，模型是**在全局数据集上训练的**，然后在不同的 tail 子集上评估：

```python
# 1. 在全局数据集上训练模型（只训练一次）
clf = LogisticRegression(random_state=1, max_iter=1000)
clf.fit(X_all, y_all)  # X_all 是整个数据集

# 2. 对每个 percentile，在对应的 tail 子集上评估
for q in percentiles:
    tail_q = ...  # 根据 percentile 选择 tail 子集
    y_pred_tail_q = clf.predict(X_tail_q)  # 使用全局训练的模型预测
    acc_q = accuracy_score(y_tail_q, y_pred_tail_q)
```

**不是**在每个 tail 子集上重新训练模型。

### 5.2 偏度方向的影响

偏度方向决定了 tail 区域的选择：

- **负偏度（sk < 0）**：tail 在投影值的左侧（较小值）
  - 使用 `np.quantile(proj, q)` 选择最小的 q 比例数据

- **正偏度（sk > 0）**：tail 在投影值的右侧（较大值）
  - 使用 `np.quantile(proj, 1.0 - q)` 选择最大的 q 比例数据

### 5.3 可视化中的 Percentile

在可视化中，通常使用固定的 percentile（如 0.1 或 0.01）来展示 tail 区域，而多 percentile 评估主要用于生成表格，展示不同 tail 大小下的性能变化。

## 6. 总结

- **Percentile** 定义了 tail 区域的大小（数据比例）
- **处理流程**：投影 → 计算阈值 → 选择 tail 子集 → 评估模型性能
- **多 Percentile 评估**：用于观察模型在不同 tail 大小下的性能变化
- **模型训练**：在全局数据集上训练一次，然后在不同 tail 子集上评估
- **应用场景**：发现模型在极端数据上的表现，检测潜在的公平性问题

