### 3.1 数据打乱与采样方式（与官方 notebook 的统一）

#### 官方 notebook 的做法

在 `Mining_U3M_Ray_Sweeping_2D_College_Admission.ipynb` 中，构造 2D 点集时使用了：

```python
final_df = data.sample(frac=1, random_state=0)
target = "admit"

x_train_new = np.array(final_df[["gre", "gpa"]])
y_train_new = np.array(final_df[target])

max_y = np.max(x_train_new[:, 1])
x_train_new_prime = np.array(
    list(map(lambda row: [max_y - row[1], row[0]], x_train_new))
)
```

要点：

- **不丢样本**：`frac=1` 表示使用整个数据集，只是随机打乱顺序。
- **固定随机种子**：`random_state=0` 保证每次运行打乱顺序一致。
- **构造两套点集**：
  - `x_train_new        = (gre, gpa)`
  - `x_train_new_prime  = [max_gpa - gpa, gre]`（旋转 / 反射后的点集，用于覆盖更多方向）。

#### 当前实现中的对齐方式

在 `experiment_ray_sweeping_2d_college_admission_official_style.py` 中，函数
`build_point_sets_from_data` 现在定义为：

```python
def build_point_sets_from_data(data, n_samples: int | None = None):
    """
    Construct the two 2D point sets used in the 2D experiment:

    - x_train_new:       (gre, gpa)
    - x_train_new_prime: [max_gpa - gpa, gre]

    Same as in the reimplementation script. By default, this function
    shuffles the entire dataset with a fixed random seed (frac=1,
    random_state=0), exactly as in the official notebook. If `n_samples`
    is provided and smaller than the dataset size, a random subset of
    that many rows is drawn with the same seed.
    """

    if not {"gre", "gpa"}.issubset(set(data.columns)):
        raise ValueError("Data must contain `gre` and `gpa` columns.")

    # Match official notebook behavior:
    # - If n_samples is None: shuffle all rows (frac=1, random_state=0)
    # - If n_samples is provided: draw a shuffled subset of that many rows.
    if n_samples is None:
        df_used = data.sample(frac=1.0, random_state=0)
    else:
        if len(data) < n_samples:
            raise ValueError(
                f"Requested {n_samples} samples, but dataset only has {len(data)} rows."
            )
        df_used = data.sample(n=n_samples, random_state=0)

    x_train_new = np.asarray(df_used[["gre", "gpa"]], dtype=float)
    max_gpa = float(np.max(x_train_new[:, 1]))
    x_train_new_prime = np.column_stack(
        (
            max_gpa - x_train_new[:, 1],
            x_train_new[:, 0],
        )
    )

    if "admit" in df_used.columns:
        targets = np.asarray(df_used["admit"], dtype=float)
    else:
        targets = None

    return x_train_new, x_train_new_prime, targets
```

一致性总结：

- **打乱方式**：默认使用 `sample(frac=1.0, random_state=0)`，与官方 notebook 的 `final_df` 完全一致。
- **采样可选**：如果显式提供 `n_samples`，则使用同样的随机种子 `random_state=0` 从全体数据中抽取固定数量的样本，保持实验可控。
- **点集构造**：`(gre, gpa)` 与 `[max_gpa - gpa, gre]` 的定义与官方 notebook 相同，确保传入官方 `MaxSkewCalculator` 的 2D 几何结构一致。


