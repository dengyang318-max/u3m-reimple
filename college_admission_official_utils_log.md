# College Admission Official-Utils Experiment (20251208_205653)

Loading College Admission dataset...
No --excel-path provided. Downloading College Admission dataset via kagglehub...
Using dataset file: C:\Users\james\.cache\kagglehub\datasets\eswarchandt\admission\versions\1\Admission.xlsx
Loaded dataset with shape: (400, 7)
Primary point set shape: (400, 2)
Rotated point set shape: (400, 2)

Visualizing primary and rotated point sets...
![official_utils point sets](official_utils_point_sets.png)

Running ray-sweeping...
Total time (official-style): 1.7188 s

Top-k high-skew directions:
| Rank | Direction (x, y) | Skew |
|------|------------------|------|
| 1 | (0.573914, 0.426086) | 0.074051 |
| 2 | (0.235295, 0.764705) | 0.067487 |
| 3 | (0.413795, 0.586205) | 0.026085 |
| 4 | (0.749999, 0.250001) | 0.006247 |
| 5 | (0.992482, 0.007518) | 0.003123 |

Visualizing top-1 direction (skew=0.074051)...

Using reimplemented top direction for visualization.

Validation: skew along chosen direction f is -0.055245
![official_utils_top1 full](official_utils_top1_full.png)
Tail subset shape (q=0.1): (40, 7)

=== Global vs Tail statistics (College Admission) ===
| Metric                             | Global | Tail |
|------------------------------------|:------:|:----:|
| Positive rate (admit=1)           | 0.318 | 0.425 |
| Accuracy (Logistic Regression)    | 0.713 | 0.650 |
| F1 (Logistic Regression)          | 0.320 | 0.562 |

=== Multi-Percentile Evaluation Table ===
| Percentile | Accuracy | F1-score | F/M ratio |
|-----------:|---------:|---------:|----------:|
|       1.00 |    0.712 |    0.320 |    1.111 |
|       0.50 |    0.646 |    0.407 |    1.276 |
|       0.20 |    0.613 |    0.492 |    0.905 |
|       0.10 |    0.650 |    0.562 |    1.105 |
|       0.08 |    0.625 |    0.571 |    1.000 |
|       0.04 |    0.562 |    0.533 |    1.286 |
![official_utils_top1 tail](official_utils_top1_tail.png)
![official_utils_top1 density](official_utils_top1_density.png)

Visualizing top-2 direction (skew=0.067487)...

Using reimplemented top direction for visualization.

Validation: skew along chosen direction f is -0.019375
![official_utils_top2 full](official_utils_top2_full.png)
Tail subset shape (q=0.1): (40, 7)

=== Global vs Tail statistics (College Admission) ===
| Metric                             | Global | Tail |
|------------------------------------|:------:|:----:|
| Positive rate (admit=1)           | 0.318 | 0.400 |
| Accuracy (Logistic Regression)    | 0.713 | 0.575 |
| F1 (Logistic Regression)          | 0.320 | 0.452 |

=== Multi-Percentile Evaluation Table ===
| Percentile | Accuracy | F1-score | F/M ratio |
|-----------:|---------:|---------:|----------:|
|       1.00 |    0.712 |    0.320 |    1.111 |
|       0.50 |    0.640 |    0.400 |    1.198 |
|       0.20 |    0.650 |    0.481 |    0.905 |
|       0.10 |    0.575 |    0.452 |    0.739 |
|       0.08 |    0.625 |    0.538 |    0.684 |
|       0.04 |    0.583 |    0.545 |    1.400 |
![official_utils_top2 tail](official_utils_top2_tail.png)
![official_utils_top2 density](official_utils_top2_density.png)

Visualizing top-3 direction (skew=0.026085)...

Using reimplemented top direction for visualization.

Validation: skew along chosen direction f is 0.001739
![official_utils_top3 full](official_utils_top3_full.png)
Tail subset shape (q=0.1): (40, 7)

=== Global vs Tail statistics (College Admission) ===
| Metric                             | Global | Tail |
|------------------------------------|:------:|:----:|
| Positive rate (admit=1)           | 0.318 | 0.175 |
| Accuracy (Logistic Regression)    | 0.713 | 0.825 |
| F1 (Logistic Regression)          | 0.320 | 0.000 |

=== Multi-Percentile Evaluation Table ===
| Percentile | Accuracy | F1-score | F/M ratio |
|-----------:|---------:|---------:|----------:|
|       1.00 |    0.714 |    0.307 |    1.090 |
|       0.50 |    0.794 |    0.128 |    0.990 |
|       0.20 |    0.850 |    0.000 |    1.222 |
|       0.10 |    0.825 |    0.000 |    1.000 |
|       0.08 |    0.812 |    0.000 |    1.133 |
|       0.04 |    0.812 |    0.000 |    0.600 |
![official_utils_top3 tail](official_utils_top3_tail.png)
![official_utils_top3 density](official_utils_top3_density.png)
