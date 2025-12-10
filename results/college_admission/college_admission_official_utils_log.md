# College Admission Official-Utils Experiment (20251208_205408)

Loading College Admission dataset...
No --excel-path provided. Downloading College Admission dataset via kagglehub...
Using dataset file: C:\Users\james\.cache\kagglehub\datasets\eswarchandt\admission\versions\1\Admission.xlsx
Loaded dataset with shape: (400, 7)
Primary point set shape: (400, 2)
Rotated point set shape: (400, 2)

Visualizing primary and rotated point sets...
![official_utils point sets](official_utils_point_sets.png)

Running ray-sweeping...
Total time (official-style): 2.0789 s

Top-k high-skew directions:
| Rank | Direction (x, y) | Skew |
|------|------------------|------|
| 1 | (1.409087, -0.409087) | 0.051051 |
| 2 | (-3.384646, 4.384646) | 0.049719 |
| 3 | (3.117606, -2.117606) | 0.041128 |
| 4 | (-0.548379, 1.548379) | 0.022656 |
| 5 | (-0.018869, 1.018869) | 0.007282 |

Visualizing top-1 direction (skew=0.051051)...

Using reimplemented top direction for visualization.

Validation: skew along chosen direction f is 0.008759
![official_utils_top1 full](official_utils_top1_full.png)
Tail subset shape (q=0.1): (40, 7)

=== Global vs Tail statistics (College Admission) ===
| Metric                             | Global | Tail |
|------------------------------------|:------:|:----:|
| Positive rate (admit=1)           | 0.318 | 0.225 |
| Accuracy (Logistic Regression)    | 0.713 | 0.775 |
| F1 (Logistic Regression)          | 0.320 | 0.000 |

=== Multi-Percentile Evaluation Table ===
| Percentile | Accuracy | F1-score | F/M ratio |
|-----------:|---------:|---------:|----------:|
|       1.00 |    0.712 |    0.320 |    1.100 |
|       0.50 |    0.765 |    0.254 |    1.247 |
|       0.20 |    0.775 |    0.308 |    1.162 |
|       0.10 |    0.775 |    0.000 |    1.222 |
|       0.08 |    0.844 |    0.000 |    1.133 |
|       0.04 |    0.875 |    0.000 |    1.000 |
![official_utils_top1 tail](official_utils_top1_tail.png)
![official_utils_top1 density](official_utils_top1_density.png)

Visualizing top-2 direction (skew=0.049719)...

Using reimplemented top direction for visualization.

Validation: skew along chosen direction f is -0.049721
![official_utils_top2 full](official_utils_top2_full.png)
Tail subset shape (q=0.1): (39, 7)

=== Global vs Tail statistics (College Admission) ===
| Metric                             | Global | Tail |
|------------------------------------|:------:|:----:|
| Positive rate (admit=1)           | 0.318 | 0.436 |
| Accuracy (Logistic Regression)    | 0.713 | 0.641 |
| F1 (Logistic Regression)          | 0.320 | 0.364 |

=== Multi-Percentile Evaluation Table ===
| Percentile | Accuracy | F1-score | F/M ratio |
|-----------:|---------:|---------:|----------:|
|       1.00 |    0.714 |    0.321 |    1.111 |
|       0.50 |    0.700 |    0.375 |    1.128 |
|       0.20 |    0.675 |    0.409 |    0.818 |
|       0.10 |    0.641 |    0.364 |    2.000 |
|       0.08 |    0.677 |    0.444 |    1.818 |
|       0.04 |    0.800 |    0.400 |    1.500 |
![official_utils_top2 tail](official_utils_top2_tail.png)
![official_utils_top2 density](official_utils_top2_density.png)

Visualizing top-3 direction (skew=0.041128)...

Using reimplemented top direction for visualization.

Validation: skew along chosen direction f is 0.011433
![official_utils_top3 full](official_utils_top3_full.png)
Tail subset shape (q=0.1): (40, 7)

=== Global vs Tail statistics (College Admission) ===
| Metric                             | Global | Tail |
|------------------------------------|:------:|:----:|
| Positive rate (admit=1)           | 0.318 | 0.300 |
| Accuracy (Logistic Regression)    | 0.713 | 0.750 |
| F1 (Logistic Regression)          | 0.320 | 0.286 |

=== Multi-Percentile Evaluation Table ===
| Percentile | Accuracy | F1-score | F/M ratio |
|-----------:|---------:|---------:|----------:|
|       1.00 |    0.712 |    0.320 |    1.100 |
|       0.50 |    0.740 |    0.350 |    1.041 |
|       0.20 |    0.738 |    0.364 |    1.286 |
|       0.10 |    0.750 |    0.286 |    1.222 |
|       0.08 |    0.812 |    0.250 |    1.133 |
|       0.04 |    0.875 |    0.500 |    1.286 |
![official_utils_top3 tail](official_utils_top3_tail.png)
![official_utils_top3 density](official_utils_top3_density.png)
