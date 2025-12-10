# College Admission Official-Utils Experiment (20251208_205727)

Loading College Admission dataset...
No --excel-path provided. Downloading College Admission dataset via kagglehub...
Using dataset file: C:\Users\james\.cache\kagglehub\datasets\eswarchandt\admission\versions\1\Admission.xlsx
Loaded dataset with shape: (400, 7)
Primary point set shape: (400, 2)
Rotated point set shape: (400, 2)

Visualizing primary and rotated point sets...
![official_utils point sets](official_utils_point_sets.png)

Running ray-sweeping...
Total time (official-style): 1.6623 s

Top-k high-skew directions:
| Rank | Direction (x, y) | Skew |
|------|------------------|------|
| 1 | (-0.596098, 0.802911) | 0.074051 |
| 2 | (-0.955779, 0.294087) | 0.067487 |
| 3 | (-0.816965, 0.576687) | 0.026085 |
| 4 | (-0.316229, 0.948683) | 0.006247 |
| 5 | (-0.000000, 1.000000) | 0.000263 |

Visualizing top-1 direction (skew=0.074051)...

Using reimplemented top direction for visualization.

Validation: skew along chosen direction f is -0.055519
![official_utils_top1 full](official_utils_top1_full.png)
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
|       0.50 |    0.690 |    0.367 |    1.128 |
|       0.20 |    0.675 |    0.435 |    0.818 |
|       0.10 |    0.641 |    0.364 |    2.000 |
|       0.08 |    0.677 |    0.444 |    1.818 |
|       0.04 |    0.800 |    0.400 |    1.500 |
![official_utils_top1 tail](official_utils_top1_tail.png)
![official_utils_top1 density](official_utils_top1_density.png)

Visualizing top-2 direction (skew=0.067487)...

Using reimplemented top direction for visualization.

Validation: skew along chosen direction f is -0.004977
![official_utils_top2 full](official_utils_top2_full.png)
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
|       0.50 |    0.765 |    0.277 |    1.222 |
|       0.20 |    0.775 |    0.308 |    1.162 |
|       0.10 |    0.775 |    0.000 |    1.353 |
|       0.08 |    0.844 |    0.000 |    1.133 |
|       0.04 |    0.875 |    0.000 |    1.286 |
![official_utils_top2 tail](official_utils_top2_tail.png)
![official_utils_top2 density](official_utils_top2_density.png)

Visualizing top-3 direction (skew=0.026085)...

Using reimplemented top direction for visualization.

Validation: skew along chosen direction f is 0.009519
![official_utils_top3 full](official_utils_top3_full.png)
Tail subset shape (q=0.1): (40, 7)

=== Global vs Tail statistics (College Admission) ===
| Metric                             | Global | Tail |
|------------------------------------|:------:|:----:|
| Positive rate (admit=1)           | 0.318 | 0.400 |
| Accuracy (Logistic Regression)    | 0.713 | 0.675 |
| F1 (Logistic Regression)          | 0.320 | 0.381 |

=== Multi-Percentile Evaluation Table ===
| Percentile | Accuracy | F1-score | F/M ratio |
|-----------:|---------:|---------:|----------:|
|       1.00 |    0.712 |    0.320 |    1.100 |
|       0.50 |    0.690 |    0.295 |    1.174 |
|       0.20 |    0.662 |    0.308 |    1.000 |
|       0.10 |    0.675 |    0.381 |    1.000 |
|       0.08 |    0.656 |    0.353 |    1.133 |
|       0.04 |    0.625 |    0.400 |    0.600 |
![official_utils_top3 tail](official_utils_top3_tail.png)
![official_utils_top3 density](official_utils_top3_density.png)
