# Chicago Crimes Official-Utils Experiment (20251208_173917)

Loading Chicago Crimes dataset (official-port experiment)...
No --csv-path provided. Downloading Chicago Crimes dataset via kagglehub...
Using dataset file: C:\Users\james\.cache\kagglehub\datasets\currie32\crimes-in-chicago\versions\1\Chicago_Crimes_2012_to_2017.csv
Loaded dataset with shape: (510372, 27)
Primary point set shape: (500, 2)
Rotated point set shape: (500, 2)

Visualizing primary and rotated point sets (official utils)...
![official_utils point sets](official_utils_point_sets.png)

Running ray-sweeping (official vs official-style)...
Total time (official-style): 2.7230 s

Top-k high-skew directions (official port on rotated set):
| Rank | Direction (x, y) | Skew |
|------|------------------|------|
| 1 | (-0.006510, 1.006510) | 0.267657 |
| 2 | (-0.504300, 1.504300) | 0.216177 |
| 3 | (-3.059919, 4.059919) | 0.112961 |
| 4 | (1.397601, -0.397601) | 0.029934 |
| 5 | (3.068332, -2.068332) | 0.023165 |

[Official port] Visualizing tail and density along top-1 direction (skew=0.267657)...

Using reimplemented top direction for visualization.

Validation: skew along chosen direction f is -0.267657
![official_utils_top1 full](official_utils_top1_full.png)
Tail subset shape (q=0.01): (5104, 27)
![official_utils_top1 tail](official_utils_top1_tail.png)
![official_utils_top1 density](official_utils_top1_density.png)

=== Global vs Tail statistics (Chicago Crimes) ===
| Metric                             | Global | Tail |
|------------------------------------|:------:|:----:|
| Positive rate (Arrest=1)          | 0.228 | 0.203 |
| Accuracy (Logistic Regression)    | 0.799 | 0.801 |
| F1 (Logistic Regression)          | 0.330 | 0.146 |

=== Multi-Percentile Evaluation Table (Chicago Crimes, full dataset) ===
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.180 |    0.824 |    0.230 |
|    0.01000 |       0.203 |    0.801 |    0.146 |
|    0.00100 |       0.226 |    0.787 |    0.260 |
|    0.00010 |       0.154 |    0.846 |    0.000 |

[Official port] Visualizing tail and density along top-2 direction (skew=0.216177)...

Using reimplemented top direction for visualization.

Validation: skew along chosen direction f is -0.216177
![official_utils_top2 full](official_utils_top2_full.png)
Tail subset shape (q=0.01): (5104, 27)
![official_utils_top2 tail](official_utils_top2_tail.png)
![official_utils_top2 density](official_utils_top2_density.png)

=== Global vs Tail statistics (Chicago Crimes) ===
| Metric                             | Global | Tail |
|------------------------------------|:------:|:----:|
| Positive rate (Arrest=1)          | 0.228 | 0.188 |
| Accuracy (Logistic Regression)    | 0.799 | 0.840 |
| F1 (Logistic Regression)          | 0.330 | 0.510 |

=== Multi-Percentile Evaluation Table (Chicago Crimes, full dataset) ===
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.171 |    0.832 |    0.239 |
|    0.01000 |       0.188 |    0.840 |    0.510 |
|    0.00100 |       0.134 |    0.844 |    0.290 |
|    0.00010 |       0.019 |    0.904 |    0.000 |

[Official port] Visualizing tail and density along top-3 direction (skew=0.112961)...

Using reimplemented top direction for visualization.

Validation: skew along chosen direction f is -0.113959
![official_utils_top3 full](official_utils_top3_full.png)
Tail subset shape (q=0.01): (5104, 27)
![official_utils_top3 tail](official_utils_top3_tail.png)
![official_utils_top3 density](official_utils_top3_density.png)

=== Global vs Tail statistics (Chicago Crimes) ===
| Metric                             | Global | Tail |
|------------------------------------|:------:|:----:|
| Positive rate (Arrest=1)          | 0.228 | 0.190 |
| Accuracy (Logistic Regression)    | 0.799 | 0.838 |
| F1 (Logistic Regression)          | 0.330 | 0.507 |

=== Multi-Percentile Evaluation Table (Chicago Crimes, full dataset) ===
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.176 |    0.827 |    0.248 |
|    0.01000 |       0.190 |    0.838 |    0.507 |
|    0.00100 |       0.192 |    0.885 |    0.642 |
|    0.00010 |       0.078 |    0.824 |    0.000 |
