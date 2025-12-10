# Chicago Crimes Official-Utils Experiment (20251208_175920)

Loading Chicago Crimes dataset (official-port experiment)...
No --csv-path provided. Downloading Chicago Crimes dataset via kagglehub...
Using dataset file: C:\Users\james\.cache\kagglehub\datasets\currie32\crimes-in-chicago\versions\1\Chicago_Crimes_2012_to_2017.csv
Loaded dataset with shape: (510372, 27)
Primary point set shape: (500, 2)
Rotated point set shape: (500, 2)

Visualizing primary and rotated point sets (official utils)...
![official_utils point sets](official_utils_point_sets.png)

Running ray-sweeping (official vs official-style)...
Total time (official-style): 2.5001 s

Top-k high-skew directions (official port on rotated set):
| Rank | Direction (x, y) | Skew |
|------|------------------|------|
| 1 | (0.999962, 0.000038) | 0.266844 |
| 2 | (0.752179, 0.247821) | 0.213582 |
| 3 | (0.570227, 0.429773) | 0.112963 |
| 4 | (0.221478, 0.778522) | 0.029929 |
| 5 | (0.402659, 0.597341) | 0.023163 |

[Official port] Visualizing tail and density along top-1 direction (skew=0.266844)...

Using reimplemented top direction for visualization.

Validation: skew along chosen direction f is -0.063947
![official_utils_top1 full](official_utils_top1_full.png)
Tail subset shape (q=0.01): (5104, 27)
![official_utils_top1 tail](official_utils_top1_tail.png)
![official_utils_top1 density](official_utils_top1_density.png)

=== Global vs Tail statistics (Chicago Crimes) ===
| Metric                             | Global | Tail |
|------------------------------------|:------:|:----:|
| Positive rate (Arrest=1)          | 0.228 | 0.213 |
| Accuracy (Logistic Regression)    | 0.799 | 0.789 |
| F1 (Logistic Regression)          | 0.330 | 0.215 |

=== Multi-Percentile Evaluation Table (Chicago Crimes, full dataset) ===
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.214 |    0.796 |    0.228 |
|    0.01000 |       0.213 |    0.789 |    0.215 |
|    0.00100 |       0.170 |    0.812 |    0.200 |
|    0.00010 |       0.255 |    0.745 |    0.250 |

[Official port] Visualizing tail and density along top-2 direction (skew=0.213582)...

Using reimplemented top direction for visualization.

Validation: skew along chosen direction f is -0.023864
![official_utils_top2 full](official_utils_top2_full.png)
Tail subset shape (q=0.01): (5017, 27)
![official_utils_top2 tail](official_utils_top2_tail.png)
![official_utils_top2 density](official_utils_top2_density.png)

=== Global vs Tail statistics (Chicago Crimes) ===
| Metric                             | Global | Tail |
|------------------------------------|:------:|:----:|
| Positive rate (Arrest=1)          | 0.228 | 0.252 |
| Accuracy (Logistic Regression)    | 0.799 | 0.760 |
| F1 (Logistic Regression)          | 0.330 | 0.241 |

=== Multi-Percentile Evaluation Table (Chicago Crimes, full dataset) ===
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.201 |    0.808 |    0.209 |
|    0.01000 |       0.252 |    0.760 |    0.241 |
|    0.00100 |       0.233 |    0.769 |    0.224 |
|    0.00010 |       0.286 |    0.735 |    0.381 |

[Official port] Visualizing tail and density along top-3 direction (skew=0.112963)...

Using reimplemented top direction for visualization.

Validation: skew along chosen direction f is 0.118742
![official_utils_top3 full](official_utils_top3_full.png)
Tail subset shape (q=0.01): (5104, 27)
![official_utils_top3 tail](official_utils_top3_tail.png)
![official_utils_top3 density](official_utils_top3_density.png)

=== Global vs Tail statistics (Chicago Crimes) ===
| Metric                             | Global | Tail |
|------------------------------------|:------:|:----:|
| Positive rate (Arrest=1)          | 0.228 | 0.207 |
| Accuracy (Logistic Regression)    | 0.799 | 0.788 |
| F1 (Logistic Regression)          | 0.330 | 0.233 |

=== Multi-Percentile Evaluation Table (Chicago Crimes, full dataset) ===
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.200 |    0.811 |    0.332 |
|    0.01000 |       0.207 |    0.788 |    0.233 |
|    0.00100 |       0.114 |    0.871 |    0.298 |
|    0.00010 |       0.058 |    0.885 |    0.000 |
