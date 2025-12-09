# Chicago Crimes Official-Utils Runs (Grouped)

This merges run details with grouping/analysis. Groups ordered by parameter impact.

## Parameter impact (recap)
- Vector transfer (`disable_vector_transfer`): main driver of differences; turning it off changes found tails/directions.
- Normalization (`use_l2_norm`): secondary; small drifts only.
- Min-shift (`disable_min_shift`): negligible alone; minor interaction when vector_transfer is off.
- Data structure (`use_linkedlist`): negligible (tiny numeric drift).
- Initialization (`use_first_intersection_init`): negligible on its own.

## Group A 鈥?Baseline (vector_transfer ON, L1)
> Percentile tables nearly identical; toggling linkedlist/min_shift/first_init causes only tiny numeric drift.

### 20251208_173917
**Parameters:**
- `use_official_style`: True
- `use_linkedlist`: False
- `use_first_intersection_init`: True
- `disable_min_shift`: False
- `disable_vector_transfer`: False
- `top_k`: 10
- `min_angle_step`: 0.3141592653589793
- `n_samples`: 500
- `csv_path`: None
- `timestamp`: 20251208_173917

![20251208_173917 top grid](20251208_173917/official_utils_top_grid.png)

**Top-1 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.180 |    0.824 |    0.230 |
|    0.01000 |       0.203 |    0.801 |    0.146 |
|    0.00100 |       0.226 |    0.787 |    0.260 |
|    0.00010 |       0.154 |    0.846 |    0.000 |

**Top-2 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.171 |    0.832 |    0.239 |
|    0.01000 |       0.188 |    0.840 |    0.510 |
|    0.00100 |       0.134 |    0.844 |    0.290 |
|    0.00010 |       0.019 |    0.904 |    0.000 |

**Top-3 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.176 |    0.827 |    0.248 |
|    0.01000 |       0.190 |    0.838 |    0.507 |
|    0.00100 |       0.192 |    0.885 |    0.642 |
|    0.00010 |       0.078 |    0.824 |    0.000 |

### 20251208_174132
**Parameters:**
- `use_official_style`: True
- `use_linkedlist`: False
- `use_first_intersection_init`: False
- `disable_min_shift`: False
- `disable_vector_transfer`: False
- `top_k`: 10
- `min_angle_step`: 0.3141592653589793
- `n_samples`: 500
- `csv_path`: None
- `timestamp`: 20251208_174132

![20251208_174132 top grid](20251208_174132/official_utils_top_grid.png)

**Top-1 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.180 |    0.824 |    0.230 |
|    0.01000 |       0.203 |    0.801 |    0.146 |
|    0.00100 |       0.226 |    0.787 |    0.260 |
|    0.00010 |       0.154 |    0.846 |    0.000 |

**Top-2 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.171 |    0.832 |    0.239 |
|    0.01000 |       0.188 |    0.840 |    0.510 |
|    0.00100 |       0.134 |    0.844 |    0.290 |
|    0.00010 |       0.019 |    0.904 |    0.000 |

**Top-3 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.176 |    0.827 |    0.248 |
|    0.01000 |       0.190 |    0.838 |    0.507 |
|    0.00100 |       0.192 |    0.885 |    0.642 |
|    0.00010 |       0.078 |    0.824 |    0.000 |

### 20251208_174346
**Parameters:**
- `use_official_style`: True
- `use_linkedlist`: False
- `use_first_intersection_init`: False
- `disable_min_shift`: True
- `disable_vector_transfer`: False
- `top_k`: 10
- `min_angle_step`: 0.3141592653589793
- `n_samples`: 500
- `csv_path`: None
- `timestamp`: 20251208_174346

![20251208_174346 top grid](20251208_174346/official_utils_top_grid.png)

**Top-1 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.180 |    0.824 |    0.230 |
|    0.01000 |       0.203 |    0.801 |    0.146 |
|    0.00100 |       0.226 |    0.787 |    0.260 |
|    0.00010 |       0.154 |    0.846 |    0.000 |

**Top-2 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.171 |    0.832 |    0.239 |
|    0.01000 |       0.188 |    0.840 |    0.510 |
|    0.00100 |       0.134 |    0.844 |    0.290 |
|    0.00010 |       0.019 |    0.904 |    0.000 |

**Top-3 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.176 |    0.827 |    0.248 |
|    0.01000 |       0.190 |    0.838 |    0.507 |
|    0.00100 |       0.192 |    0.885 |    0.642 |
|    0.00010 |       0.078 |    0.824 |    0.000 |

### 20251208_174903
**Parameters:**
- `use_official_style`: True
- `use_linkedlist`: True
- `use_first_intersection_init`: False
- `disable_min_shift`: False
- `disable_vector_transfer`: False
- `top_k`: 10
- `min_angle_step`: 0.3141592653589793
- `n_samples`: 500
- `csv_path`: None
- `timestamp`: 20251208_174903

![20251208_174903 top grid](20251208_174903/official_utils_top_grid.png)

**Top-1 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.178 |    0.826 |    0.231 |
|    0.01000 |       0.204 |    0.800 |    0.146 |
|    0.00100 |       0.226 |    0.787 |    0.260 |
|    0.00010 |       0.173 |    0.808 |    0.000 |

**Top-2 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.171 |    0.832 |    0.239 |
|    0.01000 |       0.187 |    0.841 |    0.511 |
|    0.00100 |       0.138 |    0.841 |    0.274 |
|    0.00010 |       0.019 |    0.904 |    0.000 |

**Top-3 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.176 |    0.827 |    0.248 |
|    0.01000 |       0.190 |    0.838 |    0.507 |
|    0.00100 |       0.192 |    0.885 |    0.642 |
|    0.00010 |       0.078 |    0.824 |    0.000 |

### 20251208_175104
**Parameters:**
- `use_official_style`: True
- `use_linkedlist`: True
- `use_first_intersection_init`: True
- `disable_min_shift`: False
- `disable_vector_transfer`: False
- `top_k`: 10
- `min_angle_step`: 0.3141592653589793
- `n_samples`: 500
- `csv_path`: None
- `timestamp`: 20251208_175104

![20251208_175104 top grid](20251208_175104/official_utils_top_grid.png)

**Top-1 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.178 |    0.826 |    0.231 |
|    0.01000 |       0.204 |    0.800 |    0.146 |
|    0.00100 |       0.226 |    0.787 |    0.260 |
|    0.00010 |       0.173 |    0.808 |    0.000 |

**Top-2 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.171 |    0.832 |    0.239 |
|    0.01000 |       0.187 |    0.841 |    0.511 |
|    0.00100 |       0.138 |    0.841 |    0.274 |
|    0.00010 |       0.019 |    0.904 |    0.000 |

**Top-3 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.176 |    0.827 |    0.248 |
|    0.01000 |       0.190 |    0.838 |    0.507 |
|    0.00100 |       0.192 |    0.885 |    0.642 |
|    0.00010 |       0.078 |    0.824 |    0.000 |

### 20251208_175253
**Parameters:**
- `use_official_style`: True
- `use_linkedlist`: True
- `use_first_intersection_init`: False
- `disable_min_shift`: True
- `disable_vector_transfer`: False
- `top_k`: 10
- `min_angle_step`: 0.3141592653589793
- `n_samples`: 500
- `csv_path`: None
- `timestamp`: 20251208_175253

![20251208_175253 top grid](20251208_175253/official_utils_top_grid.png)

**Top-1 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.178 |    0.826 |    0.231 |
|    0.01000 |       0.204 |    0.800 |    0.146 |
|    0.00100 |       0.226 |    0.787 |    0.260 |
|    0.00010 |       0.173 |    0.808 |    0.000 |

**Top-2 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.171 |    0.832 |    0.239 |
|    0.01000 |       0.187 |    0.841 |    0.511 |
|    0.00100 |       0.138 |    0.841 |    0.274 |
|    0.00010 |       0.019 |    0.904 |    0.000 |

**Top-3 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.176 |    0.827 |    0.248 |
|    0.01000 |       0.190 |    0.838 |    0.507 |
|    0.00100 |       0.192 |    0.885 |    0.642 |
|    0.00010 |       0.078 |    0.824 |    0.000 |


## Group B 鈥?Vector-transfer OFF (biggest change)
> Turning off vector_transfer shifts tail metrics noticeably; dominant behavioral change.

### 20251208_174522
**Parameters:**
- `use_official_style`: True
- `use_linkedlist`: False
- `use_first_intersection_init`: False
- `disable_min_shift`: False
- `disable_vector_transfer`: True
- `top_k`: 10
- `min_angle_step`: 0.3141592653589793
- `n_samples`: 500
- `csv_path`: None
- `timestamp`: 20251208_174522

![20251208_174522 top grid](20251208_174522/official_utils_top_grid.png)

**Top-1 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.201 |    0.808 |    0.209 |
|    0.01000 |       0.252 |    0.760 |    0.241 |
|    0.00100 |       0.233 |    0.769 |    0.224 |
|    0.00010 |       0.286 |    0.735 |    0.381 |

**Top-2 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.200 |    0.811 |    0.332 |
|    0.01000 |       0.207 |    0.788 |    0.233 |
|    0.00100 |       0.114 |    0.871 |    0.298 |
|    0.00010 |       0.058 |    0.885 |    0.000 |

**Top-3 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.179 |    0.823 |    0.177 |
|    0.01000 |       0.211 |    0.795 |    0.151 |
|    0.00100 |       0.233 |    0.783 |    0.225 |
|    0.00010 |       0.200 |    0.844 |    0.462 |

### 20251208_174705
**Parameters:**
- `use_official_style`: True
- `use_linkedlist`: False
- `use_first_intersection_init`: True
- `disable_min_shift`: True
- `disable_vector_transfer`: True
- `top_k`: 10
- `min_angle_step`: 0.3141592653589793
- `n_samples`: 500
- `csv_path`: None
- `timestamp`: 20251208_174705

![20251208_174705 top grid](20251208_174705/official_utils_top_grid.png)

**Top-1 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.214 |    0.796 |    0.228 |
|    0.01000 |       0.214 |    0.788 |    0.216 |
|    0.00100 |       0.170 |    0.810 |    0.198 |
|    0.00010 |       0.255 |    0.745 |    0.250 |

**Top-2 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.202 |    0.807 |    0.205 |
|    0.01000 |       0.252 |    0.761 |    0.240 |
|    0.00100 |       0.236 |    0.768 |    0.224 |
|    0.00010 |       0.277 |    0.745 |    0.400 |

**Top-3 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.200 |    0.811 |    0.332 |
|    0.01000 |       0.207 |    0.788 |    0.233 |
|    0.00100 |       0.114 |    0.871 |    0.298 |
|    0.00010 |       0.058 |    0.885 |    0.000 |

### 20251208_175635
**Parameters:**
- `use_official_style`: True
- `use_linkedlist`: True
- `use_first_intersection_init`: False
- `disable_min_shift`: False
- `disable_vector_transfer`: True
- `top_k`: 10
- `min_angle_step`: 0.3141592653589793
- `n_samples`: 500
- `csv_path`: None
- `timestamp`: 20251208_175635

![20251208_175635 top grid](20251208_175635/official_utils_top_grid.png)

**Top-1 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.214 |    0.796 |    0.228 |
|    0.01000 |       0.213 |    0.789 |    0.215 |
|    0.00100 |       0.168 |    0.814 |    0.202 |
|    0.00010 |       0.255 |    0.745 |    0.250 |

**Top-2 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.201 |    0.808 |    0.209 |
|    0.01000 |       0.252 |    0.760 |    0.241 |
|    0.00100 |       0.233 |    0.769 |    0.224 |
|    0.00010 |       0.286 |    0.735 |    0.381 |

**Top-3 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.200 |    0.811 |    0.332 |
|    0.01000 |       0.207 |    0.788 |    0.233 |
|    0.00100 |       0.114 |    0.871 |    0.298 |
|    0.00010 |       0.058 |    0.885 |    0.000 |

### 20251208_175920
**Parameters:**
- `use_official_style`: True
- `use_linkedlist`: True
- `use_first_intersection_init`: True
- `disable_min_shift`: True
- `disable_vector_transfer`: True
- `top_k`: 10
- `min_angle_step`: 0.3141592653589793
- `n_samples`: 500
- `csv_path`: None
- `timestamp`: 20251208_175920

![20251208_175920 top grid](20251208_175920/official_utils_top_grid.png)

**Top-1 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.214 |    0.796 |    0.228 |
|    0.01000 |       0.213 |    0.789 |    0.215 |
|    0.00100 |       0.170 |    0.812 |    0.200 |
|    0.00010 |       0.255 |    0.745 |    0.250 |

**Top-2 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.201 |    0.808 |    0.209 |
|    0.01000 |       0.252 |    0.760 |    0.241 |
|    0.00100 |       0.233 |    0.769 |    0.224 |
|    0.00010 |       0.286 |    0.735 |    0.381 |

**Top-3 percentile table:**
| Percentile | PosRate_tail | Accuracy | F1-score |
|-----------:|-------------:|---------:|---------:|
|    1.00000 |       0.228 |    0.799 |    0.330 |
|    0.10000 |       0.200 |    0.811 |    0.332 |
|    0.01000 |       0.207 |    0.788 |    0.233 |
|    0.00100 |       0.114 |    0.871 |    0.298 |
|    0.00010 |       0.058 |    0.885 |    0.000 |
