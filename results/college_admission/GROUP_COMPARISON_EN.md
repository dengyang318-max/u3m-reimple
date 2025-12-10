# College Admission — Group A vs Group B vs Group C: Parameter Settings & Result Differences

## Parameter Configuration Summary

### Group A — Baseline Configuration
**Key Setting:** `disable_vector_transfer: False` (vector_transfer **ON**)

**Runs:** 6 experiments
- `20251208_205408`: linkedlist=False, first_init=False, min_shift=ON
- `20251208_205427`: linkedlist=False, first_init=True, min_shift=ON
- `20251208_205445`: linkedlist=False, first_init=False, min_shift=OFF
- `20251208_205538`: linkedlist=True, first_init=False, min_shift=ON
- `20251208_205557`: linkedlist=True, first_init=True, min_shift=ON
- `20251208_205619`: linkedlist=True, first_init=False, min_shift=OFF

**Common Parameters:**
- `use_official_style`: True
- `disable_vector_transfer`: **False** (vector_transfer enabled)
- `use_l2_norm`: False (L1 normalization)
- `top_k`: 10
- `min_angle_step`: 0.314 (π/10)
- `n_samples`: None (uses all 400 records)

**Parameter Variations (within Group A):**
- `use_linkedlist`: False (3 runs) / True (3 runs)
- `use_first_intersection_init`: False (4 runs) / True (2 runs)
- `disable_min_shift`: False (4 runs) / True (2 runs)

**Observation:** Despite these variations, percentile tables are nearly identical across all 6 runs, confirming that `linkedlist`, `first_init`, and `min_shift` have negligible impact when vector_transfer is ON.

---

### Group B — Vector-transfer Disabled
**Key Setting:** `disable_vector_transfer: True` (vector_transfer **OFF**)

**Runs:** 5 experiments
- `20251208_205502`: linkedlist=False, first_init=False, min_shift=ON
- `20251208_205521`: linkedlist=False, first_init=True, min_shift=OFF
- `20251208_205636`: linkedlist=True, first_init=False, min_shift=ON
- `20251208_205653`: linkedlist=True, first_init=True, min_shift=OFF
- `20251208_205816`: official port (use_official_style=False), vector_transfer effectively OFF

**Common Parameters:**
- `disable_vector_transfer`: **True** (vector_transfer disabled)
- `use_l2_norm`: False (L1 normalization)
- `top_k`: 10
- `min_angle_step`: 0.314 (π/10)
- `n_samples`: None (uses all 400 records)

**Parameter Variations (within Group B):**
- `use_linkedlist`: False (2 runs) / True (2 runs) / N/A (1 official port)
- `use_first_intersection_init`: False (2 runs) / True (2 runs) / N/A (1 official port)
- `disable_min_shift`: False (2 runs) / True (2 runs) / N/A (1 official port)

**Observation:** Group B runs show consistent differences from Group A, with minor variations within the group due to the same secondary parameters.

---

### Group C — L2 Normalization (vector_transfer ON)
**Key Setting:** `use_l2_norm: True` (L2 normalization), `disable_vector_transfer: False` (vector_transfer **ON**)

**Runs:** 4 experiments
- `20251208_205710`: linkedlist=False, first_init=False, min_shift=ON
- `20251208_205727`: linkedlist=True, first_init=False, min_shift=ON
- `20251208_205744`: linkedlist=False, first_init=True, min_shift=ON
- `20251208_205759`: linkedlist=True, first_init=True, min_shift=OFF

**Common Parameters:**
- `use_official_style`: True
- `disable_vector_transfer`: **False** (vector_transfer enabled)
- `use_l2_norm`: **True** (L2 normalization)
- `top_k`: 10
- `min_angle_step`: 0.314 (π/10)
- `n_samples`: None (uses all 400 records)

**Parameter Variations (within Group C):**
- `use_linkedlist`: False (2 runs) / True (2 runs)
- `use_first_intersection_init`: False (2 runs) / True (2 runs)
- `disable_min_shift`: False (3 runs) / True (1 run)

**Observation:** Group C results are very similar to Group A, with only small, consistent drifts. This confirms that L2 vs L1 normalization has a much smaller impact than disabling vector_transfer.

---

## Core Parameter Differences

| Parameter | Group A | Group B | Group C |
|-----------|---------|---------|---------|
| **`disable_vector_transfer`** | **False** (ON) | **True** (OFF) | **False** (ON) |
| **`use_l2_norm`** | **False** (L1) | **False** (L1) | **True** (L2) |
| `use_linkedlist` | Mixed | Mixed | Mixed |
| `use_first_intersection_init` | Mixed | Mixed | Mixed |
| `disable_min_shift` | Mixed | Mixed | Mixed |

**Systematic differences:**
- **Group A vs Group B:** `disable_vector_transfer` (ON vs OFF) — **major impact**
- **Group A vs Group C:** `use_l2_norm` (L1 vs L2) — **minor impact**
- **Group B vs Group C:** Both `disable_vector_transfer` (OFF vs ON) and `use_l2_norm` (L1 vs L2) differ

---

## Result Differences — Percentile Analysis

**Percentiles used:** q = 1.0, 0.5, 0.2, 0.1, 0.08, 0.04

### 1. F1-score

**Group A (representative: 205408/205427/205445):**

**Top-1:**
- q=1.0: 0.320
- q=0.5: 0.254
- q=0.2: 0.308
- q=0.1: **0.000** (F1 collapses)
- q=0.08: **0.000**
- q=0.04: **0.000**

**Top-2:**
- q=1.0: 0.321
- q=0.5: 0.375
- q=0.2: 0.409
- q=0.1: 0.364
- q=0.08: 0.444
- q=0.04: 0.400

**Top-3:**
- q=1.0: 0.320
- q=0.5: 0.350
- q=0.2: 0.364
- q=0.1: 0.286
- q=0.08: 0.250
- q=0.04: 0.500

**Group B (representative: 205502, 205521, 205636, 205653):**

**Top-1:**
- q=1.0: 0.320
- q=0.5: 0.407–0.400
- q=0.2: 0.492–0.481
- q=0.1: **0.562–0.452** (F1 maintained)
- q=0.08: **0.571–0.538** (F1 maintained)
- q=0.04: **0.533–0.625** (F1 maintained)

**Top-2:**
- q=1.0: 0.320–0.307
- q=0.5: 0.400–0.407
- q=0.2: 0.481–0.483
- q=0.1: 0.452–0.562
- q=0.08: 0.538–0.571
- q=0.04: 0.533–0.545

**Top-3:**
- q=1.0: 0.307–0.320
- q=0.5: 0.128–0.400
- q=0.2: **0.000–0.517** (varies by run; some collapse, some maintain)
- q=0.1: **0.000–0.529** (varies)
- q=0.08: **0.000–0.500** (varies)
- q=0.04: **0.000–0.462** (varies)

**Group C (representative: 205710, 205744):**

**Top-1:**
- q=1.0: 0.320
- q=0.5: 0.254
- q=0.2: 0.250–0.308
- q=0.1: **0.000** (F1 collapses, same as Group A)
- q=0.08: **0.000**
- q=0.04: **0.000**

**Top-2:**
- q=1.0: 0.320–0.321
- q=0.5: 0.375
- q=0.2: 0.409
- q=0.1: 0.364
- q=0.08: 0.444
- q=0.04: 0.400

**Top-3:**
- q=1.0: 0.320
- q=0.5: 0.350
- q=0.2: 0.364
- q=0.1: 0.286–0.381
- q=0.08: 0.250–0.353
- q=0.04: 0.400–0.500

**Key Finding:** 
- **Top-1:** Group B maintains F1 (0.45–0.63) at q=0.1–0.04, while Group A and Group C both collapse to 0. This is the most dramatic difference.
- **Top-2:** All three groups maintain F1, with Group B showing slightly higher values at tight percentiles (q=0.1–0.04). Group C is nearly identical to Group A.
- **Top-3:** Group B shows more variability; some runs maintain F1 while others collapse (similar to Group A/C pattern). Group C shows minor variations from Group A (e.g., 205727/205759 have slightly different values).

---

### 2. Accuracy

**Group A:**

**Top-1:**
- q=0.1: 0.775
- q=0.08: 0.844
- q=0.04: 0.875 (high because tails are mostly negative)

**Top-2/3:**
- q=0.1: 0.641–0.750
- q=0.08: 0.677–0.812
- q=0.04: 0.800–0.875

**Group B:**

**Top-1:**
- q=0.1: 0.575–0.650 (lower than Group A)
- q=0.08: 0.625 (lower)
- q=0.04: 0.562–0.625 (much lower than Group A)

**Top-2:**
- q=0.1: 0.575–0.650
- q=0.08: 0.625
- q=0.04: 0.562–0.583

**Top-3:**
- q=0.1: 0.590–0.825 (varies; some runs high like Group A)
- q=0.08: 0.562–0.812 (varies)
- q=0.04: 0.533–0.812 (varies)

**Group C:**

**Top-1:**
- q=0.1: 0.775
- q=0.08: 0.844
- q=0.04: 0.875 (same as Group A)

**Top-2/3:**
- q=0.1: 0.641–0.750
- q=0.08: 0.677–0.812
- q=0.04: 0.800–0.875

**Key Finding:** 
- Group B shows **lower Accuracy** at tight percentiles (q=0.1–0.04) for Top-1/2, especially at q=0.04 where Group A and Group C reach 0.84–0.88 while Group B drops to 0.56–0.63. This reflects that Group B captures more positives (higher F1), which may include more false positives, reducing Accuracy.
- **Group C Accuracy is nearly identical to Group A**, confirming that L2 vs L1 normalization has minimal impact on Accuracy.

---

### 3. F/M Ratio (Female/Male Ratio)

**Group A:**
- Top-1: F/M ≈ 1.0–1.2 across percentiles
- Top-2: F/M ≈ 0.8–2.0 (more variable)
- Top-3: F/M ≈ 1.0–1.3

**Group B:**
- Top-1: F/M ≈ 0.9–1.3 (similar range)
- Top-2: F/M ≈ 0.7–1.4 (more variable, includes lower values)
- Top-3: F/M ≈ 0.6–1.5 (more variable)

**Group C:**
- Top-1: F/M ≈ 1.0–1.2 (same as Group A)
- Top-2: F/M ≈ 0.8–2.0 (similar to Group A)
- Top-3: F/M ≈ 0.6–1.3 (slightly more variable than Group A)

**Key Finding:** 
- Group B shows **more volatile F/M ratios**, especially at tight percentiles, with some runs showing ratios < 1.0 (male-dominated) or > 1.4 (female-dominated), indicating stronger demographic shifts in discovered tails.
- **Group C F/M ratios are very similar to Group A**, with only minor variations, confirming that normalization method has minimal impact on demographic composition.

---

## Visual Differences (3×3 Grids)

**Group A (e.g., `20251208_205408/official_utils_top_grid.png`):**
- **Full panels:** Uniform direction footprints; standard orientation
- **Tail panels:** Sparse tail regions; few concentrated clusters
- **Density panels:** Low positive density; scattered points

**Group B (e.g., `20251208_205502/official_utils_top_grid.png`, `20251208_205636/official_utils_top_grid.png`):**
- **Full panels:** Different direction orientation (no vector_transfer mapping back)
- **Tail panels:** More concentrated, tighter tail regions; higher point density
- **Density panels:** Denser positive blobs; more visible clusters

**Group C (e.g., `20251208_205710/official_utils_top_grid.png`, `20251208_205744/official_utils_top_grid.png`):**
- **Full panels:** Similar to Group A (same vector_transfer mapping)
- **Tail panels:** Very similar to Group A; sparse, dispersed
- **Density panels:** Similar to Group A; low positive density

**Visual Observation:** Group C grids are nearly indistinguishable from Group A, confirming that L2 vs L1 normalization has minimal visual impact.

---

## Summary: What Actually Changes

### Parameter Impact Hierarchy

1. **`disable_vector_transfer`** (ON vs OFF):
   - **Major impact:** Changes direction set; alters tail composition
   - **Result:** Group B finds directions that maintain F1 at tight percentiles (q=0.1–0.04) for Top-1, while Group A and Group C collapse to F1=0
   - **Evidence:** Top-1 F1 at q=0.04: Group B = 0.53–0.63 vs Group A = 0.00 vs Group C = 0.00

2. **`use_l2_norm`** (L1 vs L2):
   - **Minor impact:** Small, consistent drifts; much smaller than vector_transfer
   - **Result:** Group C shows nearly identical results to Group A, with only minor variations (e.g., Top-1 F1 at q=0.2: Group C = 0.250 vs Group A = 0.308)
   - **Evidence:** Group C percentile tables are almost identical to Group A; visual grids are indistinguishable

3. **`use_linkedlist`, `use_first_intersection_init`, `disable_min_shift`**:
   - **Negligible impact:** Only tiny numeric drift within each group
   - **Evidence:** Runs within each group are nearly identical despite these variations

### Core Result Differences

| Metric | Group A | Group B | Group C | Key Difference |
|--------|---------|---------|---------|----------------|
| **F1-score Top-1 (q=0.04)** | 0.000 | 0.53–0.63 | 0.000 | **Group B maintains F1; A & C collapse** |
| **F1-score Top-1 (q=0.1)** | 0.000 | 0.45–0.56 | 0.000 | **Group B maintains F1; A & C collapse** |
| **Accuracy Top-1 (q=0.04)** | 0.84–0.88 | 0.56–0.63 | 0.84–0.88 | Group B lower; A & C similar |
| **F/M Ratio variability** | Moderate | High | Moderate | Group B shows stronger shifts |
| **Tail concentration** | Sparse, dispersed | Concentrated, dense | Sparse, dispersed | Group B more structured; A & C similar |

### Interpretation

- **Group A (vector_transfer ON, L1):** Directions are mapped back to original space, producing sparse tails. At tight percentiles (q≤0.1), Top-1 F1 collapses to 0 because positives vanish. Top-2/3 may maintain F1 in some cases, but Top-1 consistently fails.

- **Group B (vector_transfer OFF, L1):** Directions are not mapped back; algorithm explores rotated space directly. This finds different directions that capture denser positive pockets, maintaining Top-1 F1 even at q=0.04 (0.53–0.63). However, this comes at the cost of lower Accuracy (0.56–0.63 vs 0.84–0.88) because more positives are captured, potentially including more false positives.

- **Group C (vector_transfer ON, L2):** Uses L2 normalization instead of L1, but with vector_transfer still ON. Results are nearly identical to Group A, with only minor variations (e.g., Top-1 F1 at q=0.2: 0.250 vs 0.308). This confirms that normalization method (L1 vs L2) has minimal impact compared to vector_transfer.

**Conclusion:** 
- The `disable_vector_transfer` parameter is the **primary systematic driver** of meaningful result differences (Group A/C vs Group B).
- The `use_l2_norm` parameter has **minor impact** (Group A vs Group C), causing only small, consistent drifts.
- All other parameters (`linkedlist`, `first_init`, `min_shift`) cause only negligible numeric drift.
- **Key trade-off:** Group B maintains F1 at tight percentiles but sacrifices Accuracy, while Group A and Group C achieve high Accuracy at tight percentiles but lose F1 entirely.

