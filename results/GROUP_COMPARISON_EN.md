# Chicago Crimes — Group A vs Group B: Parameter Settings & Result Differences

## Parameter Configuration Summary

### Group A — Baseline Configuration
**Key Setting:** `disable_vector_transfer: False` (vector_transfer **ON**)

**Runs:** 6 experiments
- `20251208_173917`: linkedlist=False, first_init=True, min_shift=ON
- `20251208_174132`: linkedlist=False, first_init=False, min_shift=ON
- `20251208_174346`: linkedlist=False, first_init=False, min_shift=OFF
- `20251208_174903`: linkedlist=True, first_init=False, min_shift=ON
- `20251208_175104`: linkedlist=True, first_init=True, min_shift=ON
- `20251208_175253`: linkedlist=True, first_init=False, min_shift=OFF

**Common Parameters:**
- `use_official_style`: True
- `disable_vector_transfer`: **False** (vector_transfer enabled)
- `use_l2_norm`: None/False (L1 normalization)
- `top_k`: 10
- `min_angle_step`: 0.314 (π/10)
- `n_samples`: 500

**Parameter Variations (within Group A):**
- `use_linkedlist`: False (3 runs) / True (3 runs)
- `use_first_intersection_init`: False (4 runs) / True (2 runs)
- `disable_min_shift`: False (4 runs) / True (2 runs)

**Observation:** Despite these variations, percentile tables are nearly identical across all 6 runs, confirming that `linkedlist`, `first_init`, and `min_shift` have negligible impact when vector_transfer is ON.

---

### Group B — Vector-transfer Disabled
**Key Setting:** `disable_vector_transfer: True` (vector_transfer **OFF**)

**Runs:** 4 experiments
- `20251208_174522`: linkedlist=False, first_init=False, min_shift=ON
- `20251208_174705`: linkedlist=False, first_init=True, min_shift=OFF
- `20251208_175635`: linkedlist=True, first_init=False, min_shift=ON
- `20251208_175920`: linkedlist=True, first_init=True, min_shift=OFF

**Common Parameters:**
- `use_official_style`: True
- `disable_vector_transfer`: **True** (vector_transfer disabled)
- `use_l2_norm`: None/False (L1 normalization)
- `top_k`: 10
- `min_angle_step`: 0.314 (π/10)
- `n_samples`: 500

**Parameter Variations (within Group B):**
- `use_linkedlist`: False (2 runs) / True (2 runs)
- `use_first_intersection_init`: False (2 runs) / True (2 runs)
- `disable_min_shift`: False (2 runs) / True (2 runs)

**Observation:** Group B runs show consistent differences from Group A, with minor variations within the group due to the same secondary parameters.

---

## Core Parameter Difference

| Parameter | Group A | Group B |
|-----------|---------|---------|
| **`disable_vector_transfer`** | **False** (ON) | **True** (OFF) |
| `use_l2_norm` | None/False (L1) | None/False (L1) |
| `use_linkedlist` | Mixed | Mixed |
| `use_first_intersection_init` | Mixed | Mixed |
| `disable_min_shift` | Mixed | Mixed |

**The only systematic difference:** `disable_vector_transfer` setting.

---

## Result Differences — Percentile Analysis

### 1. PosRate_tail (Positive Rate in Tail)

**Group A (representative: 173917/174132):**
- q=1.0: 0.228 (global baseline)
- q=0.1: 0.171–0.180
- q=0.01: 0.188–0.203
- q=0.001: 0.134–0.226
- q=0.0001: 0.019–0.154 (very sparse)

**Group B (representative: 174522, 175635, 175920):**
- q=1.0: 0.228 (same global baseline)
- q=0.1: 0.200–0.214 (slightly higher)
- q=0.01: 0.207–0.252 (**higher** than Group A)
- q=0.001: 0.114–0.233 (similar or higher)
- q=0.0001: 0.058–0.286 (**much higher** than Group A)

**Key Finding:** Group B maintains higher positive rates at mid-to-extreme percentiles (q≤0.01), especially at q=0.0001 where Group A drops to 0.019–0.154 while Group B reaches 0.058–0.286.

---

### 2. F1-score

**Group A:**
- q=0.1: F1 ≈ 0.23–0.25
- q=0.01: F1 ≈ 0.15–0.51 (varies by top direction)
- q=0.001: F1 ≈ 0.26–0.64 (some runs peak here)
- q=0.0001: **F1 = 0.000** (positives vanish, F1 collapses)

**Group B:**
- q=0.1: F1 ≈ 0.18–0.33 (similar or slightly lower)
- q=0.01: F1 ≈ 0.15–0.24 (similar range)
- q=0.001: F1 ≈ 0.20–0.30 (maintained)
- q=0.0001: **F1 = 0.25–0.46** (e.g., 174522 top-1: 0.381, top-3: 0.462; 175920 top-2: 0.381)

**Key Finding:** At q=0.0001, Group B **retains non-zero F1** (0.25–0.46) while Group A collapses to 0. This indicates Group B directions capture denser positive pockets even at extreme tails.

---

### 3. Accuracy

**Group A:**
- q=0.1: 0.82–0.83
- q=0.01: 0.80–0.84
- q=0.001: 0.78–0.89 (spikes when positives are rare)
- q=0.0001: 0.80–0.90 (high because tails are mostly negative)

**Group B:**
- q=0.1: 0.80–0.81 (slightly lower)
- q=0.01: 0.76–0.79 (lower, reflects higher positive rate)
- q=0.001: 0.77–0.87 (similar range)
- q=0.0001: 0.74–0.89 (lower at extreme due to retained positives)

**Key Finding:** Group B shows slightly lower Accuracy at mid percentiles (q=0.01–0.001) because it captures more positives (higher PosRate_tail), which may include more false positives. At q=0.0001, Group B Accuracy is lower (0.74–0.75) than Group A (0.80–0.90) because Group B retains positives while Group A tails are nearly all negative.

---

## Visual Differences (3×3 Grids)

**Group A (e.g., `20251208_173917/official_utils_top_grid.png`):**
- **Full panels:** Uniform direction footprints; standard orientation
- **Tail panels:** Sparse, dispersed tail regions; few concentrated clusters
- **Density panels:** Low positive density; scattered points

**Group B (e.g., `20251208_174522/official_utils_top_grid.png`, `20251208_175920/official_utils_top_grid.png`):**
- **Full panels:** Different direction orientation (no vector_transfer mapping back)
- **Tail panels:** More concentrated, tighter tail regions; higher point density
- **Density panels:** Denser positive blobs; more visible clusters

---

## Summary: What Actually Changes

### Parameter Impact Hierarchy

1. **`disable_vector_transfer`** (ON vs OFF):
   - **Major impact:** Changes direction set; alters tail composition
   - **Result:** Group B finds directions with denser positive pockets at mid/extreme percentiles
   - **Evidence:** F1 remains non-zero at q=0.0001 in Group B vs F1=0 in Group A

2. **`use_linkedlist`, `use_first_intersection_init`, `disable_min_shift`**:
   - **Negligible impact:** Only tiny numeric drift within each group
   - **Evidence:** Group A runs are nearly identical despite these variations; Group B runs show minor differences but same overall pattern

### Core Result Differences

| Metric | Group A | Group B | Difference |
|--------|---------|---------|------------|
| **PosRate_tail (q=0.0001)** | 0.019–0.154 | 0.058–0.286 | **Group B 2–15× higher** |
| **F1-score (q=0.0001)** | 0.000 | 0.25–0.46 | **Group B retains F1; Group A collapses** |
| **Accuracy (q=0.0001)** | 0.80–0.90 | 0.74–0.89 | Group B slightly lower (due to retained positives) |
| **Tail concentration** | Sparse, dispersed | Concentrated, dense | Group B more structured |

### Interpretation

- **Group A (vector_transfer ON):** Directions are mapped back to original space, producing sparse tails dominated by negatives. At extreme percentiles, positives vanish → F1=0.
- **Group B (vector_transfer OFF):** Directions are not mapped back; algorithm explores rotated space directly. This finds different directions that capture denser positive pockets, maintaining F1 even at q=0.0001.

**Conclusion:** The `disable_vector_transfer` parameter is the **sole systematic driver** of meaningful result differences. All other parameters (`linkedlist`, `first_init`, `min_shift`) cause only negligible numeric drift.

