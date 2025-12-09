# College Admission Official-Utils — Minoria Characteristics (with percentile tables & visuals)

Summarizes top-1/2/3 percentile behaviors using `ALL_RUNS_GRIDS_GROUPED.md` plus the 3×3 grids (full / tail / density).

## Key findings
- **Vector transfer on/off** is the only switch that materially changes which minorities are surfaced. Off → more positives in mid tails, higher F1, and more volatile F/M ratios.
- **L1 vs. L2** only nudges numbers; minority composition is essentially unchanged.
- **Other toggles (linkedlist, first_intersection_init, min_shift)** cause tiny ordering/offset effects, not substantive changes.
- At very tight tails (q ≤ 0.04) all groups often run out of positives → F1 collapses to 0; these tails are defined by geometric sparsity rather than label enrichment.

## Group A (baseline: vector_transfer ON, L1)
- **Percentile tables:** At q=0.10–0.04, F1 is often 0 while Accuracy stays high (~0.77–0.88), indicating tails are sparse and negative-heavy; F/M ≈1–1.8 (mild skew).
- **Visuals:** Tail/density plots show sparse, dispersed tail points; full plot looks evenly spread with no sharp skew cluster.
- **Takeaway:** Minorities are “sparse-direction” tails, not label-enriched; demographic skew is weak.
- **Example images:** `20251208_205408/official_utils_top_grid.png` (top-left 3×3 grid: rows full/tail/density; cols top1/2/3). Similar appearance across A-group runs.
- **Percentiles shown:** q = 1.00, 0.50, 0.20, 0.10, 0.08, 0.04 (all three top-k tables).

## Group B (vector_transfer OFF — biggest change)
- **Percentile tables:** Mid tails (q≈0.50–0.10) show much higher F1 than Group A (e.g., some top-1 runs ~0.56–0.57), meaning tails retain many more positives. Tight tails (q=0.04) can still drop to F1=0 due to sparsity. F/M ratios swing more (can be >1.4 or <1), showing stronger demographic shifts.
- **Visuals:** Tail/density plots show tails that are more concentrated and with a higher share of positives (by color). Full plots reflect different chosen directions (no vector transfer back), explaining the new tail composition.
- **Takeaway:** Turning off vector_transfer finds label-strong, more demographically skewed minorities. This is the only toggle that meaningfully changes the surfaced groups.
- **Example images:** `20251208_205502/official_utils_top_grid.png` (clearer positive concentration in tail/density for top-1/2), `20251208_205636/official_utils_top_grid.png` (similar pattern).
- **Percentiles shown:** q = 1.00, 0.50, 0.20, 0.10, 0.08, 0.04 (all three top-k tables).

## Group C (L2 normalization, vector_transfer ON)
- **Percentile tables:** Mirrors Group A; F1 at tight tails is often 0, mid-percentile metrics drift only slightly; F/M similar to A.
- **Visuals:** Tail/density similar to A — sparse tails, no new concentrated clusters.
- **Takeaway:** L2 introduces only minor numeric drift; minorities are essentially the same as baseline.
- **Example images:** `20251208_205710/official_utils_top_grid.png` (representative L2 run; visuals closely match A-group).
- **Percentiles shown:** q = 1.00, 0.50, 0.20, 0.10, 0.08, 0.04 (all three top-k tables).

## What moves the needle
- **Major:** `disable_vector_transfer` — Off → higher positive density in mid tails, more varied F/M.
- **Minor:** `use_l2_norm` — small, non-qualitative shifts.
- **Negligible:** `use_linkedlist`, `use_first_intersection_init`, `disable_min_shift` — only tiny ordering/offset effects.

## How to read a run with the grids
- Use **full** to see overall spread and chosen directions.
- Use **tail** to see which region is cut out; check if positives cluster.
- Use **density** to verify concentration vs. sparsity; relate to F1/Accuracy in the tables.

