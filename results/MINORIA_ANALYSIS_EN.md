# College Admission Official-Utils — Minoria Characteristics by Parameter Groups

Based on `ALL_RUNS_GRIDS_GROUPED.md` (top-1/2/3 percentile tables), this note summarizes what the discovered minoria looks like under different parameter groups.

## Group A — Baseline (vector_transfer ON, L1)
- **Pattern:** At tight tails (q=0.10–0.04), F1 often drops to 0 while Accuracy stays high (≈0.77–0.88). Tails are sparse and dominated by negatives; positives are too few to score recall/F1.
- **Demographic skew:** F/M ratios hover near 1–1.8, showing only mild shifts.
- **Implication:** Minoria here is largely geometric sparsity, not label-enriched; demographic shift is weak.

## Group B — Vector-transfer OFF (biggest change)
- **Pattern:** At mid tails (q≈0.50–0.10) F1 is much higher than Group A (e.g., F1≈0.56–0.57 on some top-1 runs), meaning these directions retain substantially more positives. At the tightest tails (q=0.04) F1 can still collapse to 0 due to sparsity.
- **Demographic skew:** F/M ratios are more volatile (can exceed 1.4 or drop below 1), indicating stronger demographic shifts than Group A.
- **Implication:** Disabling vector_transfer uncovers directions whose tails are label-strong (more admits) and demographically more skewed. This is the only toggle that materially changes which minorities are surfaced.

## Group C — L2 Normalization (vector_transfer ON)
- **Pattern:** Very close to Group A — F1 often 0 at the tightest tails; mid-percentile metrics track the baseline with only small drifts.
- **Demographic skew:** Similar to Group A; no notable amplification.
- **Implication:** L2 vs. L1 introduces only minor numeric drift; minoria composition stays essentially the same as baseline.

## What actually changes the minoria
- **Vector transfer (`disable_vector_transfer`)**: Turning it off is the dominant factor; it raises positive density in mid-percentile tails and yields more varied F/M ratios.
- **Normalization (`use_l2_norm`)**: Only small, non-qualitative shifts.
- **Other toggles (`use_linkedlist`, `use_first_intersection_init`, `disable_min_shift`)**: Cause tiny ordering/offset effects; no substantive change to which minorities are found.

## Practical guidance
- For stable, “official-style” minorities: keep vector_transfer ON (Group A/C); expect sparse-positive tails and mild demographic skew.
- To explore alternative minorities with stronger label lift and demographic skew: compare Group B (vector_transfer OFF) against Group A.

