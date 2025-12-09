# College Admission Official-Utils – Outcome Grouping Summary

This note groups the runs in `ALL_RUNS_GRIDS.md` by observed outcome patterns and highlights which parameters drive visible changes in percentile metrics.

## Outcome Groups

- **Group A — Baseline (vector_transfer ON, L1)**  
  Runs: `20251208_205408`, `20251208_205427`, `20251208_205445` (min_shift off), `20251208_205538`, `20251208_205557`, `20251208_205619`  
  Behavior: Percentile tables match or are nearly identical; only tiny numeric drift when `use_linkedlist` is on.  
  Params: `disable_vector_transfer=False`, `use_l2_norm=False`; toggling `use_first_intersection_init`, `use_linkedlist`, or `disable_min_shift` alone does not materially change results.

- **Group B — Vector-transfer OFF (largest change)**  
  Runs: `20251208_205502`, `20251208_205521`, `20251208_205636`, `20251208_205653`, `20251208_205816`  
  Behavior: Percentile tables shift noticeably (accuracy/F1/F/M ratios differ across tails). This is the dominant source of output change.  
  Params: `disable_vector_transfer=True` (or the official port without vector-transfer mapping).

- **Group C — L2 normalization (vector_transfer ON)**  
  Runs: `20251208_205710`, `20251208_205727`, `20251208_205744`, `20251208_205759`  
  Behavior: Small, consistent shifts vs. baseline (slightly different tail metrics); impact is modest compared to turning off vector_transfer.  
  Params: `use_l2_norm=True`, `disable_vector_transfer=False`.

## Parameter Impact (what actually moves the needle)

- **Vector transfer** (`disable_vector_transfer`): Turning it off drives the clearest, repeatable changes and alters which directions/tails are found.
- **Normalization** (`use_l2_norm`): Secondary sensitivity; shifts are modest.
- **Min-shift** (`disable_min_shift`): Little to no visible effect on its own; minor interaction only when vector_transfer is already off.
- **Data structure** (`use_linkedlist`): Negligible effect (only tiny numeric drift).
- **Initialization** (`use_first_intersection_init`): Negligible on its own; no qualitative change.

## Recommendations

- For stable, “official-style” outcomes: keep `disable_vector_transfer=False` and `use_l2_norm=False`; other toggles can vary without meaningful impact.
- To study the main behavioral difference: compare baseline (Group A) vs. vector_transfer OFF (Group B).
- To examine secondary effects: compare baseline (Group A) vs. L2 runs (Group C).

