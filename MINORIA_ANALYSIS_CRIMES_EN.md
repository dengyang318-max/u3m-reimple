# Chicago Crimes Official-Utils — Tail/Percentile Analysis (Groups A & B)

Using `ALL_RUNS_GRIDS_GROUPED.md` (now only Groups A and B), summarizing minoria/tail behavior with percentile tables and 3×3 grids.

## Quick grouping
- **Group A (vector_transfer ON, L1)**: Runs `20251208_173917`, `174132`, `174346`, `174903`, `175104`, `175253`. Percentile tables nearly identical; tiny drift from linkedlist/min_shift/first_init.
- **Group B (vector_transfer OFF)**: Runs `20251208_174522`, `174705`, `175635`, `175920`. Turning off vector_transfer materially changes tails/directions.

## Group A — What the tails look like
- **Percentiles:** q = 1.0, 0.1, 0.01, 0.001, 0.0001 (Top1/2/3 tables).
- **Tables (representative 173917/174132):**
  - Top-1: PosRate_tail ~0.18–0.23 at q=0.1; F1 ≈0.15–0.23. At q=0.0001 F1=0 (positives vanish), Accuracy spikes (~0.84).
  - Top-2/3: Similar pattern; at q=0.001 some runs reach F1≈0.64 (top-3) when a few positives remain; at q=0.0001 F1 often 0.
- **Visuals:** `20251208_173917/official_utils_top_grid.png` (and other A runs) — tails in tail/density panels are sparse and dispersed; no concentrated positive clusters; full panels look uniform.
- **Interpretation:** With vector_transfer ON, discovered directions produce sparse tails dominated by negatives; F1 collapses as percentiles tighten. Minoria is “geometric sparsity” rather than label-enrichment.

## Group B — What changes when vector_transfer is OFF
- **Percentiles:** Same q set. Material differences appear at mid/low tails.
- **Tables (e.g., 174522, 175635, 175920):**
  - Top-1: At q=0.01–0.001, PosRate_tail increases (e.g., 0.23–0.29) with F1 ~0.22–0.38 in some runs (e.g., 174522 top-1 F1=0.381 at q=0.0001). At q=0.1, F1 can be lower or higher depending on the run (reflects direction shift).
  - Top-2/3: Some runs keep non-zero F1 deeper into the tail (q=0.001 or 0.0001) compared to Group A, indicating directions that retain positives longer.
- **Visuals:** Grids like `20251208_174522/official_utils_top_grid.png`, `20251208_175920/official_utils_top_grid.png` show tail/density panels with more concentrated regions and higher positive color density than Group A. Full panels reflect different directions (no transfer back), explaining the altered tails.
- **Interpretation:** Disabling vector_transfer changes the direction set; tails capture denser positive pockets at mid/low percentiles, yielding higher F1 and more volatile tail composition.

## What actually moves the needle
- **Major:** `disable_vector_transfer` — Off → higher PosRate_tail and F1 at mid/low percentiles; tails more concentrated.
- **Minor/negligible:** `use_linkedlist`, `use_first_intersection_init`, `disable_min_shift` — only tiny ordering/offset effects in this dataset.
- **Note on extremes:** At q=0.0001, many runs (both groups) still hit F1=0 when positives vanish; Accuracy jumps because tails become tiny and mostly negative.

## How to read the grids per run
- **Full**: see the chosen direction footprint; Group B shows different orientation vs A.
- **Tail**: check tail region extent; Group B tails often tighter and more populated.
- **Density**: look for concentrated blobs; Group B shows denser blobs with positives in several runs.

