# Project Environment, Introduction, and Experiment Guide (English)

## 1) Overview
- **Goal:** Parameterized Ray-sweeping to study minority (high-skew) directions across datasets.
- **Key datasets:** Chicago Crimes (CSV via Kaggle), College Admission (Excel via Kaggle).
- **One unified core:** `ray_sweeping_2d_official_style.py` (configurable to match official vs. legacy behaviors).
- **Two unified experiment runners:**  
  - `u3m_reimpl/experiments/experiment_ray_sweeping_2d_chicago_crimes_official_utils.py`  
  - `u3m_reimpl/experiments/experiment_ray_sweeping_2d_college_admission_official_utils.py`

## 2) Environment Setup
### System
- OS: Windows / macOS / Linux
- Python: 3.10+ (recommend 3.11)

### Create & activate venv (example on Windows PowerShell)
```pwsh
python -m venv .venv
.venv\Scripts\activate
```
macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies
If `requirements.txt` exists:
```bash
pip install -r requirements.txt
```
If not, install the core set used by the experiments:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn pillow kagglehub
```

## 3) Data Preparation
### Chicago Crimes
- Default: auto-download via `kagglehub` inside the script if `--csv-path` is not provided.
- Manual: download `Chicago_Crimes_2012_to_2017.csv` (Kaggle: `currie32/crimes-in-chicago`), pass via `--csv-path`.
- Preprocessing in-script: drop NAs, derive time fields, filter Year 2015–2017, factorize categorical cols, normalize Lon/Lat to [0,1].

### College Admission
- Default: auto-download via `kagglehub` if `--excel-path` is not provided.
- Manual: download `Admission.xlsx` (Kaggle: `eswarchandt/admission`), pass via `--excel-path`.
- Preprocessing in-script: min–max normalize `gre`, `gpa` to [0,1]; shuffle with `random_state=0` (all rows by default).

## 4) Running Experiments (Typical Commands)
Activate your venv first, then run from repo root.

### Chicago Crimes (official-style runner)
Baseline (vector_transfer ON, L1):
```bash
python u3m_reimpl/experiments/experiment_ray_sweeping_2d_chicago_crimes_official_utils.py ^
  --top-k 10 --min-angle-step 0.314 --n-samples 500 --use-official-style
```
Vector-transfer OFF (Group B behavior):
```bash
python u3m_reimpl/experiments/experiment_ray_sweeping_2d_chicago_crimes_official_utils.py ^
  --top-k 10 --min-angle-step 0.314 --n-samples 500 --use-official-style ^
  --disable-vector-transfer
```
L2 normalization (optional):
```bash
python u3m_reimpl/experiments/experiment_ray_sweeping_2d_chicago_crimes_official_utils.py ^
  --top-k 10 --min-angle-step 0.314 --n-samples 500 --use-official-style ^
  --use-l2-norm
```

### College Admission (official-style runner)
Baseline (vector_transfer ON, L1):
```bash
python u3m_reimpl/experiments/experiment_ray_sweeping_2d_college_admission_official_utils.py ^
  --top-k 10 --min-angle-step 0.314
```
Vector-transfer OFF (Group B behavior):
```bash
python u3m_reimpl/experiments/experiment_ray_sweeping_2d_college_admission_official_utils.py ^
  --top-k 10 --min-angle-step 0.314 --disable-vector-transfer
```
L2 normalization (optional):
```bash
python u3m_reimpl/experiments/experiment_ray_sweeping_2d_college_admission_official_utils.py ^
  --top-k 10 --min-angle-step 0.314 --use-l2-norm
```

### Batch scripts (college)
- Bash: `u3m_reimpl/experiments/run_college_official_utils_commands.sh`
- PowerShell: `u3m_reimpl/experiments/run_college_official_utils_commands.ps1`

## 5) Key Parameters (impactful vs. minor)
- **High impact:** `--disable-vector-transfer` (changes discovered directions/tails), `--use-l2-norm` (secondary, small drift).
- **Minor/near-neutral:** `--use-linkedlist`, `--use-first-intersection-init`, `--disable-min-shift` (tiny numeric drift).
- **General controls:** `--top-k`, `--min-angle-step`, `--n-samples` (compute scale / sampling density).

## 6) Outputs & Organization
- Each run writes to `results/<dataset>_official_utils/<timestamp>/`:
  - `run_params.json` (full CLI config)
  - `*_log.md` (console + percentile tables)
  - `official_utils_top_grid.png` (3×3: full/tail/density × top1/2/3)
  - Point set plots and tail visualizations per top-k

## 7) Quick Troubleshooting
- **OOM / slow:** reduce `--n-samples` (Chicago), increase `--min-angle-step` to sample fewer directions.
- **Data download fails:** provide `--csv-path` or `--excel-path` manually.
- **Runtime errors on zero vectors:** already guarded by zero-vector checks in `ray_sweeping_2d_official_style.py`; ensure data not degenerate.

## 8) Recommended Presets
- Official-like (paper): `--use-official-style --use-linkedlist --min-angle-step 0.314` (L1, min-shift on, vector_transfer on, official init).
- Speed-oriented official-style: `--use-official-style --min-angle-step 0.314` (dict+list).
- Legacy-like ablation: `--use-official-style --use-first-intersection-init --use-l2-norm --disable-min-shift --disable-vector-transfer --min-angle-step 0.314`.

