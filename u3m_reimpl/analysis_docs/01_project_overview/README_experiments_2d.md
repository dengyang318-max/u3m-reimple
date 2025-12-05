## 2D Ray-sweeping Experiments on Real Datasets

This document explains how to run the reimplemented 2D Ray-sweeping algorithm
on the two real datasets used in the paper:

- College Admission
- Chicago Crimes

and what each experiment is doing conceptually.

All code lives in the `u3m_reimpl` package and uses **only the naive
O(n²) intersection enumeration** (no incremental / randomized variants),
to match the warm‑up experimental setting in the paper.

---

## 1. Environment Requirements

- Python 3.9+ (same as your main project)
- Dependencies (already covered by your project environment):
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn` (only for the original notebooks’ black-box models; **not**
    required by our reimplementation scripts)
  - `kagglehub` (optional, only needed if you want the scripts to
    auto‑download datasets)

Install `kagglehub` if you want automatic dataset download:

```bash
pip install kagglehub
```

You can also skip `kagglehub` and point the scripts directly to local files.

All commands below assume you are in the project root:

```bash
cd C:\Users\james\Desktop\5003-project
```

---

## 2. College Admission 2D Experiment

### 2.1 Script and Entry Point

Implementation file:

- `u3m_reimpl/experiment_ray_sweeping_2d_college_admission.py`

You can run it in **two ways**:

1. Direct Python call (with optional path):

```bash
python -m u3m_reimpl.experiment_ray_sweeping_2d_college_admission \
  --excel-path "Mining_U3Ms-main/Mining_U3Ms-main/2D/Admission.xlsx" \
  --top-k 10 \
  --min-angle-step 0.3141592653589793
```

2. Using the convenience shell script (for bash / WSL / Git Bash):

```bash
cd u3m_reimpl
./college.sh
```

> On Windows PowerShell you can still run the Python command directly, or run
> `bash college.sh` if you have Git Bash or WSL installed.

### 2.2 Dataset Loading and Preprocessing

The helper function `load_college_admission_data(...)` does:

- Load an Excel file containing at least:
  - `gre`, `gpa`, and optionally `admit`, `Gender_Male`, etc.
- Min–max normalize `gre` and `gpa` into \([0, 1]\):
  - \(\text{gre}' = (\text{gre} - \min(\text{gre}))/(\max(\text{gre}) - \min(\text{gre}))\)
  - \(\text{gpa}' = (\text{gpa} - \min(\text{gpa}))/(\max(\text{gpa}) - \min(\text{gpa}))\)

If `--excel-path` is omitted and `kagglehub` is installed, the script will:

- Call `kagglehub.dataset_download("eswarchandt/admission")`
- Use `Admission.xlsx` from the downloaded folder.

### 2.3 Building the 2D Point Sets

`build_point_sets_from_data(data)` constructs two point clouds:

- **Primary points** `x_train_new`:
  - 2D points \((\text{gre}, \text{gpa})\) in normalized coordinates.
- **Rotated points** `x_train_new_prime`:
  - \([\max(\text{gpa}) - \text{gpa}, \text{gre}]\)
  - This matches the rotation/reflection trick in the original notebook,
    to cover all directions in the dual space.
- If `admit` is present, a label vector `targets` is returned for coloring.

### 2.4 What the Experiment Does

In `main()` the script performs the following steps:

1. **Data loading & normalization**
   - Calls `load_college_admission_data` and prints dataset shape.

2. **Point-cloud construction**
   - Builds `x_train_new` and `x_train_new_prime` as above.

3. **Visualization of point clouds**
   - Calls `plot_point_sets(x_train_new, x_train_new_prime, targets)`:
     - Left subplot: \((\text{gre}, \text{gpa})\)
     - Right subplot: \([\max(\text{gpa}) - \text{gpa}, \text{gre}]\)
     - Points are colored by `admit` if available (admitted vs. not admitted),
       mirroring the original `Mining_U3M_Ray_Sweeping_2D_College_Admission.ipynb`.

4. **Ray-sweeping with naive intersection enumeration**
   - Converts the point sets to `List[Tuple[float, float]]`.
   - Calls:

     ```python
     run_ray_sweeping_naive_on_points(points_primary, top_k, min_angle_step)
     run_ray_sweeping_naive_on_points(points_rotated, top_k, min_angle_step)
     ```

   - Internally this uses:
     - `ray_sweeping_2d(..., use_incremental=False, use_randomized=False)`
     - Naive \(O(n^2)\) dual intersection enumeration.
   - Prints:
     - Runtime on primary and rotated point sets.
     - The top‑k directions and their skew values (via `SkewDirection`).

5. **Validation and tail visualization along a high‑skew direction**

   - Automatically takes the **top‑1 direction** from the primary run:

     ```python
     top_dir_vec = primary_dirs[0].direction.as_array()
     validate_and_visualize_tail(data, x_train_new, top_dir_vec)
     ```

   - `validate_and_visualize_tail(...)` does:
     - Normalize direction \(f\).
     - Project normalized points: \(p_f = x \cdot f\)
     - Compute:
       - \(\mu = \text{mean}(p_f)\), \(\tilde{m} = \text{median}(p_f)\),
         \(\sigma = \text{std}(p_f)\)
       - Skew: \((\mu - \tilde{m})/\sigma\)
     - Choose a tail percentile \(q=0.1\) (10% left or right tail depending
       on the sign of skew).
     - Extract the tail subset and visualize:
       - Points in the tail projected region, colored by `Gender_Male`
         (male vs. not male) or `admit` if available.
       - The direction line \(f\) and a shifted copy, drawn over the plot.
     - This matches the “tail of skew” visual analysis in the original notebook:
       **we identify an extreme direction and then inspect the subgroup of
       points that lie in the skewed tail along that direction**.

---

## 3. Chicago Crimes 2D Experiment

### 3.1 Script and Entry Point

Implementation file:

- `u3m_reimpl/experiment_ray_sweeping_2d_chicago_crimes.py`

Two ways to run:

1. Direct Python call (with optional path):

```bash
python u3m_reimpl/experiment_ray_sweeping_2d_chicago_crimes.py \
  --csv-path "Mining_U3Ms-main/Mining_U3Ms-main/2D/Chicago_Crimes_2012_to_2017.csv" \
  --top-k 10 \
  --min-angle-step 0.3141592653589793 \
  --n-samples 500
```

2. Convenience shell script:

```bash
cd u3m_reimpl
./crime.sh
```

If `--csv-path` is omitted and `kagglehub` is installed, the script will:

- Call `kagglehub.dataset_download("currie32/crimes-in-chicago")`
- Use `Chicago_Crimes_2012_to_2017.csv` from the downloaded folder.

### 3.2 Dataset Loading and Preprocessing

`load_chicago_crimes_data(...)` reproduces the preprocessing from
`Mining_U3M_Ray_Sweeping_2D_Chicago_Crimes.ipynb` relevant to the 2D experiment:

- Load the CSV into a DataFrame.
- Drop rows with missing values.
- Drop identifier columns `ID`, `Case Number` (if present).
- Parse `Date` into `"Year", "Month", "Day", "Hour", "Minute", "Second"`,
  then drop the original `Date` and intermediate `date2`.
- Drop `"Updated On"` if present.
- Factorize categorical columns:

  - `Block`, `IUCR`, `Description`,
  - `Location Description`, `FBI Code`, `Location`, `Primary Type`

- Filter out geographic outliers:

  - Keep only `Latitude > 40`.

- Min–max normalize:

  - `Longitude` and `Latitude` into \([0,1]\).

- Add convenience columns:

  - `Lat = Latitude`
  - `Lon = Longitude`
  - `Target = Arrest` (binary label used for coloring and validation).

### 3.3 Building the 2D Point Sets

`build_point_sets_from_data(data, n_samples=500)`:

- Randomly sample `n_samples` rows (default 500) to form the working set.
- Constructs:
  - **Primary points** `x_train_new`: \((\text{Lon}, \text{Lat})\)
  - **Rotated points** `x_train_new_prime`: \([\max(\text{Lat}) - \text{Lat}, \text{Lon}]\)
  - Optional `targets` from `Target` (Arrest) for coloring.

This mirrors the subsampling and rotation in the original Crimes notebook.

### 3.4 What the Experiment Does

In `main()` the steps are:

1. **Data loading & preprocessing**
   - Calls `load_chicago_crimes_data` and prints dataset shape.

2. **Point-cloud construction**
   - Builds `x_train_new`, `x_train_new_prime`, and `targets`.

3. **Visualization of point clouds**
   - Calls:

     ```python
     plot_point_sets(x_train_new, x_train_new_prime, targets)
     ```

   - Left subplot: `Lon` vs. `Lat`, colored by `Target` (arrested vs. not).
   - Right subplot: rotated coordinates \([\max(\text{Lat}) - \text{Lat}, \text{Lon}]\).

4. **Ray-sweeping with naive intersection enumeration**

   - Converts points to list-of-tuples and calls:

     ```python
     run_ray_sweeping_naive_on_points(points_primary, top_k, min_angle_step)
     run_ray_sweeping_naive_on_points(points_rotated, top_k, min_angle_step)
     ```

   - Again uses `ray_sweeping_2d(..., use_incremental=False, use_randomized=False)`.
   - Prints runtime and the top‑k high‑skew directions for both primary and
     rotated point sets.

5. **Validation and tail visualization along a high‑skew direction**

   - As in the College experiment, the script takes the **top‑1 primary
     direction** and calls:

     ```python
     top_dir_vec = primary_dirs[0].direction.as_array()
     validate_and_visualize_tail(data, x_train_new, top_dir_vec)
     ```

   - `validate_and_visualize_tail(...)` for Crimes:

     - Normalizes direction \(f\).
     - Computes projections \(p_f = x \cdot f\) on \((\text{Lon}, \text{Lat})\).
     - Computes mean, median, std, and skew \((\mu - \tilde{m})/\sigma\) and prints it.
     - Uses a **1% tail** (\(q = 0.01\)), as in the notebook:
       - Shifts `Lon`, `Lat` by subtracting their minima.
       - Chooses left or right tail based on the sign of skew.
     - Takes the tail subset and further randomly samples 10% for clarity.
     - Plots:
       - Tail points in \((\text{Lon}, \text{Lat})\) space,
         colored by `Target` (Arrested vs. Not Arrested).
       - The direction line \(f\) and a parallel shifted line.
       - A legend explaining the direction and the two label colors.

   - This reproduces the Crimes notebook’s “high‑skew direction” figure and
     the “tail of skew” figure, but driven entirely by your reimplemented
     Ray‑sweeping (not by the original `MaxSkewCalculator`).

---

## 4. Summary of What the Two Experiments Demonstrate

- **Shared core idea**:
  - For each real dataset, we embed each individual (or event) into a 2D
    space, run the Ray‑sweeping algorithm to find directions where the
    1D projection distribution is highly skewed, and then inspect the
    tail of that projection to see which subgroups are over‑represented
    among “extreme” points.

- **College Admission**:
  - 2D space: normalized `gre` vs. `gpa`.
  - Tail analysis often reveals directions where, for example, **male vs.
    not‑male applicants** or admitted vs. not‑admitted applicants concentrate
    on one side of the projection, corresponding to under‑ or over‑
    represented minority groups.

- **Chicago Crimes**:
  - 2D space: normalized `Longitude` vs. `Latitude`.
  - Tail analysis reveals geographic directions where **arrest vs. non‑arrest**
    events are unevenly distributed, highlighting spatial minority regions
    of high or low arrest likelihood.

In both experiments, the code you run:

1. Uses your reimplemented `ray_sweeping_2d` with naive O(n²) intersection enumeration.
2. Produces point‑cloud and tail visualizations analogous to the original
   notebooks.
3. Directly recomputes skew along the selected directions to **numerically
   validate** that the directions found by Ray‑sweeping are indeed
   high‑skew directions in the sense of Pearson’s median skewness.


