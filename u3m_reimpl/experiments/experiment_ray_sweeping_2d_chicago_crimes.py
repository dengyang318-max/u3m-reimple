from __future__ import annotations

"""
Experiment script: apply the 2D Ray-sweeping algorithm (naive enumeration)
to the Chicago Crimes dataset, following the 2D experimental procedure in
the paper “Mining the Minoria”.

This module mirrors, in a simplified and self-contained way, the official
notebook `Mining_U3M_Ray_Sweeping_2D_Chicago_Crimes.ipynb`:

- Load the Chicago Crimes dataset from a CSV file.
- Preprocess and normalize the longitude / latitude features.
- Build two 2D point clouds:
  (1) (Lon, Lat)  = (Longitude, Latitude)
  (2) a rotated / reflected version:
      [max_lat - Lat, Lon]
- Run the Ray-sweeping algorithm on each point set using the
  **naive O(n^2) intersection enumeration** (no incremental / randomized).
"""

import argparse
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import gaussian_kde

from u3m_reimpl.algorithms.ray_sweeping_2d import SkewDirection, ray_sweeping_2d
from u3m_reimpl.algorithms.ray_sweeping_2d_original import (
    SkewDirection as SkewDirectionOriginal,
    ray_sweeping_2d as ray_sweeping_2d_original,
)

# Official direction vector from the original notebook
# (used for comparison with the reimplemented algorithm)
OFFICIAL_F_DIRECTION_CRIMES = np.array([-0.253825600170482, 0.7461743998295179], dtype=float)


def _resolve_chicago_csv_path(csv_path: str | Path | None) -> Path:
    """
    Resolve the path to the Chicago Crimes CSV file.

    If csv_path is provided, use it directly. Otherwise, try to download
    the dataset using kagglehub, mimicking the official notebook:

        path = kagglehub.dataset_download("currie32/crimes-in-chicago")
        csv = path + "/Chicago_Crimes_2012_to_2017.csv"
    """

    if csv_path is not None:
        p = Path(csv_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Could not find dataset file at {p}. "
                "Please check the --csv-path argument."
            )
        return p

    # Fallback: use kagglehub to download the dataset, as in the notebook.
    try:
        import kagglehub  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "kagglehub is not installed, and no --csv-path was provided. "
            "Install kagglehub (`pip install kagglehub`) or pass --csv-path "
            "manually."
        ) from exc

    print("No --csv-path provided. Downloading Chicago Crimes dataset via kagglehub...")
    base_path = kagglehub.dataset_download("currie32/crimes-in-chicago")
    csv_file = Path(base_path) / "Chicago_Crimes_2012_to_2017.csv"
    if not csv_file.exists():
        raise FileNotFoundError(
            f"Downloaded dataset directory {base_path}, but could not find "
            f"`Chicago_Crimes_2012_to_2017.csv` inside it."
        )
    print(f"Using dataset file: {csv_file}")
    return csv_file


def load_chicago_crimes_data(csv_path: str | Path | None) -> pd.DataFrame:
    """
    Load and preprocess the Chicago Crimes dataset from a CSV file,
    following the steps in the official 2D notebook.

    The function focuses on the preprocessing steps that affect the
    (Longitude, Latitude) features used for the 2D Ray-sweeping experiment.

    Args:
        csv_path: Optional path to `Chicago_Crimes_2012_to_2017.csv` (or an
            equivalent CSV file with the same schema). If None, the function
            will attempt to download the dataset via kagglehub.

    Returns:
        A pandas DataFrame with normalized `Longitude` and `Latitude`
        columns in [0, 1], plus the original `Arrest` and derived features.
    """

    csv_path_resolved = _resolve_chicago_csv_path(csv_path)
    data = pd.read_csv(csv_path_resolved)

    # Drop rows with missing values, as in the notebook.
    data = data.dropna()

    # Drop a few identifier / bookkeeping columns if they exist.
    for col in ["ID", "Case Number"]:
        if col in data.columns:
            data = data.drop([col], axis=1)

    # Expand the Date column into several time features.
    if "Date" in data.columns:
        data["date2"] = pd.to_datetime(data["Date"], errors="coerce")
        data["Year"] = data["date2"].dt.year
        data["Month"] = data["date2"].dt.month
        data["Day"] = data["date2"].dt.day
        data["Hour"] = data["date2"].dt.hour
        data["Minute"] = data["date2"].dt.minute
        data["Second"] = data["date2"].dt.second
        data = data.drop(["Date", "date2"], axis=1)

        # Optional time-based filtering, matching the official code:
        # keep only records within a recent-year window (2015–2017) used in the paper.
        # This aligns with the official experiment's data preprocessing.
        data = data[(data["Year"] >= 2015) & (data["Year"] <= 2017)]

    # Drop the "Updated On" column if present.
    if "Updated On" in data.columns:
        data = data.drop(["Updated On"], axis=1)

    # Factorize several categorical columns used in the original notebook.
    for col in [
        "Block",
        "IUCR",
        "Description",
        "Location Description",
        "FBI Code",
        "Location",
        "Primary Type",
    ]:
        if col in data.columns:
            data[col] = pd.factorize(data[col])[0]

    # Remove outliers in latitude and normalize Lon/Lat into [0, 1].
    if "Latitude" not in data.columns or "Longitude" not in data.columns:
        raise ValueError("Expected `Latitude` and `Longitude` columns in the dataset.")

    data = data[data["Latitude"] > 40]
    lon = data["Longitude"].astype(float)
    lat = data["Latitude"].astype(float)
    data["Longitude"] = (lon - lon.min()) / (lon.max() - lon.min())
    data["Latitude"] = (lat - lat.min()) / (lat.max() - lat.min())

    # Add convenience copies used by the notebook.
    data["Lat"] = data["Latitude"]
    data["Lon"] = data["Longitude"]
    if "Arrest" in data.columns:
        data["Target"] = data["Arrest"]

    return data


def build_point_sets_from_data(
    data: pd.DataFrame, n_samples: int = 500
):
    """
    Construct the two 2D point sets used in the Chicago Crimes experiment:

    - x_train_new:       (Lon, Lat)
    - x_train_new_prime: [max_lat - Lat, Lon]

    We follow the notebook and subsample n_samples points for the
    Ray-sweeping step.
    """

    if not {"Lon", "Lat"}.issubset(set(data.columns)):
        raise ValueError("Data must contain `Lon` and `Lat` columns (normalized).")

    n_available = len(data)
    if n_available < n_samples:
        raise ValueError(
            f"Requested {n_samples} samples, but dataset only has {n_available} rows "
            "after preprocessing. Please decrease `n_samples`."
        )

    final_df = data.sample(n=n_samples, random_state=1)

    x_train_new = np.asarray(final_df[["Lon", "Lat"]], dtype=float)
    max_lat = float(np.max(x_train_new[:, 1]))
    x_train_new_prime = np.column_stack(
        (
            max_lat - x_train_new[:, 1],  # first coordinate
            x_train_new[:, 0],  # second coordinate
        )
    )

    # Optional target labels for visualization (if present).
    if "Target" in final_df.columns:
        targets = np.asarray(final_df["Target"], dtype=float)
    else:
        targets = None

    # Return the sampled dataframe as well so that downstream
    # statistics and model evaluation can be run on the same
    # subset instead of the full 1.4M rows.
    return x_train_new, x_train_new_prime, targets, final_df


def run_ray_sweeping_naive_on_points(
    points,
    top_k: int = 10,
    min_angle_step: float = np.pi / 10.0,
    vector_transfer=None,
):
    """
    Helper: run the 2D Ray-sweeping algorithm on a set of 2D points using
    only the naive O(n^2) intersection enumeration with dynamic median tracking.
    
    Args:
        points: Iterable of (x, y) coordinate pairs.
        top_k: Number of top high-skew directions to return.
        min_angle_step: Minimum angular step between sampled directions.
        vector_transfer: Optional direction transformation function, forwarded
            to `ray_sweeping_2d` so that rotated point sets can be mapped back
            to the original coordinate system (e.g. `lambda x: (-x[1], x[0])`).
    """

    start = time.perf_counter()
    results = ray_sweeping_2d(
        points,
        top_k=top_k,
        min_angle_step=min_angle_step,
        use_incremental=False,
        use_randomized=False,
        vector_transfer=vector_transfer,
    )
    elapsed = time.perf_counter() - start
    return results, elapsed


def plot_point_sets(
    x_train_new: np.ndarray,
    x_train_new_prime: np.ndarray,
    targets: np.ndarray | None = None,
) -> None:
    """
    Visualize the primary and rotated point sets, mimicking the notebook.
    """

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    if targets is not None:
        sc0 = axes[0].scatter(
            x_train_new[:, 0], x_train_new[:, 1], c=targets, s=10, cmap="viridis"
        )
        sc1 = axes[1].scatter(
            x_train_new_prime[:, 0],
            x_train_new_prime[:, 1],
            c=targets,
            s=10,
            cmap="viridis",
        )
        # Attach the colorbar to the right subplot and place it on the far right
        # so that it does not overlap the data region.
        # fig.colorbar(
        #     sc1,
        #     ax=axes[1],
        #     label="Target",
        #     fraction=0.046,
        #     pad=0.04,
        # )
    else:
        axes[0].scatter(x_train_new[:, 0], x_train_new[:, 1], s=10)
        axes[1].scatter(x_train_new_prime[:, 0], x_train_new_prime[:, 1], s=10)

    axes[0].set_title("Primary points (Lon, Lat)")
    axes[0].set_xlabel("Longitude (normalized)")
    axes[0].set_ylabel("Latitude (normalized)")

    axes[1].set_title("Rotated points [max_lat - Lat, Lon]")
    axes[1].set_xlabel("x'")
    axes[1].set_ylabel("y'")

    plt.tight_layout()
    plt.show()


def format_top_directions(directions: List[SkewDirection]) -> str:
    """
    Pretty-print helper for the top-k directions with their skew values.
    Returns a Markdown table format.
    """
    if not directions:
        return "No directions found."
    
    lines: List[str] = []
    lines.append("| Rank | Direction (x, y) | Skew |")
    lines.append("|------|------------------|------|")
    
    for i, cand in enumerate(directions, start=1):
        d = cand.direction.as_array()
        lines.append(
            f"| {i} | ({d[0]:.6f}, {d[1]:.6f}) | {cand.skew_value:.6f} |"
        )
    return "\n".join(lines)


def compare_versions(
    data,
    x_train_new,
    points_primary,
    points_rotated,
    targets,
    top_k: int,
    min_angle_step: float,
    use_official_f: bool = False,
    results_dir: Path | None = None,
):
    """Compare the original and updated versions of ray_sweeping_2d.
    
    This function runs both versions (original without min-shift/vector_transfer
    and updated with min-shift/vector_transfer) and compares their results.
    
    Args:
        data: Full dataframe (for tail statistics / plots)
        x_train_new: Primary point set as numpy array (for visualization)
        points_primary: Primary point set (original coordinates)
        points_rotated: Rotated point set
        targets: Optional label array for coloring (Target)
        top_k: Number of top directions to return
        min_angle_step: Minimum angular step between directions
        use_official_f: Whether to use the official f in visualization
    """
    print("\n" + "=" * 80)
    print("COMPARISON: Original vs Updated Implementation")
    print("=" * 80)
    
    # Run original version (no min-shift, no vector_transfer)
    print("\n[ORIGINAL VERSION] Running on primary point set...")
    start = time.perf_counter()
    primary_dirs_orig = ray_sweeping_2d_original(
        points_primary,
        top_k=top_k * 2,
        min_angle_step=min_angle_step,
        use_incremental=False,
        use_randomized=False,
    )
    primary_time_orig = time.perf_counter() - start
    print(f"  Time: {primary_time_orig:.4f} s, Found {len(primary_dirs_orig)} directions")
    
    print("[ORIGINAL VERSION] Running on rotated point set...")
    start = time.perf_counter()
    rotated_dirs_orig = ray_sweeping_2d_original(
        points_rotated,
        top_k=top_k * 2,
        min_angle_step=min_angle_step,
        use_incremental=False,
        use_randomized=False,
    )
    rotated_time_orig = time.perf_counter() - start
    print(f"  Time: {rotated_time_orig:.4f} s, Found {len(rotated_dirs_orig)} directions")
    
    # Merge original results (no vector_transfer, so directions are not mapped back)
    all_dirs_orig = primary_dirs_orig + rotated_dirs_orig
    all_dirs_orig_sorted = sorted(all_dirs_orig, key=lambda d: d.skew_value, reverse=True)
    merged_top_dirs_orig = all_dirs_orig_sorted[:top_k]
    
    # Run updated version (with min-shift and vector_transfer)
    print("\n[UPDATED VERSION] Running on primary point set...")
    start = time.perf_counter()
    primary_dirs_updated = ray_sweeping_2d(
        points_primary,
        top_k=top_k * 2,
        min_angle_step=min_angle_step,
        use_incremental=False,
        use_randomized=False,
        vector_transfer=lambda x: (x[0], x[1]),
    )
    primary_time_updated = time.perf_counter() - start
    print(f"  Time: {primary_time_updated:.4f} s, Found {len(primary_dirs_updated)} directions")
    
    print("[UPDATED VERSION] Running on rotated point set...")
    start = time.perf_counter()
    rotated_dirs_updated = ray_sweeping_2d(
        points_rotated,
        top_k=top_k * 2,
        min_angle_step=min_angle_step,
        use_incremental=False,
        use_randomized=False,
        vector_transfer=lambda x: (-x[1], x[0]),
    )
    rotated_time_updated = time.perf_counter() - start
    print(f"  Time: {rotated_time_updated:.4f} s, Found {len(rotated_dirs_updated)} directions")
    
    # Merge updated results (with vector_transfer, directions are mapped back)
    all_dirs_updated = primary_dirs_updated + rotated_dirs_updated
    all_dirs_updated_sorted = sorted(all_dirs_updated, key=lambda d: d.skew_value, reverse=True)
    merged_top_dirs_updated = all_dirs_updated_sorted[:top_k]
    
    # Print comparison
    print("\n" + "-" * 80)
    print("COMPARISON RESULTS:")
    print("-" * 80)
    
    print(f"\nOriginal Version (no min-shift, no vector_transfer):")
    print(f"  Best skew: {merged_top_dirs_orig[0].skew_value:.6f}")
    d_orig = merged_top_dirs_orig[0].direction.as_array()
    print(f"  Direction: ({d_orig[0]:.6f}, {d_orig[1]:.6f})")
    print(f"  Total time: {primary_time_orig + rotated_time_orig:.4f} s")
    
    print(f"\nUpdated Version (with min-shift and vector_transfer):")
    print(f"  Best skew: {merged_top_dirs_updated[0].skew_value:.6f}")
    d_updated = merged_top_dirs_updated[0].direction.as_array()
    print(f"  Direction: ({d_updated[0]:.6f}, {d_updated[1]:.6f})")
    print(f"  Total time: {primary_time_updated + rotated_time_updated:.4f} s")
    
    print(f"\nDifferences:")
    print(f"  Skew difference: {abs(merged_top_dirs_updated[0].skew_value - merged_top_dirs_orig[0].skew_value):.6f}")
    dir_diff = np.linalg.norm(d_updated - d_orig)
    print(f"  Direction difference (L2 norm): {dir_diff:.6f}")
    time_diff = (primary_time_updated + rotated_time_updated) - (primary_time_orig + rotated_time_orig)
    print(f"  Time difference: {time_diff:+.4f} s")
    
    print("\nTop-3 directions comparison:")
    print("\n**Original Version:**")
    print("| Rank | Direction (x, y) | Skew |")
    print("|------|------------------|------|")
    for i, d in enumerate(merged_top_dirs_orig[:3], 1):
        dir_arr = d.direction.as_array()
        print(f"| {i} | ({dir_arr[0]:.6f}, {dir_arr[1]:.6f}) | {d.skew_value:.6f} |")
    
    print("\n**Updated Version:**")
    print("| Rank | Direction (x, y) | Skew |")
    print("|------|------------------|------|")
    for i, d in enumerate(merged_top_dirs_updated[:3], 1):
        dir_arr = d.direction.as_array()
        print(f"| {i} | ({dir_arr[0]:.6f}, {dir_arr[1]:.6f}) | {d.skew_value:.6f} |")
    
    print("=" * 80 + "\n")
    
    # Visualize top-3 directions for both versions on the primary point set
    print("\nVisualizing top-3 directions for ORIGINAL VERSION (on primary set)...")
    max_dirs_to_visualize_orig = min(3, len(merged_top_dirs_orig))
    for rank in range(max_dirs_to_visualize_orig):
        top_dir_vec_orig = merged_top_dirs_orig[rank].direction.as_array()
        print(f"\n[Original] Visualizing tail along top-{rank + 1} direction...")
        validate_and_visualize_tail(
            data,
            x_train_new,
            top_dir_vec_orig,
            targets,
            use_official_f=use_official_f,
            save_prefix=f"original_top{rank+1}",
            results_dir=results_dir,
        )

    print("\nVisualizing top-3 directions for UPDATED VERSION (on primary set)...")
    max_dirs_to_visualize_updated = min(3, len(merged_top_dirs_updated))
    for rank in range(max_dirs_to_visualize_updated):
        top_dir_vec_updated = merged_top_dirs_updated[rank].direction.as_array()
        print(f"\n[Updated] Visualizing tail along top-{rank + 1} direction...")
        validate_and_visualize_tail(
            data,
            x_train_new,
            top_dir_vec_updated,
            targets,
            use_official_f=use_official_f,
            save_prefix=f"updated_top{rank+1}",
            results_dir=results_dir,
        )

    return merged_top_dirs_orig, merged_top_dirs_updated


def print_experiment_summary(
    dataset_name: str,
    n_points: int,
    primary_dirs: List[SkewDirection],
    primary_time: float,
    rotated_dirs: List[SkewDirection] | None = None,
    rotated_time: float | None = None,
    merged_dirs: List[SkewDirection] | None = None,
):
    """
    Print a comprehensive summary of the experiment results, including:
    - Dataset information
    - Performance metrics (runtime)
    - Top directions (merged from primary and rotated sets)
    """
    print("\n" + "=" * 80)
    print(f"EXPERIMENT SUMMARY: {dataset_name}")
    print("=" * 80)
    
    print(f"\nDataset Information:")
    print(f"  - Number of points: {n_points}")
    
    print(f"\nPerformance Metrics:")
    print(f"  - Primary point set:          {primary_time:.4f} s")
    if rotated_time is not None:
        print(f"  - Rotated point set:          {rotated_time:.4f} s")
        print(f"  - Total time:                 {primary_time + rotated_time:.4f} s")
    
    # Show merged results as the main output (matching official behavior)
    if merged_dirs:
        print(f"\n**Top Directions (Merged from Primary + Rotated):**")
        print(format_top_directions(merged_dirs))
    else:
        # Fallback to primary if merged not available
        print(f"\n**Top Directions:**")
        if primary_dirs:
            print(format_top_directions(primary_dirs))
    
    # Optionally show separate results for comparison
    if primary_dirs and rotated_dirs and len(primary_dirs) > 0 and len(rotated_dirs) > 0:
        print(f"\nSeparate Results (for comparison):")
        print(f"  - Primary best skew:   {primary_dirs[0].skew_value:.6f}")
        print(f"  - Rotated best skew:   {rotated_dirs[0].skew_value:.6f}")
    
    print("=" * 80 + "\n")


def validate_and_visualize_tail(
    data,
    x_train_new,
    direction_vec,
    targets_sample=None,
    use_official_f=False,
    save_prefix: str | None = None,
    results_dir: Path | None = None,
):
    """
    Validate skew value along a chosen direction and visualize the tail
    region, mimicking the original notebook.
    """

    if x_train_new.size == 0:
        return

    # Choose direction: use official f_direction if requested, otherwise use the provided direction_vec
    if use_official_f:
        f = OFFICIAL_F_DIRECTION_CRIMES.copy()
        print("\nUsing official f_direction from the original notebook for visualization.")
    else:
        f = np.asarray(direction_vec, dtype=float)
        f = f / np.linalg.norm(f)
        print("\nUsing reimplemented top direction for visualization.")

    # Project normalized points and compute skew directly.
    p_f = x_train_new @ f
    mean = float(np.mean(p_f))
    sd = float(np.std(p_f))
    median = float(np.median(p_f))
    if sd > 0.0:
        sk = (mean - median) / sd
    else:
        sk = 0.0
    print(f"\nValidation: skew along chosen direction f is {sk:.6f}")

    # ------------------------------------------------------------------
    # 1) Visualize the full point cloud with direction f overlaid
    # ------------------------------------------------------------------
    full_points = x_train_new
    fig_full, ax_full = plt.subplots(figsize=(5, 5))

    # Color by Target (Arrest) if a matching sample is provided, otherwise single color.
    color_handles: list[Line2D] = []
    if targets_sample is not None:
        if targets_sample.shape[0] != full_points.shape[0]:
            raise ValueError(
                "targets_sample length does not match number of sampled points."
            )
        colors_full = ["#1f77b4" if x == 1 else "#2ca02c" for x in targets_sample]
        ax_full.scatter(full_points[:, 0], full_points[:, 1], c=colors_full, s=10)
        color_handles = [
            Line2D([0], [0], marker="o", color="w", label="Arrested", markerfacecolor="#1f77b4", markersize=10),
            Line2D([0], [0], marker="o", color="w", label="Not Arrested", markerfacecolor="#2ca02c", markersize=10),
        ]
    else:
        ax_full.scatter(full_points[:, 0], full_points[:, 1], s=10, color="#1f77b4")

    xmin_f, xmax_f = full_points[:, 0].min(), full_points[:, 0].max()
    ymin_f, ymax_f = full_points[:, 1].min(), full_points[:, 1].max()
    x0_f, y0_f = full_points[:, 0].mean(), full_points[:, 1].mean()
    
    # Mark tail region (orange shading) - matching Figure 4(a) in paper
    # Project all points to find tail threshold
    proj_full = full_points @ f
    q_tail_viz = 0.01  # 1% tail for visualization
    thresh_tail_viz = np.quantile(proj_full, q_tail_viz if -sk < 0 else (1.0 - q_tail_viz))
    
    # Create a polygon for the tail region (triangular region in top-left corner)
    # The tail region is bounded by the direction line and the axes
    if -sk < 0:  # Negative skew: tail is on the left side
        # Points with projection < threshold are in the tail
        tail_mask_viz = proj_full < thresh_tail_viz
        # The tail region is the triangular area in the top-left
        # We'll create a polygon from (0, ymax) -> intersection with line -> (xmin, 0)
        # For simplicity, we'll shade the region where points are in the tail
        tail_points_viz = full_points[tail_mask_viz]
        if len(tail_points_viz) > 0:
            # Create a convex hull or bounding box for the tail region
            from scipy.spatial import ConvexHull
            try:
                if len(tail_points_viz) >= 3:
                    hull = ConvexHull(tail_points_viz)
                    tail_polygon = tail_points_viz[hull.vertices]
                    # Fill the polygon with orange color
                    from matplotlib.patches import Polygon
                    poly = Polygon(tail_polygon, alpha=0.3, color='orange', label='Tail region')
                    ax_full.add_patch(poly)
            except:
                # Fallback: just highlight tail points with different color
                pass
    else:  # Positive skew: tail is on the right side
        tail_mask_viz = proj_full > thresh_tail_viz
        tail_points_viz = full_points[tail_mask_viz]
        if len(tail_points_viz) > 0:
            from scipy.spatial import ConvexHull
            try:
                if len(tail_points_viz) >= 3:
                    hull = ConvexHull(tail_points_viz)
                    tail_polygon = tail_points_viz[hull.vertices]
                    from matplotlib.patches import Polygon
                    poly = Polygon(tail_polygon, alpha=0.3, color='orange', label='Tail region')
                    ax_full.add_patch(poly)
            except:
                pass
    
    # Calculate t range to ensure the line covers the entire plot area
    # For a line: x = x0 + t*f[0], y = y0 + t*f[1]
    # We need to find t_min and t_max such that the line spans from (xmin, ymin) to (xmax, ymax)
    t_candidates = []
    if abs(f[0]) > 1e-10:  # Avoid division by zero
        t_candidates.append((xmin_f - x0_f) / f[0])
        t_candidates.append((xmax_f - x0_f) / f[0])
    if abs(f[1]) > 1e-10:
        t_candidates.append((ymin_f - y0_f) / f[1])
        t_candidates.append((ymax_f - y0_f) / f[1])
    
    if t_candidates:
        t_min = min(t_candidates)
        t_max = max(t_candidates)
        # Extend slightly beyond to ensure full coverage
        t_range = t_max - t_min
        t_min -= t_range * 0.1
        t_max += t_range * 0.1
    else:
        # Fallback if f is degenerate
        t_min, t_max = -1, 1
    
    t_vals_f = np.linspace(t_min, t_max, num=200)
    x_vals_f = x0_f + t_vals_f * f[0]
    y_vals_f = y0_f + t_vals_f * f[1]
    ax_full.plot(x_vals_f, y_vals_f, color="red", linewidth=1)

    dir_handle_full = Line2D([0], [0], label="Direction f", color="red", linewidth=1)
    if color_handles:
        ax_full.legend(
            handles=[dir_handle_full] + color_handles,
            fontsize=11,
            loc="upper right",
        )
    else:
        ax_full.legend(
            handles=[dir_handle_full],
            fontsize=11,
            loc="upper right",
        )

    ax_full.set_xlim(xmin_f, xmax_f)
    ax_full.set_ylim(ymin_f, ymax_f)
    ax_full.set_aspect("equal", adjustable="box")
    # Set ticks from 0 with 0.2 interval, ensuring they cover the data range
    xmax_ticks = np.ceil(xmax_f / 0.2) * 0.2
    ymax_ticks = np.ceil(ymax_f / 0.2) * 0.2
    ax_full.set_xticks(np.arange(0, xmax_ticks + 0.2, 0.2))
    ax_full.set_yticks(np.arange(0, ymax_ticks + 0.2, 0.2))
    ax_full.set_xlabel("Normalized Longitude", fontsize=13)
    ax_full.set_ylabel("Normalized Latitude", fontsize=13)
    ax_full.set_title("Full data along high-skew direction", fontsize=13)
    
    plt.tight_layout()

    # Optionally save full-data figure and record into markdown
    if save_prefix is not None:
        base_dir = results_dir or (Path("results") / "chicago_crimes")
        base_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{save_prefix}_full.png"
        fig_full.savefig(base_dir / fname, dpi=200)
        # Markdown image link (MD file is in the same directory)
        print(f"![{save_prefix} full]({fname})")

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # 2) Tail visualization: select points in extreme q-quantile along f.
    # ------------------------------------------------------------------
    q = 0.01  # 1% tail, as in the notebook

    df = data.copy(deep=True)
    if not {"Lon", "Lat"}.issubset(df.columns):
        return

    # Shifted coordinates, following the notebook.
    df["Lat"] = df["Lat"] - df["Lat"].min()
    df["Lon"] = df["Lon"] - df["Lon"].min()

    coords = df[["Lon", "Lat"]].to_numpy(dtype=float)
    proj = coords @ f
    thresh = np.quantile(proj, q if -sk < 0 else (1.0 - q))
    if -sk < 0:
        mask = proj < thresh
    else:
        mask = proj > thresh
    tail = df.loc[mask]
    print(f"Tail subset shape (q={q}): {tail.shape}")

    if tail.empty:
        return

    # Keep a copy of the tail for statistics (before any visualization
    # subsampling), so that evaluation is run on the full tail subset.
    tail_stats = tail.copy(deep=True)

    # Optionally subsample for clearer plot (as in the notebook).
    # Only subsample when the tail is large; for very small tails we keep
    # all points to avoid ending up with an empty set.
    if len(tail) > 10000:
        tail = tail.sample(frac=0.1, random_state=1)

    points_np = tail[["Lon", "Lat"]].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(5, 5))

    # Color by Target (Arrest) if available, otherwise single color.
    legend_handles: list[Line2D] = []
    if "Target" in tail.columns:
        labels = tail["Target"].to_numpy()
        colors = ["#1f77b4" if x == 1 else "#2ca02c" for x in labels]
        ax.scatter(points_np[:, 0], points_np[:, 1], c=colors, s=10)
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", label="Arrested", markerfacecolor="#1f77b4", markersize=10),
            Line2D([0], [0], marker="o", color="w", label="Not Arrested", markerfacecolor="#2ca02c", markersize=10),
        ]
    else:
        ax.scatter(points_np[:, 0], points_np[:, 1], s=10, color="#1f77b4")

    # Draw the direction line using a parametric form centered on the full data.
    # IMPORTANT: Use the same reference point (full_points center) as in the full data plot
    # to ensure the line appears at the same position (same intercept) in both plots.
    # The tail plot uses shifted coordinates (Lon - min, Lat - min), so we need to
    # convert the full_points center to the shifted coordinate system.
    xmin, xmax = points_np[:, 0].min(), points_np[:, 0].max()
    ymin, ymax = points_np[:, 1].min(), points_np[:, 1].max()
    ymax_tail = ymax + 0.01  # Define ymax_tail once for use in both line calculation and ylim
    
    # Convert full_points center from original normalized coordinates to shifted coordinates
    # (matching the tail plot's coordinate system where df["Lon"] = data["Lon"] - data["Lon"].min())
    # Note: Since df["Lon"] = data["Lon"] - data["Lon"].min(), and data["Lon"] is already normalized to [0,1],
    # the shift is just data["Lon"].min(), which is 0.0 for normalized data. However, we calculate it
    # to be safe in case the data has been modified.
    lon_min_original = data["Lon"].min() if "Lon" in data.columns else 0.0
    lat_min_original = data["Lat"].min() if "Lat" in data.columns else 0.0
    full_center_x_shifted = x0_f - lon_min_original
    full_center_y_shifted = y0_f - lat_min_original
    
    # Use the full_points center (in shifted coordinates) as the reference point for the line
    x0, y0 = full_center_x_shifted, full_center_y_shifted
    
    # Calculate t range to ensure the line covers the entire plot area
    # For a line: x = x0 + t*f[0], y = y0 + t*f[1]
    # We need to find t_min and t_max such that the line spans from (xmin, ymin) to (xmax, ymax)
    t_candidates = []
    if abs(f[0]) > 1e-10:  # Avoid division by zero
        t_candidates.append((xmin - x0) / f[0])
        t_candidates.append((xmax - x0) / f[0])
    if abs(f[1]) > 1e-10:
        t_candidates.append((ymin - y0) / f[1])
        t_candidates.append((ymax_tail - y0) / f[1])
    
    if t_candidates:
        t_min = min(t_candidates)
        t_max = max(t_candidates)
        # Extend slightly beyond to ensure full coverage
        t_range = t_max - t_min
        t_min -= t_range * 0.1
        t_max += t_range * 0.1
    else:
        # Fallback if f is degenerate
        t_min, t_max = -1, 1
    
    t_vals = np.linspace(t_min, t_max, num=200)
    x_vals = x0 + t_vals * f[0]
    y_vals = y0 + t_vals * f[1]
    ax.plot(x_vals, y_vals, color="red", linewidth=1)

    dir_handle = Line2D([0], [0], label="Direction f", color="red", linewidth=1)
    if legend_handles:
        ax.legend(
            handles=[dir_handle] + legend_handles,
            fontsize=11,
            loc="upper right",
        )
    else:
        ax.legend(
            handles=[dir_handle],
            fontsize=11,
            loc="upper right",
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(points_np[:, 1].min(), ymax_tail)
    ax.set_aspect("equal", adjustable="box")
    # Set ticks from 0 with 0.2 interval, ensuring they cover the data range
    xmax_ticks = np.ceil(xmax / 0.2) * 0.2
    ymax_ticks = np.ceil(ymax_tail / 0.2) * 0.2
    ax.set_xticks(np.arange(0, xmax_ticks + 0.2, 0.2))
    ax.set_yticks(np.arange(0, ymax_ticks + 0.2, 0.2))
    ax.set_xlabel("Normalized Longitude", fontsize=13)
    ax.set_ylabel("Normalized Latitude", fontsize=13)
    ax.set_title("Tail subset along high-skew direction", fontsize=13)
    
    plt.tight_layout()

    # Optionally save tail figure and record into markdown
    if save_prefix is not None:
        base_dir = results_dir or (Path("results") / "chicago_crimes")
        base_dir.mkdir(parents=True, exist_ok=True)
        fname_tail = f"{save_prefix}_tail.png"
        fig.savefig(base_dir / fname_tail, dpi=200)
        print(f"![{save_prefix} tail]({fname_tail})")

    plt.tight_layout()
    plt.show()
    
    # ------------------------------------------------------------------
    # 5) Projection density plot (matching Figure 4(b) in paper)
    # ------------------------------------------------------------------
    # Project all points onto the direction f
    coords_all = df[["Lon", "Lat"]].to_numpy(dtype=float)
    proj_all = coords_all @ f
    
    # Create density plot
    fig_density, ax_density = plt.subplots(figsize=(8, 5))
    
    # Histogram
    n_bins = 50
    counts, bins, patches = ax_density.hist(proj_all, bins=n_bins, density=True, alpha=0.6, 
                                             color='lightblue', label='Histogram', edgecolor='black')
    
    # KDE (Kernel Density Estimation)
    if len(proj_all) > 1:
        kde = gaussian_kde(proj_all)
        x_kde = np.linspace(proj_all.min(), proj_all.max(), 200)
        density_kde = kde(x_kde)
        ax_density.plot(x_kde, density_kde, 'b-', linewidth=2, label='Density (KDE)')
    
    # Mark tail region (orange)
    q_tail = 0.01
    thresh_tail = np.quantile(proj_all, q_tail if -sk < 0 else (1.0 - q_tail))
    if -sk < 0:
        tail_mask = proj_all < thresh_tail
        tail_start = proj_all.min()
        tail_end = thresh_tail
    else:
        tail_mask = proj_all > thresh_tail
        tail_start = thresh_tail
        tail_end = proj_all.max()
    
    # Shade tail region
    ax_density.axvspan(tail_start, tail_end, alpha=0.3, color='orange', label='Tail')
    
    ax_density.set_xlabel('High Skew Direction', fontsize=13)
    ax_density.set_ylabel('Density', fontsize=13)
    ax_density.set_title('First high-skew direction. After projecting all the points.', fontsize=13)
    ax_density.legend(fontsize=11)
    ax_density.grid(True, alpha=0.3)
    
    plt.tight_layout()

    # Optionally save density figure and record into markdown
    if save_prefix is not None:
        base_dir = results_dir or (Path("results") / "chicago_crimes")
        base_dir.mkdir(parents=True, exist_ok=True)
        fname_density = f"{save_prefix}_density.png"
        fig_density.savefig(base_dir / fname_density, dpi=200)
        print(f"![{save_prefix} density]({fname_density})")
    
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # 6) Global vs tail statistics (label proportion + model accuracy)
    #    using a simple baseline model, matching the original implementation.
    # ------------------------------------------------------------------
    if "Target" in data.columns:
        y_all = data["Target"].to_numpy(dtype=int)
        y_tail = tail_stats["Target"].to_numpy(dtype=int)

        pos_rate_all = float(y_all.mean())
        pos_rate_tail = float(y_tail.mean()) if len(y_tail) > 0 else float("nan")

        feature_cols = [
            "Month",
            "Hour",
            "Latitude",
            "Longitude",
            "Primary Type",
            "IUCR",
            "FBI Code",
        ]
        if all(col in data.columns for col in feature_cols):
            X_all = data[feature_cols].to_numpy(dtype=float)
            X_tail = tail_stats[feature_cols].to_numpy(dtype=float)

            # Simple Logistic Regression baseline, as in the original
            # global vs tail comparison implementation.
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_all, y_all)

            y_pred_all = clf.predict(X_all)
            y_pred_tail = clf.predict(X_tail)

            acc_all = float(accuracy_score(y_all, y_pred_all))
            acc_tail = float(accuracy_score(y_tail, y_pred_tail))
            f1_all = float(f1_score(y_all, y_pred_all))
            f1_tail = float(f1_score(y_tail, y_pred_tail))
        else:
            acc_all = float("nan")
            acc_tail = float("nan")
            f1_all = float("nan")
            f1_tail = float("nan")

        print("\n=== Global vs Tail statistics (Chicago Crimes) ===")
        # Markdown table for global vs tail comparison
        print("| Metric                             | Global | Tail |")
        print("|------------------------------------|:------:|:----:|")
        print(f"| Positive rate (Arrest=1)          | {pos_rate_all:.3f} | {pos_rate_tail:.3f} |")
        if not np.isnan(acc_all):
            print(f"| Accuracy (Logistic Regression)    | {acc_all:.3f} | {acc_tail:.3f} |")
            print(f"| F1 (Logistic Regression)          | {f1_all:.3f} | {f1_tail:.3f} |")

            # --------------------------------------------------------------
            # Multi-percentile evaluation using full dataset
            # 使用全部数据，不进行采样
            # Percentiles: [1, 0.1, 0.01, 0.001, 0.0001]
            # --------------------------------------------------------------
            print("\n=== Multi-Percentile Evaluation Table (Chicago Crimes, full dataset) ===")
            percentiles = [1.0, 0.1, 0.01, 0.001, 0.0001]

            # 使用全部数据进行 percentile 评估
            # 准备用于投影的坐标（与上面 df 的处理方式保持一致：Lon/Lat 做 min-shift）
            coords_full = df[["Lon", "Lat"]].to_numpy(dtype=float)
            proj_full = coords_full @ f

            # 使用已经在全局数据上训练好的 clf，在不同 percentile 的 tail 上评估
            print("| Percentile | PosRate_tail | Accuracy | F1-score |")
            print("|-----------:|-------------:|---------:|---------:|")

            for q in percentiles:
                # 根据 skew 的方向决定取左尾还是右尾
                thresh_q = np.quantile(proj_full, q if -sk < 0 else (1.0 - q))
                if -sk < 0:
                    mask_q = proj_full < thresh_q
                else:
                    mask_q = proj_full > thresh_q

                tail_q = data.loc[mask_q]

                if len(tail_q) > 0:
                    y_tail_q = tail_q["Target"].to_numpy(dtype=int)
                    X_tail_q = tail_q[feature_cols].to_numpy(dtype=float) if all(
                        col in tail_q.columns for col in feature_cols
                    ) else None

                    pos_rate_tail_q = float(y_tail_q.mean())

                    if X_tail_q is not None and len(X_tail_q) > 0:
                        y_pred_tail_q = clf.predict(X_tail_q)
                        acc_q = float(accuracy_score(y_tail_q, y_pred_tail_q))
                        f1_q = float(f1_score(y_tail_q, y_pred_tail_q))
                        print(
                            f"| {q:>10.5f} | {pos_rate_tail_q:>11.3f} | "
                            f"{acc_q:>8.3f} | {f1_q:>8.3f} |"
                        )
                    else:
                        print(f"| {q:>10.5f} | {pos_rate_tail_q:>11.3f} | {'N/A':>8} | {'N/A':>8} |")
                else:
                    print(f"| {q:>10.5f} | {'N/A':>11} | {'N/A':>8} | {'N/A':>8} |")


def main(argv=None):
    """
    Command-line entry point for reproducing the 2D experiment on
    the Chicago Crimes dataset using the naive enumeration method.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run the 2D Ray-sweeping algorithm (naive enumeration) on the "
            "Chicago Crimes dataset, following the 2D experimental "
            "procedure from the paper."
        )
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help=(
            "Optional path to the Chicago Crimes CSV file "
            "(e.g. Chicago_Crimes_2012_to_2017.csv). "
            "If omitted, the script will try to download the dataset via kagglehub."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top directions to report (default: 10).",
    )
    parser.add_argument(
        "--min-angle-step",
        type=float,
        default=float(np.pi / 10.0),
        help=(
            "Minimum angular step between sampled directions in radians "
            "(default: pi/10)."
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help=(
            "Number of points to sample from the preprocessed dataset "
            "for the 2D experiment (default: 500, matching the paper)."
        ),
    )
    parser.add_argument(
        "--use-official-f",
        action="store_true",
        help=(
            "Use the official f_direction from the original notebook for "
            "visualization instead of the reimplemented top direction."
        ),
    )
    parser.add_argument(
        "--compare-versions",
        action="store_true",
        help=(
            "Compare the original version (no min-shift, no vector_transfer) "
            "with the updated version (with min-shift and vector_transfer). "
            "This helps evaluate the impact of these modifications."
        ),
    )

    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Set up per-run directory and Markdown logging
    # ------------------------------------------------------------------
    results_root = Path("results") / "chicago_crimes"
    results_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_root / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "chicago_crimes_log.md"
    log_file = log_path.open("w", encoding="utf-8")

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)

        def flush(self):
            for s in self.streams:
                s.flush()

    sys.stdout = Tee(sys.__stdout__, log_file)
    print(f"# Chicago Crimes Ray-Sweeping Experiment ({ts})\n")

    print("Loading Chicago Crimes dataset...")
    data = load_chicago_crimes_data(args.csv_path)
    print(f"Loaded dataset with shape: {data.shape}")

    x_train_new, x_train_new_prime, targets, sampled_df = build_point_sets_from_data(
        data, n_samples=args.n_samples
    )
    print(f"Primary point set shape: {x_train_new.shape}")
    print(f"Rotated point set shape: {x_train_new_prime.shape}")

    # Visualization of the 2D point sets, following the original notebook.
    print("\nVisualizing primary and rotated point sets...")
    plot_point_sets(x_train_new, x_train_new_prime, targets)

    # Convert to list-of-tuples for the ray_sweeping_2d interface.
    points_primary = [tuple(row) for row in x_train_new]
    points_rotated = [tuple(row) for row in x_train_new_prime]

    # If comparison is requested, run both versions and compare
    if args.compare_versions:
        compare_versions(
            data,
            x_train_new,
            points_primary,
            points_rotated,
            targets,
            top_k=args.top_k,
            min_angle_step=args.min_angle_step,
            use_official_f=args.use_official_f,
            results_dir=run_dir,
        )
        # 比较模式下已完成原始 vs 更新实现的完整对比和可视化，
        # 为避免重复实验，这里直接结束。
        return

    print("\nRunning Ray-sweeping (naive enumeration) on primary point set...")
    # Get more candidates from primary set to ensure good coverage after merging
    primary_dirs_all, primary_time = run_ray_sweeping_naive_on_points(
        points_primary,
        top_k=args.top_k * 2,  # Get more candidates for merging
        min_angle_step=args.min_angle_step,
        vector_transfer=lambda x: (x[0], x[1]),  # identity mapping for primary set
    )
    print(f"Time (primary): {primary_time:.4f} s")
    print(f"Found {len(primary_dirs_all)} directions from primary set")

    print("\nRunning Ray-sweeping (naive enumeration) on rotated point set...")
    # Get more candidates from rotated set to ensure good coverage after merging
    rotated_dirs_all, rotated_time = run_ray_sweeping_naive_on_points(
        points_rotated,
        top_k=args.top_k * 2,  # Get more candidates for merging
        min_angle_step=args.min_angle_step,
        vector_transfer=lambda x: (-x[1], x[0]),  # 90-degree rotation for rotated set
    )
    print(f"Time (rotated): {rotated_time:.4f} s")
    print(f"Found {len(rotated_dirs_all)} directions from rotated set")

    # Merge results from both point sets and select top-k
    print(f"\nMerging results from primary and rotated point sets...")
    all_dirs = primary_dirs_all + rotated_dirs_all
    # Sort by skew value (descending) and take top-k
    all_dirs_sorted = sorted(all_dirs, key=lambda d: d.skew_value, reverse=True)
    merged_top_dirs = all_dirs_sorted[:args.top_k]
    
    print(f"Merged {len(all_dirs)} total directions, selected top-{len(merged_top_dirs)}:")
    print(format_top_directions(merged_top_dirs))

    # Choose directions for validation / tail visualization using merged results
    if merged_top_dirs:
        # Visualize tails for the top-3 directions from merged results
        max_dirs_to_visualize = min(3, len(merged_top_dirs))
        for rank in range(max_dirs_to_visualize):
            top_dir_vec = merged_top_dirs[rank].direction.as_array()
            print(f"\nVisualizing tail along top-{rank + 1} merged direction...")
            validate_and_visualize_tail(
                data,
                x_train_new,
                top_dir_vec,
                targets,
                use_official_f=args.use_official_f,
            )

    # Print comprehensive experiment summary
    # Keep separate results for summary, but use merged results as the main output
    print_experiment_summary(
        dataset_name="Chicago Crimes",
        n_points=len(x_train_new),
        primary_dirs=primary_dirs_all[:args.top_k] if primary_dirs_all else [],
        primary_time=primary_time,
        rotated_dirs=rotated_dirs_all[:args.top_k] if rotated_dirs_all else [],
        rotated_time=rotated_time,
        merged_dirs=merged_top_dirs,  # Add merged results to summary
    )


if __name__ == "__main__":
    main()



