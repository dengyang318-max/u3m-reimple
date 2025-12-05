from __future__ import annotations

"""
Experiment script: apply the 2D Ray-sweeping algorithm (official-style
implementation) to the Chicago Crimes dataset, following the same 2D
experimental procedure as `experiment_ray_sweeping_2d_chicago_crimes.py`.

The only difference is that here we use the implementation that matches
the original official code (`ray_sweeping_2d_official_style.py`).
"""

import argparse
import time
from pathlib import Path

from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import gaussian_kde

from u3m_reimpl.algorithms.ray_sweeping_2d_official_linkedlist import (
    SkewDirection,
    ray_sweeping_2d_official_linkedlist,
)


# Official direction vector from the original notebook
# (used for comparison with the reimplemented algorithm)
OFFICIAL_F_DIRECTION_CRIMES = np.array(
    [-0.253825600170482, 0.7461743998295179], dtype=float
)


def _resolve_chicago_csv_path(csv_path):
    """
    Resolve the path to the Chicago Crimes CSV file (same as reimplementation script).
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


def load_chicago_crimes_data(csv_path):
    """
    Load and preprocess the Chicago Crimes dataset from a CSV file.

    This is copied from `experiment_ray_sweeping_2d_chicago_crimes.py`
    so that both experiments operate on exactly the same processed data.
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

        # Optional time-based filtering, matching the intention of the official code:
        # keep only records within a recent-year window (2015–2017) used in the paper.
        # This slightly reduces noise from very early years while following the
        # same preprocessing spirit as the original notebook.
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


def build_point_sets_from_data(data, n_samples=500):
    """
    Construct the two 2D point sets used in the Chicago Crimes experiment:

    - x_train_new:       (Lon, Lat)
    - x_train_new_prime: [max_lat - Lat, Lon]

    Exactly the same as in the reimplementation experiment script, so
    that the only change between experiments is the Ray-sweeping method.
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

    # Return the sampled dataframe as well so that downstream statistics
    # and model evaluation can be run on the same subset.
    return x_train_new, x_train_new_prime, targets, final_df


def run_ray_sweeping_official_on_points(
    points, top_k=10, min_angle_step=np.pi / 10.0, vector_transfer=None
):
    """
    Helper: run the 2D Ray-sweeping algorithm using the official-style
    implementation on a set of 2D points.
    """

    start = time.perf_counter()
    results = ray_sweeping_2d_official_linkedlist(
        points,
        top_k=top_k,
        epsilon=min_angle_step,
        vector_transfer=vector_transfer,
    )
    elapsed = time.perf_counter() - start
    return results, elapsed


def plot_point_sets(
    x_train_new,
    x_train_new_prime,
    targets=None,
):
    """
    Visualize the primary and rotated point sets, same as in the
    reimplementation experiment script.
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
    """

    lines: List[str] = []
    for i, cand in enumerate(directions, start=1):
        # In the linked-list official implementation, `direction` is a numpy array,
        # not a Direction2D object, so we use it directly.
        d = np.asarray(cand.direction, dtype=float)
        lines.append(
            f"#{i:02d}: dir=({d[0]: .6f}, {d[1]: .6f}), skew={cand.skew_value: .6f}"
        )
    return "\n".join(lines)


def print_experiment_summary(
    dataset_name,
    n_points,
    primary_dirs,
    primary_time,
    rotated_dirs=None,
    rotated_time=None,
):
    """
    Print a comprehensive summary of the experiment results, including:
    - Dataset information
    - Performance metrics (runtime)
    - Top directions

    Same format as in the reimplementation experiment script, so that
    results can be compared side-by-side.
    """
    print("\n" + "=" * 80)
    print(f"EXPERIMENT SUMMARY (OFFICIAL-STYLE): {dataset_name}")
    print("=" * 80)

    print("\nDataset Information:")
    print(f"  - Number of points: {n_points}")

    print("\nPerformance Metrics:")
    print(f"  - Primary point set:          {primary_time:.4f} s")
    if rotated_time is not None:
        print(f"  - Rotated point set:          {rotated_time:.4f} s")

    print("\nTop Directions (primary, official-style):")
    if primary_dirs:
        print(f"  - Best skew: {primary_dirs[0].skew_value:.6f}")
        d0 = np.asarray(primary_dirs[0].direction, dtype=float)
        print(
            "  - Direction: "
            f"({d0[0]:.6f}, {d0[1]:.6f})"
        )
        if len(primary_dirs) > 2:
            print(
                "  - Top-3 skew range: "
                f"[{primary_dirs[2].skew_value:.6f}, "
                f"{primary_dirs[0].skew_value:.6f}]"
            )

    if rotated_dirs:
        print("\nRotated Point Set Results (official-style):")
        print(f"  - Best skew: {rotated_dirs[0].skew_value:.6f}")
        d1 = np.asarray(rotated_dirs[0].direction, dtype=float)
        print(
            "  - Direction: "
            f"({d1[0]:.6f}, {d1[1]:.6f})"
        )

    print("=" * 80 + "\n")


def validate_and_visualize_tail(
    data,
    x_train_new,
    direction_vec,
    targets_sample=None,
    use_official_f=False,
):
    """
    Validate skew value along a chosen direction and visualize the tail
    region, reusing the same visualization logic as the reimplementation
    experiment so that only the direction source differs.
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
        print("\nUsing official-style top direction for visualization.")

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
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Arrested",
                markerfacecolor="#1f77b4",
                markersize=10,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Not Arrested",
                markerfacecolor="#2ca02c",
                markersize=10,
            ),
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
            except Exception:
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
            except Exception:
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
        print(f"Global positive rate (Arrest=1): {pos_rate_all:.3f}")
        print(f"Tail positive rate   (Arrest=1): {pos_rate_tail:.3f}")
        if not np.isnan(acc_all):
            print(f"Global accuracy of baseline model: {acc_all:.3f}")
            print(f"Tail   accuracy of baseline model: {acc_tail:.3f}")
            print(f"Global F1 of baseline model:       {f1_all:.3f}")
            print(f"Tail   F1 of baseline model:       {f1_tail:.3f}")


def main(argv=None):
    """
    Command-line entry point for reproducing the 2D experiment on
    the Chicago Crimes dataset using the official-style implementation.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run the 2D Ray-sweeping algorithm (official-style implementation) "
            "on the Chicago Crimes dataset, following the 2D experimental "
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
            "(default: pi/10, matching official implementation)."
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
            "visualization instead of the official-style top direction."
        ),
    )

    args = parser.parse_args(argv)

    print("Loading Chicago Crimes dataset...")
    data = load_chicago_crimes_data(args.csv_path)
    print(f"Loaded dataset with shape: {data.shape}")

    x_train_new, x_train_new_prime, targets, sampled_df = build_point_sets_from_data(
        data, n_samples=args.n_samples
    )
    print(f"Primary point set shape: {x_train_new.shape}")
    print(f"Rotated point set shape: {x_train_new_prime.shape}")

    # Visualization of the 2D point sets, following the original notebook.
    print("\nVisualizing primary and rotated point sets (official-style run)...")
    plot_point_sets(x_train_new, x_train_new_prime, targets)

    # Convert to list-of-tuples for the ray_sweeping_2d_official_style interface.
    points_primary = [tuple(row) for row in x_train_new]
    points_rotated = [tuple(row) for row in x_train_new_prime]

    print("\nRunning Ray-sweeping (official-style) on primary point set...")
    primary_dirs, primary_time = run_ray_sweeping_official_on_points(
        points_primary,
        top_k=args.top_k,
        min_angle_step=args.min_angle_step,
        vector_transfer=lambda x: (x[0], x[1]),
    )
    print(f"Time (primary, official-style): {primary_time:.4f} s")
    print("Top directions (primary, official-style):")
    print(format_top_directions(primary_dirs))

    print("\nRunning Ray-sweeping (official-style) on rotated point set...")
    rotated_dirs, rotated_time = run_ray_sweeping_official_on_points(
        points_rotated,
        top_k=args.top_k,
        min_angle_step=args.min_angle_step,
        vector_transfer=lambda x: (-x[1], x[0]),
    )
    print(f"Time (rotated, official-style): {rotated_time:.4f} s")
    print("Top directions (rotated, official-style):")
    print(format_top_directions(rotated_dirs))

    # Choose directions for validation / tail visualization (global plot only,
    # for direction comparison).
    if primary_dirs:
        max_dirs_to_visualize = min(3, len(primary_dirs))
        for rank in range(max_dirs_to_visualize):
            top_dir_vec = np.asarray(primary_dirs[rank].direction, dtype=float)
            print(
                f"\nVisualizing full data along top-{rank + 1} "
                "primary direction (official-style)..."
            )
            validate_and_visualize_tail(
                data,
                x_train_new,
                top_dir_vec,
                targets,
                use_official_f=args.use_official_f,
            )

    # Print comprehensive experiment summary
    print_experiment_summary(
        dataset_name="Chicago Crimes",
        n_points=len(x_train_new),
        primary_dirs=primary_dirs,
        primary_time=primary_time,
        rotated_dirs=rotated_dirs,
        rotated_time=rotated_time,
    )


if __name__ == "__main__":
    main()


