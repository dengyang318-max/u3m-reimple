from __future__ import annotations

"""
Experiment script: apply the 2D Ray-sweeping algorithm (official-style
implementation) to the College Admission dataset, following the same 2D
experimental procedure as `experiment_ray_sweeping_2d_college_admission.py`.

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
OFFICIAL_F_DIRECTION_COLLEGE = np.array(
    [-0.42608587096237255, 0.5739141290376275], dtype=float
)


def _resolve_college_excel_path(excel_path):
    """
    Resolve the path to the College Admission Excel file.
    Same logic as in the reimplementation script.
    """

    if excel_path is not None:
        p = Path(excel_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Could not find dataset file at {p}. "
                "Please check the --excel-path argument."
            )
        return p

    # Fallback: use kagglehub to download the dataset, as in the notebook.
    try:
        import kagglehub  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "kagglehub is not installed, and no --excel-path was provided. "
            "Install kagglehub (`pip install kagglehub`) or pass --excel-path "
            "manually."
        ) from exc

    print("No --excel-path provided. Downloading College Admission dataset via kagglehub...")
    base_path = kagglehub.dataset_download("eswarchandt/admission")
    excel_file = Path(base_path) / "Admission.xlsx"
    if not excel_file.exists():
        raise FileNotFoundError(
            f"Downloaded dataset directory {base_path}, but could not find "
            f"`Admission.xlsx` inside it."
        )
    print(f"Using dataset file: {excel_file}")
    return excel_file


def load_college_admission_data(excel_path):
    """
    Load the College Admission dataset and normalize GRE / GPA into [0, 1],
    exactly as in the reimplementation script.
    """

    excel_path_resolved = _resolve_college_excel_path(excel_path)
    data = pd.read_excel(excel_path_resolved)
    if "gre" not in data.columns or "gpa" not in data.columns:
        raise ValueError(
            "Expected columns `gre` and `gpa` in the dataset; "
            f"got columns: {list(data.columns)}"
        )

    gre = data["gre"].astype(float)
    gpa = data["gpa"].astype(float)

    # Min-max normalize into [0, 1], matching the notebook.
    data["gre"] = (gre - gre.min()) / (gre.max() - gre.min())
    data["gpa"] = (gpa - gpa.min()) / (gpa.max() - gpa.min())
    return data


def build_point_sets_from_data(data, n_samples: int | None = None):
    """
    Construct the two 2D point sets used in the 2D experiment:

    - x_train_new:       (gre, gpa)
    - x_train_new_prime: [max_gpa - gpa, gre]

    Same as in the reimplementation script. By default, this function
    shuffles the entire dataset with a fixed random seed (frac=1,
    random_state=0), exactly as in the official notebook. If `n_samples`
    is provided and smaller than the dataset size, a random subset of
    that many rows is drawn with the same seed.
    """

    if not {"gre", "gpa"}.issubset(set(data.columns)):
        raise ValueError("Data must contain `gre` and `gpa` columns.")

    # Match official notebook behavior:
    # - If n_samples is None: shuffle all rows (frac=1, random_state=0)
    # - If n_samples is provided: draw a shuffled subset of that many rows.
    if n_samples is None:
        df_used = data.sample(frac=1.0, random_state=0)
    else:
        if len(data) < n_samples:
            raise ValueError(
                f"Requested {n_samples} samples, but dataset only has {len(data)} rows."
            )
        df_used = data.sample(n=n_samples, random_state=0)

    x_train_new = np.asarray(df_used[["gre", "gpa"]], dtype=float)
    max_gpa = float(np.max(x_train_new[:, 1]))
    x_train_new_prime = np.column_stack(
        (
            max_gpa - x_train_new[:, 1],
            x_train_new[:, 0],
        )
    )

    if "admit" in df_used.columns:
        targets = np.asarray(df_used["admit"], dtype=float)
    else:
        targets = None

    return x_train_new, x_train_new_prime, targets


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
        fig.colorbar(
            sc1,
            ax=axes[1],
            label="admit",
            fraction=0.046,
            pad=0.04,
        )
    else:
        axes[0].scatter(x_train_new[:, 0], x_train_new[:, 1], s=10)
        axes[1].scatter(x_train_new_prime[:, 0], x_train_new_prime[:, 1], s=10)

    axes[0].set_title("Primary points (gre, gpa)")
    axes[0].set_xlabel("gre")
    axes[0].set_ylabel("gpa")

    axes[1].set_title("Rotated points [max_gpa - gpa, gre]")
    axes[1].set_xlabel("x'")
    axes[1].set_ylabel("y'")

    plt.tight_layout()
    plt.show()


def validate_and_visualize_tail(
    data,
    x_train_new,
    direction_vec,
    use_official_f=False,
):
    """
    Validate skew value along a chosen direction and visualize the tail
    region, reusing the same visualization logic as the reimplementation
    experiment, but using directions from the official-style algorithm.
    """

    if x_train_new.size == 0:
        return

    # Choose direction: use official f_direction if requested, otherwise use the provided direction_vec
    if use_official_f:
        f = OFFICIAL_F_DIRECTION_COLLEGE.copy()
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

    # 1) Full point cloud with f overlaid (copied from reimplementation script)
    full_points = x_train_new
    fig_full, ax_full = plt.subplots(figsize=(5, 5))

    color_handles: list[Line2D] = []
    if "Gender_Male" in data.columns:
        labels_full = data["Gender_Male"].to_numpy()
        colors_full = ["#1f77b4" if x == 1 else "#2ca02c" for x in labels_full]
        ax_full.scatter(full_points[:, 0], full_points[:, 1], c=colors_full, s=10)
        color_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Male",
                markerfacecolor="#1f77b4",
                markersize=10,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Not Male",
                markerfacecolor="#2ca02c",
                markersize=10,
            ),
        ]
    elif "admit" in data.columns:
        labels_full = data["admit"].to_numpy()
        colors_full = ["#1f77b4" if x == 1 else "#2ca02c" for x in labels_full]
        ax_full.scatter(full_points[:, 0], full_points[:, 1], c=colors_full, s=10)
        color_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Admitted",
                markerfacecolor="#1f77b4",
                markersize=10,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Not Admitted",
                markerfacecolor="#2ca02c",
                markersize=10,
            ),
        ]
    else:
        ax_full.scatter(full_points[:, 0], full_points[:, 1], s=10, color="#1f77b4")

    xmin_f, xmax_f = full_points[:, 0].min(), full_points[:, 0].max()
    ymin_f, ymax_f = full_points[:, 1].min(), full_points[:, 1].max()
    x0_f, y0_f = full_points[:, 0].mean(), full_points[:, 1].mean()

    # Tail region shading (same as reimplementation script)
    proj_full = full_points @ f
    q_tail_viz = 0.1
    thresh_tail_viz = np.quantile(
        proj_full, q_tail_viz if -sk < 0 else (1.0 - q_tail_viz)
    )

    if -sk < 0:
        tail_mask_viz = proj_full < thresh_tail_viz
    else:
        tail_mask_viz = proj_full > thresh_tail_viz
    tail_points_viz = full_points[tail_mask_viz]
    if len(tail_points_viz) > 0:
        from scipy.spatial import ConvexHull
        from matplotlib.patches import Polygon

        try:
            if len(tail_points_viz) >= 3:
                hull = ConvexHull(tail_points_viz)
                tail_polygon = tail_points_viz[hull.vertices]
                poly = Polygon(
                    tail_polygon, alpha=0.3, color="orange", label="Tail region"
                )
                ax_full.add_patch(poly)
        except Exception:
            pass

    # Direction line
    t_candidates = []
    if abs(f[0]) > 1e-10:
        t_candidates.append((xmin_f - x0_f) / f[0])
        t_candidates.append((xmax_f - x0_f) / f[0])
    if abs(f[1]) > 1e-10:
        t_candidates.append((ymin_f - y0_f) / f[1])
        t_candidates.append((ymax_f - y0_f) / f[1])

    if t_candidates:
        t_min = min(t_candidates)
        t_max = max(t_candidates)
        t_range = t_max - t_min
        t_min -= t_range * 0.1
        t_max += t_range * 0.1
    else:
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
    xmax_ticks = np.ceil(xmax_f / 0.2) * 0.2
    ymax_ticks = np.ceil(ymax_f / 0.2) * 0.2
    ax_full.set_xticks(np.arange(0, xmax_ticks + 0.2, 0.2))
    ax_full.set_yticks(np.arange(0, ymax_ticks + 0.2, 0.2))
    ax_full.set_xlabel("Normalized GRE", fontsize=13)
    ax_full.set_ylabel("Normalized GPA", fontsize=13)
    ax_full.set_title("Full data along high-skew direction", fontsize=13)

    plt.tight_layout()
    plt.show()

    # Tail visualization: select points in extreme q-quantile along f.
    q = 0.1  # 10% tail, as in the notebook

    df = data.copy(deep=True)
    if not {"gre", "gpa"}.issubset(df.columns):
        return

    # Re-normalize to guard against any upstream changes.
    df["gre"] = (df["gre"] - df["gre"].min()) / (df["gre"].max() - df["gre"].min())
    df["gpa"] = (df["gpa"] - df["gpa"].min()) / (df["gpa"].max() - df["gpa"].min())

    coords = df[["gre", "gpa"]].to_numpy(dtype=float)
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

    # 3) Global vs tail statistics (label proportion + model accuracy)
    if "admit" in data.columns:
        y_all = data["admit"].to_numpy(dtype=int)
        y_tail = tail["admit"].to_numpy(dtype=int)

        pos_rate_all = float(y_all.mean())
        pos_rate_tail = float(y_tail.mean()) if len(y_tail) > 0 else float("nan")

        # Use all features except target (matching official notebook)
        target = "admit"
        features = [x for x in data.columns if x != target]
        
        X_all = data[features].to_numpy(dtype=float)
        X_tail = tail[features].to_numpy(dtype=float)

        # Use Logistic Regression as in the paper
        clf = LogisticRegression(random_state=1, max_iter=1000)
        clf.fit(X_all, y_all)

        y_pred_all = clf.predict(X_all)
        y_pred_tail = clf.predict(X_tail)

        acc_all = float(accuracy_score(y_all, y_pred_all))
        acc_tail = float(accuracy_score(y_tail, y_pred_tail))
        f1_all = float(f1_score(y_all, y_pred_all))
        f1_tail = float(f1_score(y_tail, y_pred_tail))

        print("\n=== Global vs Tail statistics (College Admission) ===")
        print(f"Global positive rate (admit=1): {pos_rate_all:.3f}")
        print(f"Tail positive rate   (admit=1): {pos_rate_tail:.3f}")
        print(f"Global accuracy of baseline model: {acc_all:.3f}")
        print(f"Tail   accuracy of baseline model: {acc_tail:.3f}")
        print(f"Global F1 of baseline model:       {f1_all:.3f}")
        print(f"Tail   F1 of baseline model:       {f1_tail:.3f}")
        
        # 4) Multi-percentile evaluation table with Female/Male ratio (matching Table 3 in paper)
        print("\n=== Multi-Percentile Evaluation Table ===")
        percentiles = [1.00, 0.50, 0.20, 0.10, 0.08, 0.04]
        
        # Calculate total Female/Male ratio
        if "Gender_Male" in data.columns:
            gender_counts_all = data["Gender_Male"].value_counts()
            if len(gender_counts_all) >= 2:
                total_female_ratio = gender_counts_all[0] / gender_counts_all[1]  # Female (0) / Male (1)
            else:
                total_female_ratio = float("nan")
        else:
            total_female_ratio = float("nan")
        
        print(f"{'Percentile':<12} {'Accuracy':<12} {'F1-score':<12} {'F/M ratio':<12}")
        print("-" * 48)
        
        # Calculate for each percentile
        for q in percentiles:
            thresh_q = np.quantile(proj, q if -sk < 0 else (1.0 - q))
            if -sk < 0:
                mask_q = proj < thresh_q
            else:
                mask_q = proj > thresh_q
            tail_q = df.loc[mask_q]
            
            if len(tail_q) > 0 and "admit" in tail_q.columns:
                y_tail_q = tail_q["admit"].to_numpy(dtype=int)
                X_tail_q = tail_q[features].to_numpy(dtype=float) if all(col in tail_q.columns for col in features) else None
                
                # Calculate Female/Male ratio in tail
                if "Gender_Male" in tail_q.columns:
                    gender_counts_tail = tail_q["Gender_Male"].value_counts()
                    if len(gender_counts_tail) >= 2:
                        tail_female_ratio = gender_counts_tail[0] / gender_counts_tail[1]  # Female (0) / Male (1)
                    else:
                        tail_female_ratio = float("nan")
                else:
                    tail_female_ratio = float("nan")
                
                if X_tail_q is not None and len(X_tail_q) > 0:
                    y_pred_tail_q = clf.predict(X_tail_q)
                    acc_q = float(accuracy_score(y_tail_q, y_pred_tail_q))
                    f1_q = float(f1_score(y_tail_q, y_pred_tail_q))
                    if not np.isnan(tail_female_ratio):
                        print(f"{q:<12.2f} {acc_q:<12.3f} {f1_q:<12.3f} {tail_female_ratio:<12.3f}")
                    else:
                        print(f"{q:<12.2f} {acc_q:<12.3f} {f1_q:<12.3f} {'N/A':<12}")
                else:
                    print(f"{q:<12.2f} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
            else:
                print(f"{q:<12.2f} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

    points_np = tail[["gre", "gpa"]].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(5, 5))

    # Color by Gender_Male if available, otherwise by admit, otherwise single color.
    legend_handles: list[Line2D] = []
    if "Gender_Male" in tail.columns:
        labels = tail["Gender_Male"].to_numpy()
        colors = ["#1f77b4" if x == 1 else "#2ca02c" for x in labels]
        ax.scatter(points_np[:, 0], points_np[:, 1], c=colors, s=10)
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", label="Male", markerfacecolor="#1f77b4", markersize=10),
            Line2D([0], [0], marker="o", color="w", label="Not Male", markerfacecolor="#2ca02c", markersize=10),
        ]
    elif "admit" in tail.columns:
        labels = tail["admit"].to_numpy()
        colors = ["#1f77b4" if x == 1 else "#2ca02c" for x in labels]
        ax.scatter(points_np[:, 0], points_np[:, 1], c=colors, s=10)
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", label="Admitted", markerfacecolor="#1f77b4", markersize=10),
            Line2D([0], [0], marker="o", color="w", label="Not Admitted", markerfacecolor="#2ca02c", markersize=10),
        ]
    else:
        ax.scatter(points_np[:, 0], points_np[:, 1], s=10, color="#1f77b4")

    # Draw the direction line using a parametric form centered on the full data.
    # IMPORTANT: Use the same reference point (full_points center) as in the full data plot
    # to ensure the line appears at the same position (same intercept) in both plots.
    # The tail plot re-normalizes the entire dataset (df), but since data is already in [0,1]
    # and the re-normalization is based on the entire dataset's min/max (which are 0 and 1),
    # the re-normalized coordinates are identical to the original normalized coordinates.
    # Therefore, we can directly use x0_f, y0_f as the reference point.
    xmin, xmax = points_np[:, 0].min(), points_np[:, 0].max()
    ymin, ymax = points_np[:, 1].min(), points_np[:, 1].max()
    ymax_tail = ymax + 0.01  # Define ymax_tail once for use in both line calculation and ylim
    
    # Use the full_points center directly (since coordinate systems are equivalent)
    x0, y0 = x0_f, y0_f
    
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
    ax.set_xlabel("Normalized GRE", fontsize=13)
    ax.set_ylabel("Normalized GPA", fontsize=13)
    ax.set_title("Tail subset along high-skew direction", fontsize=13)

    plt.tight_layout()
    plt.show()
    
    # ------------------------------------------------------------------
    # 5) Projection density plot (similar to Figure 4(b) for Chicago Crimes)
    # ------------------------------------------------------------------
    # Project all points onto the direction f
    coords_all = df[["gre", "gpa"]].to_numpy(dtype=float)
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
    
    # Mark tail region (orange) - using 0.1 percentile as in the visualization
    q_tail = 0.1
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


def format_top_directions(directions: List[SkewDirection]) -> str:
    """
    Pretty-print helper for the top-k directions with their skew values.
    """

    lines: List[str] = []
    for i, cand in enumerate(directions, start=1):
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
    Print a comprehensive summary of the experiment results.
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


def main(argv=None):
    """
    Command-line entry point for reproducing the 2D experiment on
    the College Admission dataset using the official-style implementation.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run the 2D Ray-sweeping algorithm (official-style implementation) on "
            "the College Admission dataset, following the 2D experimental "
            "procedure from the paper."
        )
    )
    parser.add_argument(
        "--excel-path",
        type=str,
        default=None,
        help=(
            "Optional path to the College Admission Excel file "
            "(e.g. Admission.xlsx) with at least `gre`, `gpa`, and `admit` "
            "columns. If omitted, the script will try to download the "
            "dataset via kagglehub."
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
        "--use-official-f",
        action="store_true",
        help=(
            "Use the official f_direction from the original notebook for "
            "visualization instead of the official-style top direction."
        ),
    )

    args = parser.parse_args(argv)

    print("Loading College Admission dataset...")
    data = load_college_admission_data(args.excel_path)
    print(f"Loaded dataset with shape: {data.shape}")

    x_train_new, x_train_new_prime, targets = build_point_sets_from_data(data)
    print(f"Primary point set shape: {x_train_new.shape}")
    print(f"Rotated point set shape: {x_train_new_prime.shape}")

    print("\nVisualizing primary and rotated point sets (official-style run)...")
    plot_point_sets(x_train_new, x_train_new_prime, targets)

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

    # Visualize a few top directions (global plot only, for direction comparison)
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
                use_official_f=args.use_official_f,
            )

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

    print_experiment_summary(
        dataset_name="College Admission",
        n_points=len(x_train_new),
        primary_dirs=primary_dirs,
        primary_time=primary_time,
        rotated_dirs=rotated_dirs,
        rotated_time=rotated_time,
    )


if __name__ == "__main__":
    main()


