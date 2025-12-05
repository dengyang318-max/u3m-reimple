from __future__ import annotations

"""
Experiment script: apply the 2D Ray-sweeping algorithm (naive enumeration)
to a real dataset, following the 2D experimental procedure in the paper.

This module mirrors, in a simplified form, the official notebook
`Mining_U3M_Ray_Sweeping_2D_College_Admission.ipynb`:

- Load a college admission dataset with GRE and GPA features.
- Normalize GRE and GPA into [0, 1].
- Build two 2D point clouds:
  (1) (gre, gpa)
  (2) a rotated / reflected version used in the paper:
      [max_gpa - gpa, gre]
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

from u3m_reimpl.algorithms.ray_sweeping_2d import SkewDirection, ray_sweeping_2d
from u3m_reimpl.algorithms.ray_sweeping_2d_original import (
    SkewDirection as SkewDirectionOriginal,
    ray_sweeping_2d as ray_sweeping_2d_original,
)

# Official direction vector from the original notebook
# (used for comparison with the reimplemented algorithm)
OFFICIAL_F_DIRECTION_COLLEGE = np.array([-0.42608587096237255, 0.5739141290376275], dtype=float)


def _resolve_college_excel_path(excel_path: str | Path | None) -> Path:
    """
    Resolve the path to the College Admission Excel file.

    If excel_path is provided, use it directly. Otherwise, try to download
    the dataset using kagglehub, mimicking the official notebook:

        path = kagglehub.dataset_download("eswarchandt/admission")
        excel = path + "/Admission.xlsx"
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


def load_college_admission_data(excel_path: str | Path | None) -> pd.DataFrame:
    """
    Load the College Admission dataset from an Excel file and normalize
    GRE / GPA into [0, 1], as in the original 2D experiment.

    Args:
        excel_path: Optional path to `Admission.xlsx` (or an equivalent Excel
            file) that contains at least the columns `gre`, `gpa`, and `admit`.
            If None, the function will attempt to download the dataset via
            kagglehub.

    Returns:
        A pandas DataFrame with normalized `gre` and `gpa` columns.
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


def build_point_sets_from_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Construct the two 2D point sets used in the 2D experiment:
    
    - x_train_new:       (gre, gpa)
    - x_train_new_prime: [max_gpa - gpa, gre]

    数据打乱方式与官方 notebook 一致：先对整张表做

        final_df = data.sample(frac=1, random_state=0)

    再从 `final_df` 中取出点集和标签。
    """

    if not {"gre", "gpa"}.issubset(set(data.columns)):
        raise ValueError("Data must contain `gre` and `gpa` columns.")

    # 完全对齐官方 notebook：使用 frac=1, random_state=0 打乱顺序
    final_df = data.sample(frac=1.0, random_state=0)

    x_train_new = np.asarray(final_df[["gre", "gpa"]], dtype=float)
    max_gpa = float(np.max(x_train_new[:, 1]))
    x_train_new_prime = np.column_stack(
        (
            max_gpa - x_train_new[:, 1],  # first coordinate
            x_train_new[:, 0],  # second coordinate
        )
    )

    # Optional target labels for visualization (if present).
    targets: np.ndarray | None
    if "admit" in final_df.columns:
        targets = np.asarray(final_df["admit"], dtype=float)
    else:
        targets = None

    return x_train_new, x_train_new_prime, targets


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
        f = OFFICIAL_F_DIRECTION_COLLEGE.copy()
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

    # Color by Gender_Male if available, otherwise by admit, otherwise single color.
    color_handles: list[Line2D] = []
    if "Gender_Male" in data.columns:
        labels_full = data["Gender_Male"].to_numpy()
        colors_full = ["#1f77b4" if x == 1 else "#2ca02c" for x in labels_full]
        ax_full.scatter(full_points[:, 0], full_points[:, 1], c=colors_full, s=10)
        color_handles = [
            Line2D([0], [0], marker="o", color="w", label="Male", markerfacecolor="#1f77b4", markersize=10),
            Line2D([0], [0], marker="o", color="w", label="Not Male", markerfacecolor="#2ca02c", markersize=10),
        ]
    elif "admit" in data.columns:
        labels_full = data["admit"].to_numpy()
        colors_full = ["#1f77b4" if x == 1 else "#2ca02c" for x in labels_full]
        ax_full.scatter(full_points[:, 0], full_points[:, 1], c=colors_full, s=10)
        color_handles = [
            Line2D([0], [0], marker="o", color="w", label="Admitted", markerfacecolor="#1f77b4", markersize=10),
            Line2D([0], [0], marker="o", color="w", label="Not Admitted", markerfacecolor="#2ca02c", markersize=10),
        ]
    else:
        ax_full.scatter(full_points[:, 0], full_points[:, 1], s=10, color="#1f77b4")

    xmin_f, xmax_f = full_points[:, 0].min(), full_points[:, 0].max()
    ymin_f, ymax_f = full_points[:, 1].min(), full_points[:, 1].max()
    x0_f, y0_f = full_points[:, 0].mean(), full_points[:, 1].mean()
    
    # Mark tail region (orange shading) - similar to Figure 4(a) for Chicago Crimes
    # Project all points to find tail threshold
    proj_full = full_points @ f
    q_tail_viz = 0.1  # 10% tail for visualization (matching the tail visualization)
    thresh_tail_viz = np.quantile(proj_full, q_tail_viz if -sk < 0 else (1.0 - q_tail_viz))
    
    # Create a polygon for the tail region
    if -sk < 0:  # Negative skew: tail is on the left side
        tail_mask_viz = proj_full < thresh_tail_viz
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
    ax_full.set_xlabel("Normalized GRE", fontsize=13)
    ax_full.set_ylabel("Normalized GPA", fontsize=13)
    ax_full.set_title("Full data along high-skew direction", fontsize=13)

    plt.tight_layout()

    # Optionally save full-data figure and record into markdown
    if save_prefix is not None:
        base_dir = results_dir or (Path("results") / "college_admission")
        base_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{save_prefix}_full.png"
        fig_full.savefig(base_dir / fname, dpi=200)
        # Markdown image link (MD file is in the same directory)
        print(f"![{save_prefix} full]({fname})")

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

    # ------------------------------------------------------------------
    # 3) Global vs tail statistics (label proportion + model accuracy)
    # ------------------------------------------------------------------
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
        # Markdown table for global vs tail comparison
        print("| Metric                             | Global | Tail |")
        print("|------------------------------------|:------:|:----:|")
        print(f"| Positive rate (admit=1)           | {pos_rate_all:.3f} | {pos_rate_tail:.3f} |")
        print(f"| Accuracy (Logistic Regression)    | {acc_all:.3f} | {acc_tail:.3f} |")
        print(f"| F1 (Logistic Regression)          | {f1_all:.3f} | {f1_tail:.3f} |")
        
        # ------------------------------------------------------------------
        # 4) Multi-percentile evaluation table with Female/Male ratio (matching Table 3 in paper)
        # ------------------------------------------------------------------
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

        # Markdown table header
        print("| Percentile | Accuracy | F1-score | F/M ratio |")
        print("|-----------:|---------:|---------:|----------:|")

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
                    fm_str = f"{tail_female_ratio:.3f}" if not np.isnan(tail_female_ratio) else "N/A"
                    print(f"| {q:>10.2f} | {acc_q:>8.3f} | {f1_q:>8.3f} | {fm_str:>8} |")
                else:
                    print(f"| {q:>10.2f} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} |")
            else:
                print(f"| {q:>10.2f} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} |")

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

    # Optionally save tail figure and record into markdown
    if save_prefix is not None:
        base_dir = results_dir or (Path("results") / "college_admission")
        base_dir.mkdir(parents=True, exist_ok=True)
        fname_tail = f"{save_prefix}_tail.png"
        fig.savefig(base_dir / fname_tail, dpi=200)
        print(f"![{save_prefix} tail]({fname_tail})")

    plt.show()
    
    # ------------------------------------------------------------------
    # 5) Projection density plot (similar to Figure 4(b) for Chicago Crimes)
    # ------------------------------------------------------------------
    # Project all points onto the direction f
    coords_all = df[["gre", "gpa"]].to_numpy(dtype=float)
    proj_all = coords_all @ f
    
    # Create density plot
    from scipy.stats import gaussian_kde
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

    # Optionally save density figure and record into markdown
    if save_prefix is not None:
        base_dir = results_dir or (Path("results") / "college_admission")
        base_dir.mkdir(parents=True, exist_ok=True)
        fname_density = f"{save_prefix}_density.png"
        fig_density.savefig(base_dir / fname_density, dpi=200)
        print(f"![{save_prefix} density]({fname_density})")

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


def main(argv=None):
    """
    Command-line entry point for reproducing the 2D experiment on
    the College Admission dataset using the naive enumeration method.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run the 2D Ray-sweeping algorithm (naive enumeration) on the "
            "College Admission dataset, following the 2D experimental "
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
            "(default: pi/10)."
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
    results_root = Path("results") / "college_admission"
    results_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_root / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "college_admission_log.md"
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
    print(f"# College Admission Ray-Sweeping Experiment ({ts})\n")

    print("Loading College Admission dataset...")
    data = load_college_admission_data(args.excel_path)
    print(f"Loaded dataset with shape: {data.shape}")

    x_train_new, x_train_new_prime, targets = build_point_sets_from_data(data)
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
                use_official_f=args.use_official_f,
            )
        
        # Also visualize using the official top-1 direction from the original notebook
        print("\n" + "=" * 80)
        print("Visualizing tail using official top-1 direction from the original notebook...")
        print(f"Official top-1 direction: {OFFICIAL_F_DIRECTION_COLLEGE}")
        validate_and_visualize_tail(
            data,
            x_train_new,
            OFFICIAL_F_DIRECTION_COLLEGE,
            use_official_f=True,  # Force use of official direction
        )

    # Print comprehensive experiment summary
    # Keep separate results for summary, but use merged results as the main output
    print_experiment_summary(
        dataset_name="College Admission",
        n_points=len(x_train_new),
        primary_dirs=primary_dirs_all[:args.top_k] if primary_dirs_all else [],
        primary_time=primary_time,
        rotated_dirs=rotated_dirs_all[:args.top_k] if rotated_dirs_all else [],
        rotated_time=rotated_time,
        merged_dirs=merged_top_dirs,  # Add merged results to summary
    )


if __name__ == "__main__":
    main()


