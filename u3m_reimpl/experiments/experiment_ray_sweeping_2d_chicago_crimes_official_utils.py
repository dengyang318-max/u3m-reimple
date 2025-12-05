from __future__ import annotations

"""
Experiment script: run the ORIGINAL official `utils.ray_sweep.MaxSkewCalculator`
on the Chicago Crimes dataset, following (as closely as possible) the same
experimental procedure as our reimplementation scripts.

This allows a direct comparison between:
- our reimplementation (dynamic-median versions), and
- the exact original algorithm implementation from `Mining_U3Ms-main/utils`.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from u3m_reimpl.experiments.experiment_ray_sweeping_2d_chicago_crimes_official_style import (
    load_chicago_crimes_data,
    build_point_sets_from_data,
)
from u3m_reimpl.experiments.experiment_ray_sweeping_2d_chicago_crimes import (
    validate_and_visualize_tail as validate_and_visualize_tail_naive,
)


def _import_official_max_skew_calculator():
    """
    Import `MaxSkewCalculator` from the official repository:
    `Mining_U3Ms-main/Mining_U3Ms-main/utils/ray_sweep.py`
    """
    base_dir = Path(__file__).resolve().parents[2]  # project root
    official_root = base_dir / "Mining_U3Ms-main" / "Mining_U3Ms-main"
    sys.path.insert(0, str(official_root))
    try:
        from utils.ray_sweep import MaxSkewCalculator  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Could not import MaxSkewCalculator from official utils.ray_sweep. "
            "Please ensure that `Mining_U3Ms-main/Mining_U3Ms-main` exists "
            "next to `u3m_reimpl`."
        ) from exc
    return MaxSkewCalculator


def format_top_directions_from_heap(entries: List[tuple]) -> str:
    """
    Official `get_high_skews` yields entries like (skew, direction_tuple).
    This helper pretty-prints them in Markdown table format.
    """
    if not entries:
        return "No directions found."
    
    lines: List[str] = []
    lines.append("| Rank | Direction (x, y) | Skew |")
    lines.append("|------|------------------|------|")
    
    for i, (sk, f_dir) in enumerate(entries, start=1):
        f = np.asarray(f_dir, dtype=float)
        lines.append(
            f"| {i} | ({f[0]:.6f}, {f[1]:.6f}) | {sk:.6f} |"
        )
    return "\n".join(lines)


def visualize_full_points_with_direction(points: np.ndarray, labels: np.ndarray | None, f_direction, title: str):
    """Simple helper to visualize a 2D point cloud with a direction overlaid."""
    fig, ax = plt.subplots(figsize=(5, 5))
    xmin, xmax = np.min(points[:, 0]), np.max(points[:, 0])
    ymin, ymax = np.min(points[:, 1]), np.max(points[:, 1])

    if labels is not None:
        ax.scatter(
            x=points[:, 0],
            y=points[:, 1],
            c=["#1f77b4" if x == 1 else "#2ca02c" for x in labels],
            s=10,
        )
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", label="Positive",
                   markerfacecolor="#1f77b4", markersize=10),
            Line2D([0], [0], marker="o", color="w", label="Negative",
                   markerfacecolor="#2ca02c", markersize=10),
        ]
    else:
        ax.scatter(points[:, 0], points[:, 1], s=10, color="#1f77b4")
        legend_handles = []

    f = np.asarray(f_direction, dtype=float)
    f = f / np.linalg.norm(f)
    x_vals = np.arange(xmin, 2 * xmax)
    y_vals = np.array(list(map(lambda x: x * f[1] / f[0], x_vals)))
    ax.plot(x_vals, y_vals, color="red")
    ax.plot(x_vals, y_vals + max(ymax, xmax), color="red")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Normalized Longitude", fontsize=13)
    ax.set_ylabel("Normalized Latitude", fontsize=13)
    ax.set_title(title, fontsize=13)

    if legend_handles:
        line = Line2D([0], [0], label="Direction f", color="red", linewidth=1)
        ax.legend(handles=[line] + legend_handles, fontsize=11, loc="upper right")

    plt.tight_layout()
    plt.show()


def evaluate_tail_statistics(data: pd.DataFrame, f_direction, q: float = 0.01):
    """
    Reproduce the tail selection and statistics using a fixed direction f,
    similar to the official notebook.
    """
    f = np.asarray(f_direction, dtype=float)
    f = f / np.linalg.norm(f)

    df = data.copy(deep=True)
    df = df[["Lon", "Lat"]]
    df["Lat"] = data["Lat"] - data["Lat"].min()
    df["Lon"] = data["Lon"] - data["Lon"].min()

    coords = df[["Lon", "Lat"]].to_numpy(dtype=float)
    proj = coords @ f
    sk = (np.mean(proj) - np.median(proj)) / np.std(proj)

    q1 = np.quantile(proj, q if -sk < 0 else (1.0 - q))
    tail = data[df.apply(
        lambda x: np.dot((x[0], x[1]), f) < q1 if -sk < 0 else np.dot((x[0], x[1]), f) > q1,
        axis=1,
    )]
    print("Tail shape:", tail.shape, "Full shape:", df.shape)

    # Simple baseline model on tail vs global (same as notebook)
    features = [
        "Month",
        "Hour",
        "Latitude",
        "Longitude",
        "Primary Type",
        "IUCR",
        "FBI Code",
    ]
    target = "Arrest"
    if all(col in data.columns for col in features):
        clf = LogisticRegression(max_iter=1000)
        X_all = data[features].to_numpy(dtype=float)
        y_all = data[target].to_numpy(dtype=int)
        clf.fit(X_all, y_all)

        X_tail = tail[features].to_numpy(dtype=float)
        y_tail = tail[target].to_numpy(dtype=int)

        y_pred_all = clf.predict(X_all)
        y_pred_tail = clf.predict(X_tail)

        acc_all = accuracy_score(y_all, y_pred_all)
        acc_tail = accuracy_score(y_tail, y_pred_tail)
        f1_all = f1_score(y_all, y_pred_all)
        f1_tail = f1_score(y_tail, y_pred_tail)

        print("\n=== Global vs Tail statistics (Chicago Crimes, official utils) ===")
        print("| Metric                             | Global | Tail |")
        print("|------------------------------------|:------:|:----:|")
        print(f"| Positive rate (Arrest=1)          | {y_all.mean():.3f} | {y_tail.mean():.3f} |")
        print(f"| Accuracy (Logistic Regression)    | {acc_all:.3f} | {acc_tail:.3f} |")
        print(f"| F1 (Logistic Regression)          | {f1_all:.3f} | {f1_tail:.3f} |")


def main(argv=None):
    """
    Run the official `MaxSkewCalculator` on the Chicago Crimes dataset.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the official utils.ray_sweep.MaxSkewCalculator on the "
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
            "If omitted, will try to download via kagglehub (same as notebooks)."
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

    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Set up per-run directory and Markdown logging (same style as naive experiment)
    # ------------------------------------------------------------------
    results_root = Path("results") / "chicago_crimes_official_utils"
    results_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_root / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "chicago_crimes_official_utils_log.md"
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
    print(f"# Chicago Crimes Official-Utils Experiment ({ts})\n")

    MaxSkewCalculator = _import_official_max_skew_calculator()

    print("Loading Chicago Crimes dataset (official utils experiment)...")
    data = load_chicago_crimes_data(args.csv_path)
    print(f"Loaded dataset with shape: {data.shape}")

    x_train_new, x_train_new_prime, targets, sampled_df = build_point_sets_from_data(
        data, n_samples=args.n_samples
    )
    print(f"Primary point set shape: {x_train_new.shape}")
    print(f"Rotated point set shape: {x_train_new_prime.shape}")

    # Simple visualization of primary / rotated sets, similar to naive script.
    print("\nVisualizing primary and rotated point sets (official utils)...")
    fig_pts, axes_pts = plt.subplots(1, 2, figsize=(11, 5))
    if targets is not None:
        axes_pts[0].scatter(x_train_new[:, 0], x_train_new[:, 1], c=targets, s=10)
        axes_pts[1].scatter(x_train_new_prime[:, 0], x_train_new_prime[:, 1], c=targets, s=10)
    else:
        axes_pts[0].scatter(x_train_new[:, 0], x_train_new[:, 1], s=10)
        axes_pts[1].scatter(x_train_new_prime[:, 0], x_train_new_prime[:, 1], s=10)
    axes_pts[0].set_title("Primary points (Lon, Lat)")
    axes_pts[0].set_xlabel("Lon")
    axes_pts[0].set_ylabel("Lat")
    axes_pts[1].set_title("Rotated points [max_lat - Lat, Lon]")
    axes_pts[1].set_xlabel("x'")
    axes_pts[1].set_ylabel("y'")
    plt.tight_layout()
    overview_name = "official_utils_point_sets.png"
    fig_pts.savefig(run_dir / overview_name, dpi=200)
    print(f"![official_utils point sets]({overview_name})")
    plt.show()

    # Build points as DataFrames, as in the official notebooks
    points = pd.DataFrame(x_train_new)
    points_prime = pd.DataFrame(x_train_new_prime)

    print("\nRunning MaxSkewCalculator (official utils) on primary & rotated sets...")
    skew_heap: list = []

    start = time.time()
    max_skew_1 = MaxSkewCalculator(points, skew_heap, lambda x: tuple([x[0], x[1]]), args.min_angle_step)
    max_skew_2 = MaxSkewCalculator(points_prime, skew_heap, lambda x: tuple([-x[1], x[0]]), args.min_angle_step)

    max_skew_1.preprocess()
    max_skew_2.preprocess()

    max_skew_1.train()
    max_skew_2.train()
    elapsed = time.time() - start
    print(f"Total time (official utils): {elapsed:.4f} s")

    # Extract top-k from the shared heap (same as notebook)
    topk_entries = list(max_skew_2.get_high_skews(top_k=args.top_k))
    print("\nTop-k high-skew directions (official utils on rotated set):")
    print(format_top_directions_from_heap(topk_entries))

    # 对 top-3 高偏度方向依次进行可视化（全数据 + tail + 密度曲线），
    # 保持和 naive 实验完全一致的风格和保存方式。
    # 
    # 注意：使用 validate_and_visualize_tail_naive 函数，该函数已包含：
    # - 全数据 + Tail 区域（带方向线）
    # - Tail 子集散点
    # - 投影密度曲线（含 tail 阴影）
    # - Global vs Tail 统计对比
    # - 多 Percentile 评估表（使用全部数据，percentiles=[1.0, 0.1, 0.01, 0.001, 0.0001]）
    # 并在 run_dir 中保存对应图片和 Markdown 链接。
    if topk_entries:
        max_dirs_to_visualize = min(3, len(topk_entries))
        for rank in range(max_dirs_to_visualize):
            best_skew, best_f = topk_entries[rank]
            print(
                f"\n[Official utils] Visualizing tail and density "
                f"along top-{rank + 1} direction (skew={best_skew:.6f})..."
            )

            # 使用与 Chicago naive 实验相同的可视化函数（包含多 percentile 评估）
            # 注意：该函数内部已经包含模型训练和 percentile 评估，无需重复训练
            validate_and_visualize_tail_naive(
                data=data,
                x_train_new=x_train_new,
                direction_vec=np.asarray(best_f, dtype=float),
                targets_sample=targets,
                use_official_f=False,
                save_prefix=f"official_utils_top{rank+1}",
                results_dir=run_dir,
            )


if __name__ == "__main__":
    main()


