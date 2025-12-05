from __future__ import annotations

"""
Experiment script: run the ORIGINAL official `utils.ray_sweep.MaxSkewCalculator`
on the College Admission dataset, following (as closely as possible) the same
experimental procedure as our reimplementation scripts.

This lets us directly compare:
- our reimplementation (dynamic-median versions), and
- the exact original algorithm from `Mining_U3Ms-main/utils`.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

from u3m_reimpl.experiments.experiment_ray_sweeping_2d_college_admission_official_style import (
    load_college_admission_data,
    build_point_sets_from_data,
)
from u3m_reimpl.experiments.experiment_ray_sweeping_2d_college_admission import (
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
    Pretty-print helper for entries yielded by official `get_high_skews`:
    (skew, direction_tuple). Returns a Markdown table format.
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

    ax.set_xlim(xmin, xmax + 0.01)
    ax.set_ylim(ymin, ymax + 0.01)
    ax.set_xlabel("Normalized GRE", fontsize=13)
    ax.set_ylabel("Normalized GPA", fontsize=13)
    ax.set_title(title, fontsize=13)

    if legend_handles:
        line = Line2D([0], [0], label="Direction f", color="red", linewidth=1)
        ax.legend(handles=[line] + legend_handles, fontsize=11, loc="upper right")

    plt.tight_layout()
    plt.show()


def evaluate_tail_statistics(data: pd.DataFrame, f_direction, q: float = 0.10):
    """
    Reproduce tail selection and statistics along a fixed direction f,
    following the official notebook logic.
    """
    f = np.asarray(f_direction, dtype=float)
    f = f / np.linalg.norm(f)

    df = data.copy(deep=True)
    df = df[["gre", "gpa"]]
    df["gre"] = (data["gre"] - data["gre"].min()) / (data["gre"].max() - data["gre"].min())
    df["gpa"] = (data["gpa"] - data["gpa"].min()) / (data["gpa"].max() - data["gpa"].min())

    coords = df[["gre", "gpa"]].to_numpy(dtype=float)
    proj = coords @ f
    sk = (np.mean(proj) - np.median(proj)) / np.std(proj)

    q1 = np.quantile(proj, q if -sk < 0 else (1.0 - q))
    tail = data[df.apply(
        lambda x: np.dot((x[0], x[1]), f) < q1 if -sk < 0 else np.dot((x[0], x[1]), f) > q1,
        axis=1,
    )]
    print("Tail shape:", tail.shape, "Full shape:", df.shape)

    # Baseline model as in the official notebook:
    # use an MLPClassifier trained on all features except the target.
    target = "admit"
    features = [x for x in data.columns if x != target]
    if all(col in data.columns for col in features):
        X_all = data[features].to_numpy(dtype=float)
        y_all = data[target].to_numpy(dtype=int)

        clf = MLPClassifier(random_state=1)
        clf.fit(X_all, y_all)

        X_tail = tail[features].to_numpy(dtype=float)
        y_tail = tail[target].to_numpy(dtype=int)

        y_pred_all = clf.predict(X_all)
        y_pred_tail = clf.predict(X_tail)

        acc_all = accuracy_score(y_all, y_pred_all)
        acc_tail = accuracy_score(y_tail, y_pred_tail)
        f1_all = f1_score(y_all, y_pred_all)
        f1_tail = f1_score(y_tail, y_pred_tail)

        print("\n=== Global vs Tail statistics (College Admission, official utils) ===")
        print("| Metric                             | Global | Tail |")
        print("|------------------------------------|:------:|:----:|")
        print(f"| Positive rate (admit=1)           | {y_all.mean():.3f} | {y_tail.mean():.3f} |")
        print(f"| Accuracy (MLPClassifier)          | {acc_all:.3f} | {acc_tail:.3f} |")
        print(f"| F1 (MLPClassifier)                | {f1_all:.3f} | {f1_tail:.3f} |")


def main(argv=None):
    """
    Run the official `MaxSkewCalculator` on the College Admission dataset.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the official utils.ray_sweep.MaxSkewCalculator on the "
            "College Admission dataset, following the 2D experimental "
            "procedure from the paper."
        )
    )
    parser.add_argument(
        "--excel-path",
        type=str,
        default=None,
        help=(
            "Optional path to the Admission.xlsx file. If omitted, the script "
            "will try to download the dataset via kagglehub (same as notebook)."
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
        default=None,
        help=(
            "Optional number of points to sample from the preprocessed dataset. "
            "If omitted, the script will follow the original notebook and use "
            "all rows, shuffled with random_state=0."
        ),
    )

    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Set up per-run directory and Markdown logging (same style as naive experiment)
    # ------------------------------------------------------------------
    results_root = Path("results") / "college_admission_official_utils"
    results_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_root / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "college_admission_official_utils_log.md"
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
    print(f"# College Admission Official-Utils Experiment ({ts})\n")

    MaxSkewCalculator = _import_official_max_skew_calculator()

    print("Loading College Admission dataset (official utils experiment)...")
    data = load_college_admission_data(args.excel_path)
    print(f"Loaded dataset with shape: {data.shape}")

    x_train_new, x_train_new_prime, targets = build_point_sets_from_data(
        data, n_samples=args.n_samples
    )
    print(f"Primary point set shape: {x_train_new.shape}")
    print(f"Rotated point set shape: {x_train_new_prime.shape}")

    # Simple visualization of primary / rotated sets, mirroring the naive script:
    # if targets are available (admit), use two colors for positive / negative.
    print("\nVisualizing primary and rotated point sets (official utils)...")
    fig_pts, axes_pts = plt.subplots(1, 2, figsize=(11, 5))
    if targets is not None:
        colors = ["#1f77b4" if t == 1 else "#2ca02c" for t in targets]
        sc0 = axes_pts[0].scatter(x_train_new[:, 0], x_train_new[:, 1], c=colors, s=10)
        sc1 = axes_pts[1].scatter(
            x_train_new_prime[:, 0],
            x_train_new_prime[:, 1],
            c=colors,
            s=10,
        )
        # Optional legend to distinguish admitted / not admitted
        from matplotlib.lines import Line2D

        legend_handles = [
            Line2D([0], [0], marker="o", color="w", label="Admitted",
                   markerfacecolor="#1f77b4", markersize=10),
            Line2D([0], [0], marker="o", color="w", label="Not Admitted",
                   markerfacecolor="#2ca02c", markersize=10),
        ]
        axes_pts[0].legend(handles=legend_handles, fontsize=9, loc="upper right")
    else:
        sc0 = axes_pts[0].scatter(x_train_new[:, 0], x_train_new[:, 1], s=10)
        sc1 = axes_pts[1].scatter(x_train_new_prime[:, 0], x_train_new_prime[:, 1], s=10)

    axes_pts[0].set_title("Primary points (gre, gpa)")
    axes_pts[0].set_xlabel("gre")
    axes_pts[0].set_ylabel("gpa")
    axes_pts[1].set_title("Rotated points [max_gpa - gpa, gre]")
    axes_pts[1].set_xlabel("x'")
    axes_pts[1].set_ylabel("y'")
    plt.tight_layout()
    # Save this overview as well into the run directory
    overview_name = "official_utils_point_sets.png"
    fig_pts.savefig(run_dir / overview_name, dpi=200)
    print(f"![official_utils point sets]({overview_name})")
    plt.show()

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

    topk_entries = list(max_skew_2.get_high_skews(top_k=args.top_k))
    print("\nTop-k high-skew directions (official utils on rotated set):")
    print(format_top_directions_from_heap(topk_entries))

    # 对 top-3 高偏度方向依次进行可视化（全数据 + tail + 密度曲线），
    # 保持和 naive 实验完全一致的风格和保存方式。
    if topk_entries:
        max_dirs_to_visualize = min(3, len(topk_entries))
        for rank in range(max_dirs_to_visualize):
            best_skew, best_f = topk_entries[rank]
            print(
                f"\n[Official utils] Visualizing tail and density "
                f"along top-{rank + 1} direction (skew={best_skew:.6f})..."
            )

            validate_and_visualize_tail_naive(
                data=data,
                x_train_new=x_train_new,
                direction_vec=np.asarray(best_f, dtype=float),
                use_official_f=False,
                save_prefix=f"official_utils_top{rank+1}",
                results_dir=run_dir,
            )


if __name__ == "__main__":
    main()


