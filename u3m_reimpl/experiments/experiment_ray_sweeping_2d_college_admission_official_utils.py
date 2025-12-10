from __future__ import annotations

"""
Unified experiment script for College Admission dataset with configurable algorithm variants.

This script supports all implementation variants through command-line parameters:
- Official-style (dict+list) vs Official port (LinkedList)
- Different initialization strategies
- Optional preprocessing steps (min-shift, vector_transfer)
- Normalization methods (L1 vs L2)

Visualization:
- Point sets visualization: implemented directly in main() (lines 151-163)
- Tail visualization: via validate_and_visualize_tail_naive() (lines 207-215)
  This function includes: full data plot, tail subset plot, density curve, 
  global vs tail statistics, and multi-percentile evaluation.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from u3m_reimpl.experiments.experiment_ray_sweeping_2d_college_admission_official_style import (
    load_college_admission_data,
    build_point_sets_from_data,
)
from u3m_reimpl.experiments.experiment_ray_sweeping_2d_college_admission import (
    validate_and_visualize_tail as validate_and_visualize_tail_naive,
)
from u3m_reimpl.algorithms.ray_sweeping_2d_official import ray_sweeping_2d_official, SkewDirection
from u3m_reimpl.algorithms.ray_sweeping_2d_official_style import ray_sweeping_2d_official_style


def format_top_directions(entries: List[tuple]) -> str:
    """Format top-k directions as Markdown table."""
    if not entries:
        return "No directions found."
    
    lines = ["| Rank | Direction (x, y) | Skew |", "|------|------------------|------|"]
    for i, (sk, f_dir) in enumerate(entries, start=1):
        f = np.asarray(f_dir, dtype=float)
        lines.append(f"| {i} | ({f[0]:.6f}, {f[1]:.6f}) | {sk:.6f} |")
    return "\n".join(lines)


def main(argv=None):
    """Run ray-sweeping experiment on College Admission dataset with configurable variants."""
    parser = argparse.ArgumentParser(
        description="Unified ray-sweeping experiment with configurable algorithm variants."
    )
    parser.add_argument(
        "--excel-path",
        type=str,
        default=None,
        help=(
            "Optional path to the Admission.xlsx file. "
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
        "--use-official-style",
        action="store_true",
        help="Use official-style implementation (dict+list) instead of linked-list official port.",
    )
    parser.add_argument(
        "--use-linkedlist",
        action="store_true",
        help="(Only when --use-official-style) use official-style with LinkedList storage; otherwise dict+list.",
    )
    parser.add_argument(
        "--use-first-intersection-init",
        action="store_true",
        help="(Only when --use-official-style) start from first intersection and push first direction (ray_sweeping_2d style).",
    )
    parser.add_argument(
        "--disable-min-shift",
        action="store_true",
        help="(Only when --use-official-style) disable min-shift preprocessing.",
    )
    parser.add_argument(
        "--disable-vector-transfer",
        action="store_true",
        help="(Only when --use-official-style) disable vector_transfer (use identity mapping).",
    )
    parser.add_argument(
        "--use-l2-norm",
        action="store_true",
        help="(Only when --use-official-style) use L2 normalization instead of official L1.",
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

    # Setup results directory and logging
    results_root = Path("results") / "college_admission_official_utils"
    results_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_root / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save parameters
    with (run_dir / "run_params.json").open("w", encoding="utf-8") as f:
        json.dump({**vars(args), "timestamp": ts}, f, indent=2, ensure_ascii=False)
    
    # Setup logging
    log_file = (run_dir / "college_admission_official_utils_log.md").open("w", encoding="utf-8")
    
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

    # Load data and build point sets
    print("Loading College Admission dataset...")
    data = load_college_admission_data(args.excel_path)
    print(f"Loaded dataset with shape: {data.shape}")
    
    x_train_new, x_train_new_prime, targets = build_point_sets_from_data(
        data, n_samples=args.n_samples
    )
    print(f"Primary point set shape: {x_train_new.shape}")
    print(f"Rotated point set shape: {x_train_new_prime.shape}")
    
    # Visualize point sets
    print("\nVisualizing primary and rotated point sets...")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, pts, title in zip(axes, [x_train_new, x_train_new_prime], 
                               ["Primary points (gre, gpa)", "Rotated points [max_gpa - gpa, gre]"]):
        if targets is not None:
            colors = ["#1f77b4" if t == 1 else "#2ca02c" for t in targets]
            ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=10)
        else:
            ax.scatter(pts[:, 0], pts[:, 1], s=10)
        ax.set_title(title)
        ax.set_xlabel("gre" if "Primary" in title else "x'")
        ax.set_ylabel("gpa" if "Primary" in title else "y'")
    plt.tight_layout()
    fig.savefig(run_dir / "official_utils_point_sets.png", dpi=200)
    print("![official_utils point sets](official_utils_point_sets.png)")
    plt.close()

    # Run ray-sweeping algorithm
    print("\nRunning ray-sweeping...")
    start = time.time()
    vector_transfer = lambda x: tuple([-x[1], x[0]])  # Map from rotated to original space
    
    if args.use_official_style:
        if args.disable_vector_transfer:
            vector_transfer = lambda x: (x[0], x[1])
        
        topk_dirs = ray_sweeping_2d_official_style(
            x_train_new_prime,
            top_k=args.top_k,
            min_angle_step=args.min_angle_step,
            vector_transfer=vector_transfer,
            use_linkedlist=args.use_linkedlist,
            use_first_intersection_init=args.use_first_intersection_init,
            use_min_shift=not args.disable_min_shift,
            enable_vector_transfer=not args.disable_vector_transfer,
            use_l1_norm=not args.use_l2_norm,
        )
        impl_name = "official-style"
    else:
        topk_dirs = ray_sweeping_2d_official(
            x_train_new_prime,
            top_k=args.top_k,
            epsilon=args.min_angle_step,
            vector_transfer=vector_transfer,
        )
        impl_name = "official"
    
    elapsed = time.time() - start
    print(f"Total time ({impl_name}): {elapsed:.4f} s")
    
    # Output results
    topk_entries = [(d.skew_value, d.direction.as_array()) for d in topk_dirs]
    print("\nTop-k high-skew directions:")
    print(format_top_directions(topk_entries))

    # Visualize top-3 directions
    if topk_entries:
        for rank in range(min(3, len(topk_entries))):
            skew_val, direction = topk_entries[rank]
            print(f"\nVisualizing top-{rank + 1} direction (skew={skew_val:.6f})...")
            validate_and_visualize_tail_naive(
                data=data,
                x_train_new=x_train_new,
                direction_vec=np.asarray(direction, dtype=float),
                use_official_f=False,
                save_prefix=f"official_utils_top{rank+1}",
                results_dir=run_dir,
            )


if __name__ == "__main__":
    main()
