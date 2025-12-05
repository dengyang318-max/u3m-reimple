"""
Benchmark comparison: three intersection enumeration methods

Compare the runtime of three intersection enumeration methods:
1. Naive enumeration (use_incremental=False, use_randomized=False)
2. Incremental divide-and-conquer (use_incremental=True)
3. Randomized incremental (use_randomized=True)
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path to import u3m_reimpl
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from u3m_reimpl.algorithms.ray_sweeping_2d import ray_sweeping_2d


def generate_points_in_first_quadrant(n: int, seed: int = 42) -> list:
    """Generate n random points in the first quadrant (using normal distribution).

    Args:
        n: number of points
        seed: random seed for reproducibility

    Returns:
        A list of points, each is a (x, y) tuple with x > 0, y > 0.
    """
    np.random.seed(seed)
    points = []
    while len(points) < n:
        x = np.random.normal(1.0, 0.5)
        y = np.random.normal(1.0, 0.5)
        # 只保留第一象限的点
        if x > 0.0 and y > 0.0:
            points.append((float(x), float(y)))
    return points


def benchmark_method(points: list, use_incremental: bool = False,
                     use_randomized: bool = False, top_k: int = 5) -> float:
    """Benchmark the runtime of a single method.

    Args:
        points: list of points
        use_incremental: whether to use incremental divide-and-conquer method
        use_randomized: whether to use randomized incremental method
        top_k: number of top-k directions to compute

    Returns:
        Runtime in seconds.
    """
    start_time = time.perf_counter()
    try:
        ray_sweeping_2d(
            points,
            top_k=top_k,
            use_incremental=use_incremental,
            use_randomized=use_randomized,
        )
        end_time = time.perf_counter()
        return end_time - start_time
    except Exception as e:
        method_name = "randomized" if use_randomized else ("incremental" if use_incremental else "naive")
        print(f"Error with n={len(points)}, method={method_name}: {e}")
        return float('inf')


def run_benchmark():
    """Run the full benchmark comparison."""
    # Different numbers of points to test
    point_counts = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    
    # Store results
    naive_times = []
    incremental_times = []
    randomized_times = []
    
    print("Starting Ray-sweeping intersection enumeration benchmark...")
    print("(Comparing three CPU-only intersection enumeration methods)")
    print("=" * 60)
    
    for n in point_counts:
        print(f"\nNumber of points: {n}")
        
        # Generate test points
        points = generate_points_in_first_quadrant(n, seed=42)
        
        # Naive enumeration
        print(f"  Testing naive enumeration...", end=" ", flush=True)
        naive_time = benchmark_method(points, use_incremental=False, use_randomized=False, top_k=5)
        naive_times.append(naive_time)
        print(f"done: {naive_time:.4f} s")

        # Incremental divide-and-conquer
        print(f"  Testing incremental divide-and-conquer...", end=" ", flush=True)
        incremental_time = benchmark_method(points, use_incremental=True, use_randomized=False, top_k=5)
        incremental_times.append(incremental_time)
        print(f"done: {incremental_time:.4f} s")

        # Randomized incremental
        print(f"  Testing randomized incremental...", end=" ", flush=True)
        randomized_time = benchmark_method(points, use_incremental=False, use_randomized=True, top_k=5)
        randomized_times.append(randomized_time)
        print(f"done: {randomized_time:.4f} s")

        # Speedup relative to naive
        if incremental_time > 0 and naive_time > 0:
            speedup_inc = naive_time / incremental_time
            print(f"  Incremental speedup: {speedup_inc:.2f}x")
        if randomized_time > 0 and naive_time > 0:
            speedup_rand = naive_time / randomized_time
            print(f"  Randomized incremental speedup: {speedup_rand:.2f}x")
    
    print("\n" + "=" * 60)
    print("Benchmark finished!")
    
    return point_counts, naive_times, incremental_times, randomized_times


def plot_results(point_counts, naive_times, incremental_times, randomized_times):
    """Plot line chart for performance comparison."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot lines
    ax.plot(
        point_counts, 
        naive_times, 
        marker='o', 
        linestyle='-', 
        linewidth=2,
        markersize=8,
        label='Naive enumeration (Naive O(n²))',
        color='#2E86AB'
    )
    
    ax.plot(
        point_counts, 
        incremental_times, 
        marker='s', 
        linestyle='--', 
        linewidth=2,
        markersize=8,
        label='Incremental divide-and-conquer',
        color='#A23B72'
    )
    
    ax.plot(
        point_counts,
        randomized_times,
        marker='^',
        linestyle='-.',
        linewidth=2,
        markersize=8,
        label='Randomized incremental (O(m+n log n))',
        color='#F18F01'
    )
    
    # Axes
    ax.set_xlabel('Number of Points', fontsize=12, fontweight='bold')
    ax.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Ray-sweeping Performance Comparison\nThree Intersection Enumeration Methods',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(fontsize=10, loc='upper left')
    
    # Use log-scale on y if necessary
    all_times = [t for t in naive_times + incremental_times + randomized_times if t > 0 and t < float('inf')]
    if all_times and max(all_times) / min(all_times) > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Runtime (seconds, log scale)',
                     fontsize=12, fontweight='bold')
    
    # Optional data labels (only when there are not many points)
    if len(point_counts) <= 10:
        for i, (n, t_naive, t_inc, t_rand) in enumerate(zip(point_counts, naive_times, incremental_times, randomized_times)):
            if t_naive < float('inf'):
                ax.annotate(
                    f'{t_naive:.3f}s',
                    (n, t_naive),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=7,
                    color='#2E86AB',
                )
            if t_inc < float('inf'):
                ax.annotate(
                    f'{t_inc:.3f}s',
                    (n, t_inc),
                    textcoords="offset points",
                    xytext=(0, -15),
                    ha='center',
                    fontsize=7,
                    color='#A23B72',
                )
            if t_rand < float('inf'):
                ax.annotate(
                    f'{t_rand:.3f}s',
                    (n, t_rand),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=7,
                    color='#F18F01',
                )
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent
    output_file = output_dir / 'benchmark_comparison_more_points.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_file}")
    
    # Show figure
    plt.show()


def main():
    """Main entry point."""
    print("Ray-sweeping algorithm performance benchmark")
    print("=" * 60)
    print("Configuration:")
    print("  - Number of points: 500 to 4000")
    print("  - Point distribution: normal distribution in the first quadrant")
    methods_str = "Naive vs Incremental vs Randomized incremental (all on CPU)"
    print(f"  - Compared methods: {methods_str}")
    print("=" * 60)
    
    # Run benchmark
    point_counts, naive_times, incremental_times, randomized_times = run_benchmark()
    
    # Print summary table
    print("\n" + "=" * 60)
    print("Summary table:")
    header = f"{'n':<8} {'Naive (s)':<15} {'Incremental (s)':<18} {'Randomized (s)':<18} {'Best method':<12}"
    print(header)
    print("-" * 80)
    
    for n, t_naive, t_inc, t_rand in zip(
        point_counts, naive_times, incremental_times, randomized_times
    ):
        times = {
            'Naive': t_naive if t_naive < float('inf') else float('inf'),
            'Incremental': t_inc if t_inc < float('inf') else float('inf'),
            'Randomized': t_rand if t_rand < float('inf') else float('inf'),
        }
        best_method = (
            min(times, key=times.get)
            if any(t < float('inf') for t in times.values())
            else 'N/A'
        )

        row = f"{n:<8} {t_naive:<15.4f} {t_inc:<18.4f} {t_rand:<18.4f} {best_method:<12}"
        print(row)

    # Average speedup
    print("\n" + "=" * 60)
    print("Average speedup (relative to naive):")
    valid_pairs_inc = [(t_naive, t_inc) for t_naive, t_inc in zip(naive_times, incremental_times) 
                       if t_naive < float('inf') and t_inc < float('inf') and t_inc > 0]
    valid_pairs_rand = [(t_naive, t_rand) for t_naive, t_rand in zip(naive_times, randomized_times) 
                        if t_naive < float('inf') and t_rand < float('inf') and t_rand > 0]
    
    if valid_pairs_inc:
        avg_speedup_inc = (
            sum(t_naive / t_inc for t_naive, t_inc in valid_pairs_inc)
            / len(valid_pairs_inc)
        )
        print(f"  Incremental: {avg_speedup_inc:.2f}x")
    if valid_pairs_rand:
        avg_speedup_rand = (
            sum(t_naive / t_rand for t_naive, t_rand in valid_pairs_rand)
            / len(valid_pairs_rand)
        )
        print(f"  Randomized: {avg_speedup_rand:.2f}x")

    # Plot figure
    print("\nGenerating plot...")
    plot_results(point_counts, naive_times, incremental_times, randomized_times)


if __name__ == "__main__":
    main()

