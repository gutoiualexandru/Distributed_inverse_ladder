#!/usr/bin/env python3
"""
Circuit Depth and CNOT Count Comparison

Generates side-by-side plots showing:
- Left: Circuit depth vs number of qubits
- Right: CNOT count vs number of qubits

For implementations: naive, batch-2, batch-3, batch-5, batch-ceil(n/4)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from plot_utils import (
    get_naive_metrics,
    get_batch2_metrics,
    LadderMetrics,
    ladder_depth_recursive,
    ladder_cx_recursive,
    set_publication_style,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate circuit depth and CNOT count comparison"
    )
    parser.add_argument(
        "--n-min", type=int, default=5, help="Min qubit count (default: 5)"
    )
    parser.add_argument(
        "--n-max", type=int, default=1000, help="Max qubit count (default: 1000)"
    )
    parser.add_argument(
        "--n-points", type=int, default=200, help="Number of n points (default: 200)"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
        help="Output directory",
    )
    args = parser.parse_args()

    set_publication_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Generate n values - every integer from 4 to 1000
    n_values = np.arange(max(4, args.n_min), min(1001, args.n_max + 1))

    # Compute metrics for all implementations
    implementations = {
        "Naive": [],
        "Batch-2": [],
        "Batch-3": [],
        "Batch-5": [],
        r"Batch-$\lceil n/4 \rceil$": [],
    }

    for n in n_values:
        # Naive
        naive = get_naive_metrics(n)
        implementations["Naive"].append(naive)

        # Batch-2
        batch2 = get_batch2_metrics(n)
        implementations["Batch-2"].append(batch2)

        # Batch-3
        batch3 = LadderMetrics(
            n_qubits=n,
            depth=ladder_depth_recursive(n, chunk_size=3),
            n_cx=ladder_cx_recursive(n, chunk_size=3),
        )
        implementations["Batch-3"].append(batch3)

        # Batch-5
        batch5 = LadderMetrics(
            n_qubits=n,
            depth=ladder_depth_recursive(n, chunk_size=5),
            n_cx=ladder_cx_recursive(n, chunk_size=5),
        )
        implementations["Batch-5"].append(batch5)

        # Batch-ceil(n/4)
        chunk_ceil_n4 = (n + 3) // 4
        batch_ceil_n4 = LadderMetrics(
            n_qubits=n,
            depth=ladder_depth_recursive(n, chunk_size=chunk_ceil_n4),
            n_cx=ladder_cx_recursive(n, chunk_size=chunk_ceil_n4),
        )
        implementations[r"Batch-$\lceil n/4 \rceil$"].append(batch_ceil_n4)

    # Extract depth and CNOT arrays
    depths = {
        name: np.array([m.depth for m in metrics])
        for name, metrics in implementations.items()
    }
    cnots = {
        name: np.array([m.n_cx for m in metrics])
        for name, metrics in implementations.items()
    }

    # Define colors and line styles
    styles = {
        "Naive": {"color": "black", "linestyle": "-", "linewidth": 2.0, "marker": None, "markevery": None},
        "Batch-2": {"color": "#1f77b4", "linestyle": "--", "linewidth": 1.8, "marker": "s", "markevery": 100},
        "Batch-3": {"color": "#8B4513", "linestyle": "-.", "linewidth": 1.6, "marker": "^", "markevery": 100},
        "Batch-5": {"color": "#FF1493", "linestyle": ":", "linewidth": 2.0, "marker": "D", "markevery": 100},
        r"Batch-$\lceil n/4 \rceil$": {
            "color": "#00CED1",
            "linestyle": (0, (3, 1, 1, 1)),
            "linewidth": 1.4,
            "marker": None,
            "markevery": None,
        },
    }

    # Create figure with two subplots
    fig, (ax_depth, ax_cnot) = plt.subplots(1, 2, figsize=(10.0, 4.0))

    # Plot depth (left panel) - log x, linear y
    for name in implementations.keys():
        ax_depth.semilogx(
            n_values,
            depths[name],
            label=name,
            color=styles[name]["color"],
            linestyle=styles[name]["linestyle"],
            linewidth=styles[name]["linewidth"],
            marker=styles[name]["marker"],
            markevery=styles[name]["markevery"],
            markersize=5,
        )

    ax_depth.set_xlabel(r"Number of qubits $n$", fontsize=11)
    ax_depth.set_ylabel(r"Circuit depth", fontsize=11)
    ax_depth.set_xlim(n_values[0], n_values[-1])
    ax_depth.set_ylim(0, 200)
    ax_depth.grid(True, which="both", alpha=0.3)
    ax_depth.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_depth.set_title("Inverse Ladder Circuit Depth", fontsize=11, fontweight="bold")

    # Plot CNOT count (right panel) - linear-linear
    for name in implementations.keys():
        ax_cnot.plot(
            n_values,
            cnots[name],
            label=name,
            color=styles[name]["color"],
            linestyle=styles[name]["linestyle"],
            linewidth=styles[name]["linewidth"],
            marker=styles[name]["marker"],
            markevery=styles[name]["markevery"],
            markersize=5,
        )

    ax_cnot.set_xlabel(r"Number of qubits $n$", fontsize=11)
    ax_cnot.set_ylabel(r"CNOT count", fontsize=11)
    ax_cnot.set_xlim(args.n_min, args.n_max)
    ax_cnot.grid(True, which="both", alpha=0.3)
    ax_cnot.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax_cnot.set_title("Inverse Ladder CNOT Count", fontsize=11, fontweight="bold")

    # Add reference line y = 2n (do this after main plots but before inset)
    n_ref = np.array([n_values[0], n_values[-1]])
    ax_cnot.plot(n_ref, 2 * n_ref, 'k--', linewidth=1.0, alpha=0.5, label=r"$2n$ (reference)")
    # Update legend to include the new line
    ax_cnot.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # Add inset zoom for n = 4 to 20
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_inset = inset_axes(ax_cnot, width="35%", height="35%", loc="upper left", borderpad=2.5)

    # Plot zoomed data
    mask = (n_values >= 4) & (n_values <= 20)
    n_zoom = n_values[mask]

    for name in implementations.keys():
        y_zoom = cnots[name][mask]
        ax_inset.plot(
            n_zoom,
            y_zoom,
            color=styles[name]["color"],
            linestyle=styles[name]["linestyle"],
            linewidth=1.2,
        )

    # Add 2n reference in inset
    n_zoom_ref = np.linspace(4, 20, 50)
    ax_inset.plot(n_zoom_ref, 2 * n_zoom_ref, 'k--', linewidth=0.8, alpha=0.5)

    ax_inset.set_xlim(4, 20)
    ax_inset.set_ylim(0, 50)
    ax_inset.tick_params(labelsize=7)
    ax_inset.grid(True, alpha=0.2)

    fig.tight_layout()

    # Save outputs
    pdf_path = args.out_dir / "depth_cnot_comparison.pdf"
    png_path = args.out_dir / "depth_cnot_comparison.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    print(f"Wrote: {pdf_path}")
    print(f"Wrote: {png_path}")

    # Print analysis for key values of n
    print("\nCircuit Metrics Analysis:")
    print("-" * 80)
    print(f"{'n':>6} | {'Implementation':^25} | {'Depth':>8} | {'CNOT Count':>12}")
    print("-" * 80)

    for n_sample in [10, 50, 100, 500, 1000]:
        if n_sample > args.n_max or n_sample < args.n_min:
            continue
        idx = np.where(n_values == n_sample)[0]
        if len(idx) == 0:
            continue
        idx = idx[0]

        for name in implementations.keys():
            metrics = implementations[name][idx]
            print(
                f"{n_sample:>6} | {name:^25} | {metrics.depth:>8} | {metrics.n_cx:>12}"
            )
        print("-" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
