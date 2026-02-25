#!/usr/bin/env python3
"""
Distributed quantum system resource estimates.

Generates side-by-side plots showing:
- Left: Module capacity vs module budget for different qubit counts
- Right: Transmission rounds vs ring size for different module sizes
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def layered_module_counts(n_qubits: int, module_capacity: int) -> list[int]:
    """Return the number of modules per layer for the distributed ladder scheme."""
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive.")
    if module_capacity < 2:
        raise ValueError("module_capacity must be at least 2.")

    counts: list[int] = []
    current = math.ceil(n_qubits / module_capacity)
    counts.append(current)

    while current > 1:
        current = math.ceil(current / module_capacity)
        counts.append(current)

    return counts


def total_modules_required(n_qubits: int, module_capacity: int) -> int:
    """Total modules across all layers needed to host an n-qubit CNOT ring."""
    return sum(layered_module_counts(n_qubits, module_capacity))


def module_count_bounds(
    n_qubits: int, module_capacity: int
) -> tuple[float, float]:
    """Analytic lower and upper bounds on the total module count.

    The bounds implement
        (n - 1) / (m - 1) <= N_modules <= n / (m - 1) + ceil(log_m n)
    where n is the qubit count and m is the module qubit capacity.
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive.")
    if module_capacity < 2:
        raise ValueError("module_capacity must be at least 2.")

    denominator = module_capacity - 1
    lower = (n_qubits - 1) / denominator
    log_term = math.ceil(math.log(n_qubits, module_capacity))
    upper = (n_qubits / denominator) + log_term
    return lower, upper


def integer_transmission_rounds(num_dependents: int, module_qubits: int) -> int:
    """Ceil-based bound on the number of communication rounds in the protocol."""
    if num_dependents <= 1:
        return 0
    if module_qubits < 2:
        raise ValueError("module_qubits must be at least 2.")
    depth = math.ceil(math.log(num_dependents, module_qubits))
    return 2 * depth + 2


def continuous_transmission_rounds(num_dependents: int, module_qubits: int) -> float:
    """Idealized (real-valued) transmission rounds, kept for comparison."""
    if num_dependents <= 1:
        return 1.0
    if module_qubits < 2:
        raise ValueError("module_qubits must be at least 2.")
    return 2 * math.log(num_dependents, module_qubits) + 1


def plot_module_requirements(
    ax: plt.Axes,
    *,
    module_capacities: Sequence[int],
    target_sizes: Sequence[int],
) -> None:
    colors = ("tab:orange", "tab:blue", "tab:green", "tab:purple")
    for color, n_qubits in zip(colors, target_sizes):
        counts = [
            total_modules_required(n_qubits, capacity)
            for capacity in module_capacities
        ]
        lower_bounds = []
        upper_bounds = []
        for capacity in module_capacities:
            lower, upper = module_count_bounds(n_qubits, capacity)
            lower_bounds.append(lower)
            upper_bounds.append(upper)

        ax.plot(
            module_capacities,
            counts,
            marker="x",
            markersize=6,
            color=color,
            label=f"CNOT ring on {n_qubits} qubits",
        )
        ax.fill_between(
            module_capacities,
            lower_bounds,
            upper_bounds,
            color=color,
            alpha=0.12,
            label=None,
        )
        ax.plot(
            module_capacities,
            lower_bounds,
            linestyle="--",
            linewidth=1.0,
            color=color,
            alpha=0.7,
        )
        ax.plot(
            module_capacities,
            upper_bounds,
            linestyle=":",
            linewidth=1.0,
            color=color,
            alpha=0.7,
        )

    ax.set_xlabel("Qubit capacity per module")
    ax.set_ylabel("Total modules required")
    ax.set_title("Module capacity vs. module budget")
    ax.set_xticks(list(range(8, 34, 2)))
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    ax.grid(True, axis="y")
    ax.legend()


def plot_transmission_rounds(
    ax: plt.Axes,
    *,
    total_qubits: Sequence[int],
    module_qubit_options: Sequence[int],
) -> None:
    colors = {
        4: "tab:orange",
        8: "tab:blue",
        16: "tab:green",
        32: "tab:purple",
    }
    for module_qubits in module_qubit_options:
        raw_dependents = [
            max(1.0, n_qubits / module_qubits) for n_qubits in total_qubits
        ]
        discrete_dependents = [math.ceil(dep) for dep in raw_dependents]
        discrete = [
            integer_transmission_rounds(dep, module_qubits)
            for dep in discrete_dependents
        ]
        continuous = [
            continuous_transmission_rounds(dep, module_qubits)
            for dep in raw_dependents
        ]
        color = colors.get(module_qubits, "tab:gray")
        ax.step(
            total_qubits,
            discrete,
            where="post",
            color=color,
            label=f"{module_qubits} qubits/module",
        )
        ax.plot(
            total_qubits,
            continuous,
            linestyle="--",
            color=color,
        )

    ax.set_xlabel("Total qubits in CNOT ring")
    ax.set_ylabel("Transmission rounds required")
    ax.set_title("Transmission rounds vs. ring size")
    ax.set_xscale("log", base=2)
    xticks = [2**k for k in range(2, 11) if 2**k <= total_qubits[-1]]
    ax.set_xticks(xticks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.grid(True, which="both", axis="x", linestyle="--", alpha=0.5)
    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.5)
    ax.legend()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate distributed resource estimate figure"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
        help="Output directory",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    module_capacities = list(range(8, 33))
    target_sizes = (32, 64, 128, 256)

    total_qubit_range = list(range(4, 1025))
    module_qubit_options = (4, 8, 16, 32)

    fig, (ax_modules, ax_transmissions) = plt.subplots(
        1, 2, figsize=(14, 5)
    )

    plot_module_requirements(
        ax_modules,
        module_capacities=module_capacities,
        target_sizes=target_sizes,
    )
    plot_transmission_rounds(
        ax_transmissions,
        total_qubits=total_qubit_range,
        module_qubit_options=module_qubit_options,
    )

    fig.tight_layout()

    pdf_path = args.out_dir / "resource_estimate.pdf"
    png_path = args.out_dir / "resource_estimate.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    print(f"Wrote: {pdf_path}")
    print(f"Wrote: {png_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
