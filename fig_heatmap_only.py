#!/usr/bin/env python3
"""
Heatmap: Fidelity ratio (batch-2 vs naive) and batch-2 fidelity in parameter space using AMFDN model.

2x2 layout:
- Top row: Fidelity ratio log2(F_batch2 / F_naive) for n=50 and n=100
- Bottom row: Batch-2 fidelity F for n=50 and n=100

X-axis: CX gate error (epsilon)
Y-axis: Coherence ratio (T2/t_cx)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from qiskit.converters import circuit_to_dag, dag_to_circuit

from ladder_circuits import build_naive_ladder, build_batch2_ladder


def depolarizing_thermal_from_errors_dict(circ, error_params, num_qubits, gate_times, t1_times, t2_times, p=0.5):
    """
    Depolarizing model with T1/T2 thermal decay applied layer-by-layer.

    Args:
        circ: Qiskit circuit
        error_params: Dict of gate error rates
        num_qubits: Number of qubits
        gate_times: Dict of gate durations
        t1_times: List of T1 times per qubit
        t2_times: List of T2 times per qubit
        p: Entanglement parameter (0 to 1). p=0 adds correction terms (upper bound),
           p=1 omits them (lower bound).

    Returns product of all qubit fidelities.
    """
    qubits_fidelity = [1.0 for _ in range(num_qubits)]

    dag = circuit_to_dag(circ)
    layers = [dag_to_circuit(layer['graph']) for layer in dag.layers()]

    for layer in layers:
        slowest_gate = 0
        for gate in layer:
            gate_name = gate.name

            if gate_name in gate_times:
                slowest_gate = max(slowest_gate, gate_times[gate_name])

            if gate.operation.num_qubits == 1:
                if gate.operation.name in error_params:
                    qubit = circ.find_bit(gate.qubits[0]).index
                    infidelity = error_params[gate.operation.name]
                    depol = 2 * infidelity
                    qubits_fidelity[qubit] = (1 - depol) * qubits_fidelity[qubit] + (1 - p) * depol / 2
            else:
                if gate.operation.name in error_params:
                    qubit_1 = circ.find_bit(gate.qubits[0]).index
                    qubit_2 = circ.find_bit(gate.qubits[1]).index
                    fid_1, fid_2 = qubits_fidelity[qubit_1], qubits_fidelity[qubit_2]
                    infidelity = error_params[gate.operation.name]
                    depol = (4 / 3) * infidelity
                    c = (1 - p) * 0.5 * (np.sqrt((1 - depol) * (fid_1 + fid_2)**2 + depol) - np.sqrt(1 - depol) * (fid_1 + fid_2))
                    qubits_fidelity[qubit_1] = np.sqrt(1 - depol) * fid_1 + c
                    qubits_fidelity[qubit_2] = np.sqrt(1 - depol) * fid_2 + c

        # Apply T1/T2 decay for this layer
        for i in range(num_qubits):
            t1_decay = np.exp(-slowest_gate / t1_times[i])
            t2_decay = 0.5 * np.exp(-slowest_gate / t2_times[i]) + 0.5
            qubits_fidelity[i] *= t1_decay * t2_decay

    return np.prod(qubits_fidelity)


def compute_amfdn_fidelity(circ, n, epsilon, t2_over_tcx, t_cx=1e-6, p=0.5):
    """
    Compute AMFDN fidelity for given circuit and parameters.

    Args:
        circ: Qiskit circuit
        n: Number of qubits
        epsilon: CX gate error rate
        t2_over_tcx: Coherence ratio T2/t_cx
        t_cx: CX gate time (default 1us, used as reference)
        p: Entanglement parameter (0.5 = middle ground)

    Returns:
        Fidelity value
    """
    # Derive times from ratio
    T2 = t2_over_tcx * t_cx
    T1 = 2 * T2  # Typical T1/T2 ratio

    # Gate errors: CX error is epsilon, single-qubit errors are ~10x smaller
    error_params = {
        'cx': epsilon,
        'x': epsilon / 10,
        'h': epsilon / 10,
        'rz': epsilon / 100,
    }

    gate_times = {'cx': t_cx}
    t1_times = [T1] * n
    t2_times = [T2] * n

    return depolarizing_thermal_from_errors_dict(
        circ, error_params, n, gate_times, t1_times, t2_times, p=p
    )


def compute_heatmap_amfdn(n: int, eps_values: np.ndarray, ratio_values: np.ndarray):
    """
    Compute fidelity ratio heatmap and batch-2 fidelity using AMFDN model for given n.

    Returns:
        ratio_grid: 2D array of log10(F_batch2 / F_naive)
        batch2_fidelity_grid: 2D array of F_batch2
    """
    print(f"  Building circuits for n={n}...")
    naive_circ = build_naive_ladder(n)
    batch2_circ = build_batch2_ladder(n)

    n_eps = len(eps_values)
    n_ratio = len(ratio_values)
    ratio_grid = np.zeros((n_ratio, n_eps))
    batch2_fidelity_grid = np.zeros((n_ratio, n_eps))

    total_points = n_eps * n_ratio
    computed = 0

    print(f"  Computing {total_points} grid points...")

    for i, ratio_val in enumerate(ratio_values):
        for j, eps_val in enumerate(eps_values):
            # Compute AMFDN fidelity for both circuits
            F_naive = compute_amfdn_fidelity(naive_circ, n, eps_val, ratio_val)
            F_batch2 = compute_amfdn_fidelity(batch2_circ, n, eps_val, ratio_val)

            # Store batch-2 fidelity
            batch2_fidelity_grid[i, j] = F_batch2

            # Log10 ratio
            if F_naive > 0 and F_batch2 > 0:
                ratio_grid[i, j] = np.log10(F_batch2 / F_naive)
            else:
                ratio_grid[i, j] = 0

            computed += 1
            if computed % 500 == 0:
                print(f"    Progress: {computed}/{total_points} ({100*computed/total_points:.1f}%)")

    return ratio_grid, batch2_fidelity_grid


def main() -> int:
    parser = argparse.ArgumentParser(description="Heatmap: Fidelity ratio in parameter space (AMFDN model)")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent / "figures")
    parser.add_argument("--grid-size", type=int, default=50,
                        help="Grid resolution (default: 50, use smaller for faster computation)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Parameter grids (coarser for AMFDN since it's computationally intensive)
    n_grid = args.grid_size
    eps_values = np.geomspace(1e-4, 1e-1, n_grid)
    ratio_values = np.geomspace(1e2, 1e8, n_grid)  # Start from 10^2

    # Two qubit counts
    n_values = [50, 100]

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Normalization for ratio heatmaps (top row) - narrower range for better granularity
    vmin_ratio, vmax_ratio = -2, 2
    norm_ratio = TwoSlopeNorm(vmin=vmin_ratio, vcenter=0, vmax=vmax_ratio)

    log_eps = np.log10(eps_values)
    log_ratio = np.log10(ratio_values)
    LOG_EPS, LOG_RATIO = np.meshgrid(log_eps, log_ratio)

    # Store data for both n values
    ratio_grids = []
    batch2_grids = []

    for idx, n in enumerate(n_values):
        print(f"\nComputing AMFDN heatmap for n={n}...")
        ratio_grid, batch2_grid = compute_heatmap_amfdn(n, eps_values, ratio_values)
        ratio_grids.append(ratio_grid)
        batch2_grids.append(batch2_grid)
        print(f"  Done. Ratio range: [{ratio_grid.min():.2f}, {ratio_grid.max():.2f}]")
        print(f"        Batch-2 F range: [{batch2_grid.min():.2e}, {batch2_grid.max():.2e}]")

    # ========== Top row: Ratio heatmaps ==========
    for idx, n in enumerate(n_values):
        ax = axes[0, idx]
        ratio_grid = ratio_grids[idx]

        # Plot heatmap using pcolormesh with better color distinction
        im_ratio = ax.pcolormesh(log_eps, log_ratio, ratio_grid,
                                 cmap='seismic', norm=norm_ratio, shading='gouraud')

        # Contour at break-even
        ax.contour(LOG_EPS, LOG_RATIO, ratio_grid, levels=[0.0],
                   colors='black', linewidths=2.5, linestyles='-')

        # Format axes
        ax.set_xlabel(r'CNOT Gate Error $\epsilon$', fontsize=12)
        ax.set_ylabel(r'Coherence Ratio $T_2 / t_{0}$', fontsize=12)

        xticks = [-4, -3, -2, -1]
        yticks = [2, 3, 4, 5, 6, 7, 8]
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([f'$10^{{{x}}}$' for x in xticks])
        ax.set_yticklabels([f'$10^{{{y}}}$' for y in yticks])

        ax.set_title(f'Fidelity Ratio ($n={n}$ qubits)', fontsize=13, fontweight='bold')

        # Annotations
        ax.text(-3.7, 2.5, 'Batch-2\nwins', fontsize=9, color='navy', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(-1.5, 7, 'Naive\nwins', fontsize=9, color='darkred', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ========== Bottom row: Batch-2 fidelity heatmaps ==========
    # Use linear scale for fidelity
    for idx, n in enumerate(n_values):
        ax = axes[1, idx]
        batch2_grid = batch2_grids[idx]

        # Plot heatmap using pcolormesh for pixel-perfect alignment
        im_fid = ax.pcolormesh(log_eps, log_ratio, batch2_grid,
                               cmap='viridis', vmin=0, vmax=1, shading='gouraud')

        # Contour lines at specific fidelity levels
        contour_levels = [0.1, 0.5, 0.7, 0.9]
        cs = ax.contour(LOG_EPS, LOG_RATIO, batch2_grid, levels=contour_levels,
                        colors='white', linewidths=1.5, linestyles='-')
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')

        # Format axes
        ax.set_xlabel(r'CNOT Gate Error $\epsilon$', fontsize=12)
        ax.set_ylabel(r'Coherence Ratio $T_2 / t_{0}$', fontsize=12)

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([f'$10^{{{x}}}$' for x in xticks])
        ax.set_yticklabels([f'$10^{{{y}}}$' for y in yticks])

        ax.set_title(f'Batch-2 Fidelity ($n={n}$ qubits)', fontsize=13, fontweight='bold')

    # ========== Colorbars ==========
    fig.subplots_adjust(left=0.06, right=0.85, top=0.92, bottom=0.06, wspace=0.20, hspace=0.25)

    # Colorbar for ratio (top row)
    cbar_ax_ratio = fig.add_axes([0.87, 0.54, 0.02, 0.36])
    cbar_ratio = fig.colorbar(im_ratio, cax=cbar_ax_ratio)
    cbar_ratio.set_ticks(np.arange(-2, 2.5, 0.5))
    cbar_ratio.set_label(r'$\log_{10}(F_{\mathrm{batch\text{-}2}}/F_{\mathrm{naive}})$', fontsize=12)

    # Colorbar for fidelity (bottom row)
    cbar_ax_fid = fig.add_axes([0.87, 0.06, 0.02, 0.36])
    cbar_fid = fig.colorbar(im_fid, cax=cbar_ax_fid)
    cbar_fid.set_label(r'$F_{\mathrm{batch\text{-}2}}$', fontsize=12)

    fig.savefig(args.out_dir / 'fig_heatmap_n50_n100.pdf')
    fig.savefig(args.out_dir / 'fig_heatmap_n50_n100.png', dpi=150)
    plt.close(fig)

    print(f"\nWrote: {args.out_dir / 'fig_heatmap_n50_n100.png'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
