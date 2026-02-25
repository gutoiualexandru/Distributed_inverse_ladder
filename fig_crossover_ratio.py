#!/usr/bin/env python3
"""
Plot of epsilon * T2/t_0 vs n characterizing the crossover point
where Batch-2 becomes better than Naive.

For each n, we find the T2/t_0 value where F_batch2 = F_naive for various
epsilon values, then compute and plot epsilon * T2/t_0 with error bars.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from qiskit.converters import circuit_to_dag, dag_to_circuit
from ladder_circuits import build_naive_ladder, build_batch2_ladder


def depolarizing_thermal_from_errors_dict(circ, error_params, num_qubits, gate_times, t1_times, t2_times, p=0.5):
    """
    Depolarizing model with T1/T2 thermal decay applied layer-by-layer.
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


def compute_fidelity_ratio(naive_circ, batch2_circ, n, epsilon, t2_over_tcx, t_cx=1e-6, p=0.5):
    """
    Compute the ratio F_batch2 / F_naive - 1 (so crossover is at 0).
    """
    T2 = t2_over_tcx * t_cx
    T1 = 2 * T2

    error_params = {
        'cx': epsilon,
        'x': epsilon / 10,
        'h': epsilon / 10,
        'rz': epsilon / 100,
    }

    gate_times = {'cx': t_cx}
    t1_times = [T1] * n
    t2_times = [T2] * n

    F_naive = depolarizing_thermal_from_errors_dict(
        naive_circ, error_params, n, gate_times, t1_times, t2_times, p=p
    )
    F_batch2 = depolarizing_thermal_from_errors_dict(
        batch2_circ, error_params, n, gate_times, t1_times, t2_times, p=p
    )

    if F_naive > 0:
        return F_batch2 / F_naive - 1.0
    return 0.0


def find_crossover_t2_ratio(naive_circ, batch2_circ, n, epsilon, p=0.5, ratio_min=1e2, ratio_max=1e8):
    """
    Find the T2/t_0 value where Batch-2 becomes better than Naive.

    Returns None if no crossover exists in the given range.
    """
    def objective(log_ratio):
        t2_ratio = 10**log_ratio
        return compute_fidelity_ratio(naive_circ, batch2_circ, n, epsilon, t2_ratio, p=p)

    log_min = np.log10(ratio_min)
    log_max = np.log10(ratio_max)

    # Check if crossover exists
    val_min = objective(log_min)
    val_max = objective(log_max)

    # At low T2/t_0 (high decoherence), Batch-2 should win (ratio > 0)
    # At high T2/t_0 (low decoherence), Naive should win (ratio < 0)
    if val_min <= 0 or val_max >= 0:
        # No crossover in this range
        return None

    try:
        log_crossover = brentq(objective, log_min, log_max, xtol=1e-4)
        return 10**log_crossover
    except ValueError:
        return None


def is_marker_point(n, marker_step=25):
    """Check if n should have extra samples for error bars."""
    return n % marker_step == 0 or n == 5


def main() -> int:
    parser = argparse.ArgumentParser(description="Crossover ratio epsilon*T2/t_0 vs n")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent / "figures")
    parser.add_argument("--n-min", type=int, default=5, help="Minimum n")
    parser.add_argument("--n-max", type=int, default=200, help="Maximum n")
    parser.add_argument("--n-step", type=int, default=1, help="Step size for n")
    parser.add_argument("--marker-step", type=int, default=25, help="Step size for markers")
    parser.add_argument("--num-epsilon", type=int, default=10,
                        help="Number of epsilon values to sample")
    parser.add_argument("--num-p", type=int, default=20,
                        help="Number of p values for marker points")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Range of n values (compute for every point)
    n_values = list(range(args.n_min, args.n_max + 1, args.n_step))

    # Range of epsilon values (logarithmically spaced)
    epsilon_values = np.geomspace(1e-4, 1e-1, args.num_epsilon)
    p_values = np.linspace(0, 1, args.num_p)

    print(f"Computing crossover for n in [{args.n_min}, {args.n_max}], step {args.n_step}")
    print(f"Using {args.num_epsilon} epsilon values")
    print(f"Marker points (every {args.marker_step}) get {args.num_p} extra p samples")

    # Store results: for each n, collect epsilon * T2/t_0 values
    results = {n: [] for n in n_values}

    for n in n_values:
        is_marker = is_marker_point(n, args.marker_step)
        print(f"n = {n}{'*' if is_marker else ''}...", end=" ", flush=True)
        naive_circ = build_naive_ladder(n)
        batch2_circ = build_batch2_ladder(n)

        # Standard samples with p=0.5
        for eps in epsilon_values:
            t2_ratio = find_crossover_t2_ratio(naive_circ, batch2_circ, n, eps, p=0.5)
            if t2_ratio is not None:
                results[n].append(eps * t2_ratio)

        # Extra samples at marker points: vary p from 0 to 1
        if is_marker:
            for p in p_values:
                for eps in epsilon_values:
                    t2_ratio = find_crossover_t2_ratio(naive_circ, batch2_circ, n, eps, p=p)
                    if t2_ratio is not None:
                        results[n].append(eps * t2_ratio)

        print(f"found {len(results[n])} crossovers")

    # Compute statistics for each n
    n_with_data = []
    means = []
    stds = []
    mins = []
    maxs = []

    for n in n_values:
        if len(results[n]) >= 2:  # Need at least 2 points for meaningful stats
            data = np.array(results[n])
            n_with_data.append(n)
            means.append(np.mean(data))
            stds.append(np.std(data))
            mins.append(np.min(data))
            maxs.append(np.max(data))

    n_with_data = np.array(n_with_data)
    means = np.array(means)
    stds = np.array(stds)
    mins = np.array(mins)
    maxs = np.array(maxs)

    # Save the data
    np.savez(args.out_dir / 'crossover_data.npz',
             n_values=n_with_data,
             means=means,
             stds=stds,
             mins=mins,
             maxs=maxs)
    print(f"Wrote: {args.out_dir / 'crossover_data.npz'}")

    # ---- Publication-quality plot ----
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'cm',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'lines.linewidth': 1.2,
        'lines.markersize': 5,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'axes.grid': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

    # Linear fit
    slope, intercept = np.polyfit(n_with_data, means, 1)
    fit_line = slope * n_with_data + intercept

    ss_res = np.sum((means - fit_line) ** 2)
    ss_tot = np.sum((means - np.mean(means)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f"\nLinear fit: y = {slope:.4f} * n + {intercept:.4f}")
    print(f"R² = {r_squared:.6f}")

    # Colors
    color_data = '#2E5A87'
    color_fit = '#C44E52'
    color_batch2 = '#4A90A4'
    color_naive = '#D4A574'

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    # Determine marker positions
    marker_indices = [i for i, n in enumerate(n_with_data)
                      if n % args.marker_step == 0 or n == n_with_data[0] or n == n_with_data[-1]]

    # Fill regions above and below the crossover line
    ax.fill_between(n_with_data, 0, means, alpha=0.2, color=color_batch2,
                    edgecolor='none')
    ax.fill_between(n_with_data, means, 195, alpha=0.2, color=color_naive,
                    edgecolor='none')

    # Shaded uncertainty region at marker points (interpolated)
    marker_n = n_with_data[marker_indices]
    marker_stds = stds[marker_indices]
    upper = np.interp(n_with_data, marker_n, means[marker_indices] + marker_stds)
    lower = np.interp(n_with_data, marker_n, means[marker_indices] - marker_stds)
    ax.fill_between(n_with_data, lower, upper, alpha=0.35, color=color_data,
                    edgecolor='none', label='Uncertainty')

    # Data line
    ax.plot(n_with_data, means, '-', color=color_data, linewidth=1.2, label='Simulation', zorder=3)

    # Linear fit
    ax.plot(n_with_data, fit_line, '-', color=color_fit, linewidth=1.8,
            label=f'Linear fit: $y = {slope:.2f}n {intercept:+.1f}$', zorder=2)

    # Error bars at marker positions
    marker_means = means[marker_indices]
    ax.errorbar(marker_n, marker_means, yerr=marker_stds,
                fmt='o', markersize=4, color=color_data,
                ecolor=color_data, elinewidth=1, capsize=2, capthick=1,
                markerfacecolor='white', markeredgewidth=1.2, zorder=4)

    # Axis labels
    ax.set_xlabel(r'Number of qubits $n$')
    ax.set_ylabel(r'$\varepsilon \cdot T_2 / t_0$ at crossover')

    # Axis limits and ticks
    ax.set_xlim(0, 210)
    ax.set_ylim(0, 195)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_yticks([0, 50, 100, 150])

    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax.tick_params(which='minor', length=2, direction='in')

    # Region labels
    ax.text(150, 40, 'Batch-2 wins', fontsize=10, color='#2A6070',
            fontweight='bold', ha='center', va='center')
    ax.text(150, 160, 'Naive wins', fontsize=10, color='#8B6914',
            fontweight='bold', ha='center', va='center')

    # Legend
    legend = ax.legend(loc='upper left', frameon=True, framealpha=0.95,
                       edgecolor='none', fancybox=False)
    legend.get_frame().set_linewidth(0)

    fig.tight_layout()

    fig.savefig(args.out_dir / 'fig_crossover_ratio.pdf')
    fig.savefig(args.out_dir / 'fig_crossover_ratio.png', dpi=300)
    plt.close(fig)

    print(f"\nWrote: {args.out_dir / 'fig_crossover_ratio.pdf'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
