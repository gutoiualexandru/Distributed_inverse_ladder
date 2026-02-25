"""
Shared utilities for publication-quality plots.

Contains ladder metrics functions and matplotlib styling used by
the figure-generation scripts.
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import numpy as np
import matplotlib.pyplot as plt


class LadderMetrics(NamedTuple):
    """Circuit metrics for a ladder implementation."""
    n_qubits: int
    depth: int
    n_cx: int


@lru_cache(maxsize=None)
def ladder_depth_recursive(n: int, chunk_size: int) -> int:
    """Compute circuit depth for recursive ladder decomposition."""
    if n <= 1:
        return 0
    k = max(2, int(chunk_size))
    if n <= max(4, k):
        return n - 1
    blocks = n // k
    if blocks < 2:
        return n - 1
    return (k - 1) + ladder_depth_recursive(blocks, k) + (k - 1)


@lru_cache(maxsize=None)
def ladder_cx_recursive(n: int, chunk_size: int) -> int:
    """Compute CNOT count for recursive ladder decomposition."""
    if n <= 1:
        return 0
    k = max(2, int(chunk_size))
    if n <= max(4, k):
        return n - 1
    blocks = n // k
    remainder = n % k
    if blocks < 2:
        return n - 1

    total = blocks * (k - 1) + ladder_cx_recursive(blocks, k) + (blocks - 1) * (k - 1)
    if remainder:
        total += 1 + ladder_cx_recursive(remainder, k)
    return total


def get_naive_metrics(n: int) -> LadderMetrics:
    """Get metrics for naive ladder implementation."""
    return LadderMetrics(n_qubits=n, depth=n - 1, n_cx=n - 1)


def get_batch2_metrics(n: int) -> LadderMetrics:
    """Get metrics for batch-2 (chunk_size=2) ladder implementation."""
    return LadderMetrics(
        n_qubits=n,
        depth=ladder_depth_recursive(n, chunk_size=2),
        n_cx=ladder_cx_recursive(n, chunk_size=2),
    )


def set_publication_style(use_latex: bool = False) -> None:
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'text.usetex': use_latex,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 1.5,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'mathtext.fontset': 'cm',
    })
