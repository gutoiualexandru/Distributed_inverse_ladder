"""
CNOT inverse ladder circuit builders (Algorithm 1: DILadder).

Implements the recursive decomposition of the CNOT inverse ladder
operator IL_1^(n) as described in the paper, providing Qiskit
QuantumCircuit objects for fidelity estimation.

Usage:
    from ladder_circuits import build_naive_ladder, build_batch2_ladder

    naive_circ = build_naive_ladder(n_qubits=100)
    batch2_circ = build_batch2_ladder(n_qubits=100)
"""

from __future__ import annotations

from typing import List, Sequence, Tuple
from functools import lru_cache

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Qubit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


def _check_qiskit():
    """Raise ImportError if Qiskit is not available."""
    if not QISKIT_AVAILABLE:
        raise ImportError(
            "Qiskit is required for ladder circuit construction. "
            "Install with: pip install qiskit"
        )


# ---------------------------------------------------------------------------
# Algorithm 1: DILadder  (paper §3, Algorithm 1)
# ---------------------------------------------------------------------------

def ladder_naive(
    circuit: 'QuantumCircuit',
    qubits: Sequence[int | 'Qubit'],
) -> None:
    """Apply the naive CNOT inverse ladder on *qubits* (LadderNaive)."""
    for control, target in zip(qubits, qubits[1:]):
        circuit.cx(control, target)


def choose_block_size(n: int, chunk_size: int | None) -> int:
    """ChooseBlockSize(n, chunk_size) from the paper."""
    if chunk_size is not None:
        candidate = max(2, int(chunk_size))
    else:
        candidate = max(2, round((n + 2) / 4))
    return min(candidate, n - 1)


def partition_blocks(
    qubits: Sequence[int | 'Qubit'], block: int
) -> tuple[list[list[int | 'Qubit']], list[int | 'Qubit']]:
    """PartitionBlocks(qubits, block) from the paper."""
    if block < 2 or block >= len(qubits):
        return [], list(qubits)
    full = len(qubits) // block
    blocks = [list(qubits[i * block : (i + 1) * block]) for i in range(full)]
    remainder = list(qubits[full * block :])
    return blocks, remainder


def di_ladder(
    circuit: 'QuantumCircuit',
    qubits: Sequence[int | 'Qubit'],
    *,
    chunk_size: int | None = None,
) -> None:
    """
    DILadder(circuit, qubits, chunk_size)  --  Algorithm 1 from the paper.

    Applies the CNOT inverse ladder IL_1^(n) on *qubits* via recursive
    decomposition with factor *chunk_size*.
    """
    ordered = list(qubits)
    n = len(ordered)
    if n <= 1:
        return
    if n <= max(4, chunk_size):
        ladder_naive(circuit, ordered)
        return

    block = choose_block_size(n, chunk_size)
    blocks, remainder = partition_blocks(ordered, block)
    if len(blocks) < 2:
        ladder_naive(circuit, ordered)
        return

    # Step 1: apply DILadder on each block
    for chunk in blocks:
        di_ladder(circuit, chunk, chunk_size=chunk_size)

    # Step 2: apply DILadder on representatives (last qubit of each block)
    new_qubits = [blk[-1] for blk in blocks]
    di_ladder(circuit, new_qubits, chunk_size=chunk_size)

    # Step 3: fan-out CNOTs from last qubit of blocks[i-1] to all-but-last of blocks[i]
    for idx in range(1, len(blocks)):
        control = blocks[idx - 1][-1]
        for target in blocks[idx][:-1]:
            circuit.cx(control, target)

    # Step 4: handle remainder
    if remainder:
        circuit.cx(blocks[-1][-1], remainder[0])
        di_ladder(circuit, remainder, chunk_size=chunk_size)


# ---------------------------------------------------------------------------
# Convenience builders (used by figure scripts)
# ---------------------------------------------------------------------------

def build_naive_ladder(n_qubits: int) -> 'QuantumCircuit':
    """Build a naive CNOT inverse ladder circuit (depth n-1, CNOT count n-1)."""
    _check_qiskit()
    if n_qubits < 2:
        raise ValueError("Ladder requires at least 2 qubits")
    circ = QuantumCircuit(n_qubits)
    ladder_naive(circ, list(range(n_qubits)))
    return circ


def build_batch_ladder(n_qubits: int, chunk_size: int = 2) -> 'QuantumCircuit':
    """Build a recursive CNOT inverse ladder using DILadder (Algorithm 1)."""
    _check_qiskit()
    if n_qubits < 2:
        raise ValueError("Ladder requires at least 2 qubits")
    circ = QuantumCircuit(n_qubits)
    di_ladder(circ, list(range(n_qubits)), chunk_size=chunk_size)
    return circ


def build_batch2_ladder(n_qubits: int) -> 'QuantumCircuit':
    """Convenience: DILadder with chunk_size=2."""
    return build_batch_ladder(n_qubits, chunk_size=2)


# ---------------------------------------------------------------------------
# Metrics (analytical, no circuit construction needed)
# ---------------------------------------------------------------------------

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


def get_ladder_metrics(n: int, ladder_type: str = 'naive') -> Tuple[int, int]:
    """Get (depth, n_cx) for a ladder circuit."""
    if ladder_type == 'naive':
        return (n - 1, n - 1)
    elif ladder_type.startswith('batch'):
        chunk_size = int(ladder_type[5:]) if len(ladder_type) > 5 else 2
        return (
            ladder_depth_recursive(n, chunk_size),
            ladder_cx_recursive(n, chunk_size)
        )
    else:
        raise ValueError(f"Unknown ladder type: {ladder_type}")
