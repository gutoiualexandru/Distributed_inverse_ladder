# Distributed Ladder Circuit -- Figure Generation

Scripts for reproducing the figures in the paper.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Generate all figures at once:

```bash
python generate_all_figures.py
```

Or run individual scripts:

```bash
python depth_cnot_comparison.py
python fig_heatmap_only.py
python fig_crossover_ratio.py
python distributed_specs.py
```

Output PDFs and PNGs are saved to `figures/`.

## Script-to-Figure Mapping

| Script | Output | Description |
|---|---|---|
| `depth_cnot_comparison.py` | `depth_cnot_comparison.pdf` | Circuit depth and CNOT count comparison (naive vs batch decompositions) |
| `fig_heatmap_only.py` | `fig_heatmap_n50_n100.pdf` | Fidelity ratio heatmaps for n=50 and n=100 in (epsilon, T2/t0) parameter space |
| `fig_crossover_ratio.py` | `fig_crossover_ratio.pdf` | Crossover condition epsilon * T2/t0 vs number of qubits |
| `distributed_specs.py` | `resource_estimate.pdf` | Module requirements and transmission rounds for distributed architectures |

## Shared Modules

| Module | Purpose |
|---|---|
| `plot_utils.py` | Ladder metrics computation and matplotlib publication styling |
| `ladder_circuits.py` | Qiskit circuit builders for naive and recursive CNOT ladders |

## Notes

- `fig_heatmap_only.py` and `fig_crossover_ratio.py` are computationally intensive (may take several minutes). Use `--grid-size` or `--n-max` to reduce computation for testing.
- All scripts accept `--out-dir` to specify a custom output directory.
