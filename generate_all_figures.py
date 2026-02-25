#!/usr/bin/env python3
"""
Generate all publication figures.

Usage:
    python generate_all_figures.py                    # Generate all figures
    python generate_all_figures.py --out-dir ./figs   # Custom output directory
    python generate_all_figures.py --scripts fig_heatmap depth  # Specific figures only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


FIGURE_SCRIPTS = [
    ("depth_cnot_comparison.py", "Circuit depth and CNOT count comparison"),
    ("fig_heatmap_only.py",      "Fidelity ratio heatmaps (n=50, n=100)"),
    ("fig_crossover_ratio.py",   "Crossover ratio vs n"),
    ("distributed_specs.py",     "Distributed resource estimates"),
]


def run_figure_script(script: str, script_dir: Path, out_dir: Path) -> int:
    """Run a single figure generation script. Returns exit code."""
    cmd = [sys.executable, str(script_dir / script), "--out-dir", str(out_dir)]

    print(f"\n{'=' * 60}")
    print(f"Running: {script}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, cwd=script_dir)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate all publication figures")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
        help="Output directory for figures (default: ./figures)",
    )
    parser.add_argument(
        "--scripts",
        nargs="+",
        default=None,
        help="Run only scripts matching these patterns (e.g., 'heatmap' 'depth')",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Filter scripts if specific ones requested
    scripts_to_run = FIGURE_SCRIPTS
    if args.scripts:
        scripts_to_run = [
            (s, desc) for s, desc in FIGURE_SCRIPTS
            if any(pattern in s for pattern in args.scripts)
        ]
        if not scripts_to_run:
            print(f"No matching scripts found for: {args.scripts}")
            print(f"Available: {[s for s, _ in FIGURE_SCRIPTS]}")
            return 1

    print("=" * 60)
    print("Figure Generator")
    print("=" * 60)
    print(f"Output dir: {args.out_dir}")
    print(f"Scripts:    {len(scripts_to_run)} figure(s)")
    for script, desc in scripts_to_run:
        print(f"  - {script}: {desc}")
    print("=" * 60)

    failed = []
    for script, desc in scripts_to_run:
        exit_code = run_figure_script(script, script_dir, args.out_dir)
        if exit_code != 0:
            failed.append(script)
            print(f"\n[ERROR] {script} failed with exit code {exit_code}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total:      {len(scripts_to_run)}")
    print(f"Successful: {len(scripts_to_run) - len(failed)}")
    print(f"Failed:     {len(failed)}")

    if failed:
        print(f"\nFailed scripts:")
        for script in failed:
            print(f"  - {script}")
        return 1

    print(f"\nAll figures generated in: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
