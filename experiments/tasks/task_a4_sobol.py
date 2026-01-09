"""
Task A4: Sobol Probes

Generate Sobol low-discrepancy probes and compare with continuous probes.
Creates phase heatmaps and histograms showing phase distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from experiments.probe_generators import generate_probe_bank_sobol, generate_probe_bank_continuous
from experiments.diversity_analysis import compute_diversity_metrics


def run_task_a4(N: int = 32, K: int = 64, M: int = 8, seed: int = 42,
                results_dir: str = "results/A4_sobol_probes", verbose: bool = True) -> Dict:
    """
    Run Task A4: Sobol Probe Analysis.

    Args:
        N: Number of RIS elements
        K: Number of probes
        M: Sensing budget (not directly used in this task)
        seed: Random seed
        results_dir: Directory to save results
        verbose: Whether to print progress

    Returns:
        Dictionary with results
    """
    if verbose:
        print("\n" + "="*70)
        print("Task A4: Sobol Probes")
        print("="*70)

    # Create results directories
    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Generate Sobol probes
    if verbose:
        print(f"Generating {K} Sobol probes with N={N} elements...")
    sobol_bank = generate_probe_bank_sobol(N, K, seed)

    # Generate continuous probes for comparison
    continuous_bank = generate_probe_bank_continuous(N, K, seed)

    # Compute diversity metrics
    if verbose:
        print("Computing diversity metrics...")
    sobol_metrics = compute_diversity_metrics(sobol_bank)
    continuous_metrics = compute_diversity_metrics(continuous_bank)

    # Create phase heatmap comparison
    if verbose:
        print("Creating phase heatmap...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Sobol heatmap
    ax = axes[0]
    sns.heatmap(sobol_bank.phases / np.pi, ax=ax, cmap='RdYlBu_r',
                cbar_kws={'label': 'Phase (π)'}, vmin=0, vmax=2)
    ax.set_xlabel('RIS Element Index')
    ax.set_ylabel('Probe Index')
    ax.set_title('Sobol Probes (low-discrepancy)')

    # Continuous heatmap
    ax = axes[1]
    sns.heatmap(continuous_bank.phases / np.pi, ax=ax, cmap='RdYlBu_r',
                cbar_kws={'label': 'Phase (π)'}, vmin=0, vmax=2)
    ax.set_xlabel('RIS Element Index')
    ax.set_ylabel('Probe Index')
    ax.set_title('Continuous Probes (random)')

    plt.tight_layout()
    heatmap_path = os.path.join(plots_dir, "phase_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"  Saved: {heatmap_path}")
    plt.close()

    # Create phase histogram
    if verbose:
        print("Creating phase histogram...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sobol histogram
    ax = axes[0]
    sobol_phases_flat = sobol_bank.phases.flatten()
    ax.hist(sobol_phases_flat / np.pi, bins=50, color='steelblue',
            edgecolor='black', alpha=0.7)
    ax.set_xlabel('Phase (π)')
    ax.set_ylabel('Count')
    ax.set_title('Sobol Probe Phase Distribution')
    ax.grid(True, alpha=0.3)

    # Continuous histogram
    ax = axes[1]
    continuous_phases_flat = continuous_bank.phases.flatten()
    ax.hist(continuous_phases_flat / np.pi, bins=50, color='coral',
            edgecolor='black', alpha=0.7)
    ax.set_xlabel('Phase (π)')
    ax.set_ylabel('Count')
    ax.set_title('Continuous Probe Phase Distribution')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    histogram_path = os.path.join(plots_dir, "phase_histogram.png")
    plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"  Saved: {histogram_path}")
    plt.close()

    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Task A4: Sobol Probes Results\n")
        f.write("="*70 + "\n\n")
        f.write("Configuration:\n")
        f.write(f"  N (RIS elements): {N}\n")
        f.write(f"  K (probes): {K}\n")
        f.write(f"  Seed: {seed}\n\n")

        f.write("Sobol Probe Diversity Metrics:\n")
        for key, val in sobol_metrics.items():
            f.write(f"  {key}: {val}\n")

        f.write("\nContinuous Probe Diversity Metrics (for comparison):\n")
        for key, val in continuous_metrics.items():
            f.write(f"  {key}: {val}\n")

    if verbose:
        print(f"  Saved: {metrics_path}")
        print("\nSobol Probe Diversity:")
        print(f"  Metric: {sobol_metrics['metric_type']}")
        print(f"  Mean: {sobol_metrics['mean']:.4f}")
        print(f"  Std: {sobol_metrics['std']:.4f}")
        print(f"  Range: [{sobol_metrics['min']:.4f}, {sobol_metrics['max']:.4f}]")

    return {
        'sobol_bank': sobol_bank,
        'continuous_bank': continuous_bank,
        'sobol_metrics': sobol_metrics,
        'continuous_metrics': continuous_metrics,
        'plots': [heatmap_path, histogram_path],
        'metrics_file': metrics_path
    }
