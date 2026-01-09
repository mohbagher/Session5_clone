"""
Task D1: Seed Variation

Train with multiple seeds to measure performance stability.
Creates a boxplot of eta values and reports mean/std.
"""

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import get_config
from data_generation import create_dataloaders
from evaluation import evaluate_model
from experiments.probe_generators import get_probe_bank
from model import create_model
from training import train


def run_task_d1(N: int = 32, K: int = 64, M: int = 8, seed: int = 42,
                results_dir: str = "results/D1_seed_variation", verbose: bool = True) -> Dict:
    """
    Run Task D1: Seed Variation.

    Args:
        N: Number of RIS elements
        K: Number of probes
        M: Sensing budget
        seed: Base random seed
        results_dir: Directory to save results
        verbose: Whether to print progress

    Returns:
        Dictionary with results
    """
    if verbose:
        print("\n" + "="*70)
        print("Task D1: Seed Variation")
        print("="*70)

    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    seeds: List[int] = [1, 2, 3, 4, 5]
    eta_values = []

    for run_seed in seeds:
        if verbose:
            print(f"\n--- Seed = {run_seed} ---")

        config = get_config(
            system={'N': N, 'K': K, 'M': M},
            data={'n_train': 20000, 'n_val': 2000, 'n_test': 2000, 'seed': run_seed},
            training={'n_epochs': 50, 'batch_size': 128}
        )

        probe_bank = get_probe_bank('continuous', N, K, run_seed)
        train_loader, val_loader, test_loader, metadata = create_dataloaders(config, probe_bank)

        model = create_model(config)
        model, history = train(model, train_loader, val_loader, config, metadata)

        eval_results = evaluate_model(
            model, test_loader, config,
            metadata['test_powers_full'],
            metadata['test_labels'],
            metadata['test_observed_indices'],
            metadata['test_optimal_powers']
        )
        eta_values.append(eval_results.eta_top1)

        if verbose:
            print(f"  η_top1: {eval_results.eta_top1:.4f}")

    # Plot boxplot
    if verbose:
        print("\nCreating seed variation plot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(eta_values, vert=True, patch_artist=True,
               boxprops=dict(facecolor='steelblue', color='black'),
               medianprops=dict(color='yellow'))
    ax.scatter(np.ones(len(eta_values)), eta_values, color='black', zorder=3)
    ax.set_ylabel('η (Power Ratio)')
    ax.set_title(f'Seed Variation (N={N}, K={K}, M={M})')
    ax.set_xticks([1])
    ax.set_xticklabels(['η_top1'])
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    boxplot_path = os.path.join(plots_dir, "eta_seed_boxplot.png")
    plt.savefig(boxplot_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"  Saved: {boxplot_path}")
    plt.close()

    metrics_path = os.path.join(results_dir, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Task D1: Seed Variation Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  N (RIS elements): {N}\n")
        f.write(f"  K (probes): {K}\n")
        f.write(f"  M (sensing budget): {M}\n")
        f.write(f"  Seeds: {seeds}\n\n")
        f.write("η_top1 values:\n")
        for run_seed, eta in zip(seeds, eta_values):
            f.write(f"  Seed {run_seed}: η = {eta:.4f}\n")
        f.write("\nSummary:\n")
        f.write(f"  Mean η: {np.mean(eta_values):.4f}\n")
        f.write(f"  Std η:  {np.std(eta_values):.4f}\n")

    if verbose:
        print(f"  Saved: {metrics_path}")

    return {
        'seeds': seeds,
        'eta_values': eta_values,
        'plots': [boxplot_path],
        'metrics_file': metrics_path
    }
