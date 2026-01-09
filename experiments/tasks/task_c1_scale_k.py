"""
Task C1: Scale K

Evaluate how performance changes as the probe bank size K increases.
Generates eta vs K and eta vs M/K plots.
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


def run_task_c1(N: int = 32, K: int = 64, M: int = 8, seed: int = 42,
                results_dir: str = "results/C1_scale_k", verbose: bool = True) -> Dict:
    """
    Run Task C1: Scale K study.

    Tests K = 32, 64, 128, 256 with a fixed sensing budget M
    (clipped to K when K < M).

    Args:
        N: Number of RIS elements
        K: Default number of probes (not used directly; list is fixed)
        M: Sensing budget
        seed: Random seed
        results_dir: Directory to save results
        verbose: Whether to print progress

    Returns:
        Dictionary with results
    """
    if verbose:
        print("\n" + "="*70)
        print("Task C1: Scale K")
        print("="*70)

    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    K_values: List[int] = [32, 64, 128, 256]
    eta_values = []
    ratio_values = []

    for K_test in K_values:
        M_test = min(M, K_test)
        ratio_values.append(M_test / K_test)

        if verbose:
            print(f"\n--- K = {K_test} (M = {M_test}) ---")

        config = get_config(
            system={'N': N, 'K': K_test, 'M': M_test},
            data={'n_train': 20000, 'n_val': 2000, 'n_test': 2000, 'seed': seed},
            training={'n_epochs': 50, 'batch_size': 128}
        )

        probe_bank = get_probe_bank('continuous', N, K_test, seed)
        train_loader, val_loader, test_loader, metadata = create_dataloaders(config, probe_bank)

        model = create_model(config)
        model, history = train(model, train_loader, val_loader, config, metadata)

        results = evaluate_model(
            model, test_loader, config,
            metadata['test_powers_full'],
            metadata['test_labels'],
            metadata['test_observed_indices'],
            metadata['test_optimal_powers']
        )
        eta_values.append(results.eta_top1)

        if verbose:
            print(f"  η_top1: {results.eta_top1:.4f}")

    # Plot eta vs K
    if verbose:
        print("\nCreating eta vs K plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(K_values, eta_values, marker='o', linewidth=2, color='steelblue')
    ax.set_xlabel('Probe Bank Size K')
    ax.set_ylabel('η (Power Ratio)')
    ax.set_title(f'Performance vs Probe Bank Size (N={N}, M={M})')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    for k_val, eta in zip(K_values, eta_values):
        ax.text(k_val, eta + 0.02, f'{eta:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    eta_vs_k_path = os.path.join(plots_dir, "eta_vs_K.png")
    plt.savefig(eta_vs_k_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"  Saved: {eta_vs_k_path}")
    plt.close()

    # Plot eta vs M/K ratio
    if verbose:
        print("Creating eta vs M/K plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ratio_values, eta_values, marker='s', linewidth=2, color='coral')
    ax.set_xlabel('M/K Ratio')
    ax.set_ylabel('η (Power Ratio)')
    ax.set_title('Performance vs Measurement Ratio')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    for ratio, eta in zip(ratio_values, eta_values):
        ax.text(ratio, eta + 0.02, f'{eta:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    eta_vs_ratio_path = os.path.join(plots_dir, "eta_vs_M_over_K.png")
    plt.savefig(eta_vs_ratio_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"  Saved: {eta_vs_ratio_path}")
    plt.close()

    metrics_path = os.path.join(results_dir, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Task C1: Scale K Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  N (RIS elements): {N}\n")
        f.write(f"  M (sensing budget): {M}\n")
        f.write(f"  K values tested: {K_values}\n")
        f.write(f"  Seed: {seed}\n\n")
        f.write("Results (η_top1 for each K):\n")
        for k_val, ratio, eta in zip(K_values, ratio_values, eta_values):
            f.write(f"  K={k_val:3d} (M/K={ratio:.2%}): η = {eta:.4f}\n")

    if verbose:
        print(f"  Saved: {metrics_path}")

    return {
        'K_values': K_values,
        'M_values': [min(M, k) for k in K_values],
        'eta_values': eta_values,
        'plots': [eta_vs_k_path, eta_vs_ratio_path],
        'metrics_file': metrics_path
    }
