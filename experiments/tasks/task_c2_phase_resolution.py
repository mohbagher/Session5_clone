"""
Task C2: Phase Resolution

Compare continuous, 1-bit, 2-bit, and Hadamard probes across sensing budgets.
Generates eta vs M curves and a performance-per-bit comparison at a fixed M.
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


def run_task_c2(N: int = 32, K: int = 64, M: int = 8, seed: int = 42,
                results_dir: str = "results/C2_phase_resolution", verbose: bool = True) -> Dict:
    """
    Run Task C2: Phase Resolution comparison.

    Args:
        N: Number of RIS elements
        K: Number of probes
        M: Sensing budget
        seed: Random seed
        results_dir: Directory to save results
        verbose: Whether to print progress

    Returns:
        Dictionary with results
    """
    if verbose:
        print("\n" + "="*70)
        print("Task C2: Phase Resolution")
        print("="*70)

    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    probe_types = ['continuous', 'binary', '2bit', 'hadamard']
    type_labels = {
        'continuous': 'Continuous',
        'binary': '1-bit',
        '2bit': '2-bit',
        'hadamard': 'Hadamard'
    }
    bits_map = {
        'continuous': '∞',
        'binary': '1',
        '2bit': '2',
        'hadamard': '1'
    }

    M_values: List[int] = [2, 4, 8, 16, 32]
    M_values = [m for m in M_values if m <= K]

    results = {
        'M_values': M_values,
        'probe_types': probe_types,
        'eta_results': {}
    }

    for probe_type in probe_types:
        if verbose:
            print(f"\n{'='*70}")
            print(f"Testing probe type: {probe_type}")
            print(f"{'='*70}")

        eta_values = []
        for M_test in M_values:
            if verbose:
                print(f"\n--- M = {M_test} ---")

            config = get_config(
                system={'N': N, 'K': K, 'M': M_test},
                data={'n_train': 20000, 'n_val': 2000, 'n_test': 2000, 'seed': seed},
                training={'n_epochs': 50, 'batch_size': 128}
            )

            probe_bank = get_probe_bank(probe_type, N, K, seed)
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

        results['eta_results'][probe_type] = eta_values

    # Plot eta vs M for each probe type
    if verbose:
        print("\nCreating eta vs M plot...")
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'continuous': 'steelblue',
        'binary': 'coral',
        '2bit': 'green',
        'hadamard': 'gold'
    }
    markers = {
        'continuous': 'o',
        'binary': 's',
        '2bit': '^',
        'hadamard': 'D'
    }

    for probe_type in probe_types:
        eta_vals = results['eta_results'][probe_type]
        ax.plot(M_values, eta_vals, marker=markers[probe_type],
                color=colors[probe_type], linewidth=2, markersize=8,
                label=type_labels[probe_type])

    ax.set_xlabel('Sensing Budget M')
    ax.set_ylabel('η (Power Ratio)')
    ax.set_title(f'Phase Resolution Comparison (N={N}, K={K})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    eta_vs_m_path = os.path.join(plots_dir, "eta_vs_M.png")
    plt.savefig(eta_vs_m_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"  Saved: {eta_vs_m_path}")
    plt.close()

    # Performance per control bit at a fixed M
    if verbose:
        print("Creating performance vs control bits plot...")
    reference_M = M if M in M_values else M_values[min(len(M_values) - 1, 0)]
    ref_index = M_values.index(reference_M)

    fig, ax = plt.subplots(figsize=(10, 6))
    perf_values = [results['eta_results'][p][ref_index] for p in probe_types]
    labels = [f"{type_labels[p]}\n({bits_map[p]} bit)" for p in probe_types]
    bars = ax.bar(labels, perf_values, color=[colors[p] for p in probe_types],
                  edgecolor='black', alpha=0.85)
    ax.set_ylabel('η (Power Ratio)')
    ax.set_title(f'Performance vs Control Bits (M={reference_M}, K={K})')
    ax.set_ylim([0, 1.1])
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, perf_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    perf_bits_path = os.path.join(plots_dir, "eta_vs_control_bits.png")
    plt.savefig(perf_bits_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"  Saved: {perf_bits_path}")
    plt.close()

    metrics_path = os.path.join(results_dir, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Task C2: Phase Resolution Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  N (RIS elements): {N}\n")
        f.write(f"  K (probes): {K}\n")
        f.write(f"  M values tested: {M_values}\n")
        f.write(f"  Seed: {seed}\n\n")
        f.write("Results (η_top1 for each probe type):\n\n")
        for probe_type in probe_types:
            f.write(f"{type_labels[probe_type]}:\n")
            eta_vals = results['eta_results'][probe_type]
            for m_val, eta in zip(M_values, eta_vals):
                f.write(f"  M={m_val:2d}: η = {eta:.4f}\n")
            f.write("\n")

        f.write("Control Bits Reference (at M = ")
        f.write(f"{reference_M}):\n")
        for probe_type, eta in zip(probe_types, perf_values):
            f.write(f"  {type_labels[probe_type]} ({bits_map[probe_type]} bit): η = {eta:.4f}\n")

    if verbose:
        print(f"  Saved: {metrics_path}")

    results['plots'] = [eta_vs_m_path, perf_bits_path]
    results['metrics_file'] = metrics_path
    results['reference_M'] = reference_M
    results['reference_eta'] = dict(zip(probe_types, perf_values))

    return results
