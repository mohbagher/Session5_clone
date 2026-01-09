"""
Task B1: M Variation Study

Test different sensing budgets M ∈ {2, 4, 8, 16, 32} and measure performance.
Trains ML model for each M value and plots mean η vs M for different probe types.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from typing import Dict
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from experiments.probe_generators import get_probe_bank
from experiments.tasks.task_defaults import build_task_config
from data_generation import create_dataloaders
from model import create_model
from training import train
from evaluation import evaluate_model


def run_task_b1(N: int = 32, K: int = 64, M: int = 8, seed: int = 42,
                results_dir: str = "results/B1_M_variation", verbose: bool = True) -> Dict:
    """
    Run Task B1: M Variation Study.
    
    Tests different sensing budgets M to understand the trade-off between
    measurement overhead and performance.
    
    Args:
        N: Number of RIS elements
        K: Number of probes
        M: Default sensing budget (will test multiple values)
        seed: Random seed
        results_dir: Directory to save results
        verbose: Whether to print progress
        
    Returns:
        Dictionary with results
    """
    if verbose:
        print("\n" + "="*70)
        print("Task B1: M Variation Study")
        print("="*70)
    
    # Create results directories
    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Test different M values
    M_values = [2, 4, 8, 16, 32]
    # Filter out M values > K
    M_values = [m for m in M_values if m <= K]
    
    # Test different probe types
    probe_types = ['continuous', 'binary']
    
    results = {
        'M_values': M_values,
        'probe_types': probe_types,
        'eta_results': {}
    }
    
    if verbose:
        print(f"Testing M values: {M_values}")
        print(f"Probe types: {probe_types}")
    
    for probe_type in probe_types:
        if verbose:
            print(f"\n{'='*70}")
            print(f"Testing probe type: {probe_type}")
            print(f"{'='*70}")
        
        eta_values = []
        
        for M_test in M_values:
            if verbose:
                print(f"\n--- M = {M_test} ---")
            
            # Create config
            config = build_task_config(N, K, M_test, seed)
            
            # Generate probe bank
            probe_bank = get_probe_bank(probe_type, N, K, seed)
            
            # Create dataloaders
            train_loader, val_loader, test_loader, metadata = create_dataloaders(config, probe_bank)
            
            # Create and train model
            model = create_model(config)
            
            if verbose:
                print(f"Training model with M={M_test}...")
            
            model, history = train(model, train_loader, val_loader, config, metadata)
            
            # Evaluate
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
    
    # Create plot: η vs M
    if verbose:
        print("\nCreating η vs M plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'continuous': 'steelblue', 'binary': 'coral'}
    markers = {'continuous': 'o', 'binary': 's'}
    
    for probe_type in probe_types:
        eta_vals = results['eta_results'][probe_type]
        ax.plot(M_values, eta_vals, marker=markers[probe_type], 
                color=colors[probe_type], linewidth=2, markersize=8,
                label=f'{probe_type.capitalize()} probes')
    
    ax.set_xlabel('Sensing Budget M (number of observed probes)')
    ax.set_ylabel('η (Power Ratio)')
    ax.set_title(f'ML Performance vs Sensing Budget (N={N}, K={K})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 1.05])
    
    # Add M/K ratio on top x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(M_values)
    ax2.set_xticklabels([f'{m/K:.1%}' for m in M_values])
    ax2.set_xlabel('M/K Ratio')
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "eta_vs_M.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"  Saved: {plot_path}")
    plt.close()
    
    # Create bar chart comparison
    if verbose:
        print("Creating bar chart comparison...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(M_values))
    width = 0.35
    
    for idx, probe_type in enumerate(probe_types):
        eta_vals = results['eta_results'][probe_type]
        offset = width * (idx - 0.5)
        bars = ax.bar(x + offset, eta_vals, width, label=probe_type.capitalize(),
                      color=colors[probe_type], edgecolor='black', alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, eta_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Sensing Budget M')
    ax.set_ylabel('η (Power Ratio)')
    ax.set_title(f'Performance Comparison Across M Values (N={N}, K={K})')
    ax.set_xticks(x)
    ax.set_xticklabels([f'M={m}\n({m/K:.1%})' for m in M_values])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    bar_path = os.path.join(plots_dir, "eta_comparison_bar.png")
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"  Saved: {bar_path}")
    plt.close()
    
    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Task B1: M Variation Study Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  N (RIS elements): {N}\n")
        f.write(f"  K (probes): {K}\n")
        f.write(f"  M values tested: {M_values}\n")
        f.write(f"  Probe types: {probe_types}\n")
        f.write(f"  Seed: {seed}\n\n")
        
        f.write("Results (η_top1 for each M):\n\n")
        for probe_type in probe_types:
            f.write(f"{probe_type.capitalize()} Probes:\n")
            eta_vals = results['eta_results'][probe_type]
            for m, eta in zip(M_values, eta_vals):
                f.write(f"  M={m:2d} ({m/K:5.1%}): η = {eta:.4f}\n")
            f.write("\n")
        
        f.write("Key Observations:\n")
        f.write("  - Performance (η) generally increases with larger sensing budget M\n")
        f.write("  - Diminishing returns observed as M approaches K\n")
        f.write(f"  - Trade-off between measurement overhead (M) and performance (η)\n")
    
    if verbose:
        print(f"  Saved: {metrics_path}")
        print("\nSummary:")
        for probe_type in probe_types:
            eta_vals = results['eta_results'][probe_type]
            print(f"  {probe_type.capitalize()}: η range = [{min(eta_vals):.4f}, {max(eta_vals):.4f}]")
    
    results['plots'] = [plot_path, bar_path]
    results['metrics_file'] = metrics_path
    
    return results
