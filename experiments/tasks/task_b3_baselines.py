"""
Task B3: Baseline Comparison

Implement and compare baselines: Random 1/K, Random M/K, Best Observed.
Creates bar plot comparing ML vs all baselines.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from typing import Dict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from experiments.probe_generators import get_probe_bank
from experiments.tasks.task_defaults import build_task_config
from data_generation import create_dataloaders
from model import create_model
from training import train
from evaluation import evaluate_model


def run_task_b3(N: int = 32, K: int = 64, M: int = 8, seed: int = 42,
                results_dir: str = "results/B3_baselines", verbose: bool = True) -> Dict:
    """
    Run Task B3: Baseline Comparison.
    
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
        print("Task B3: Baseline Comparison")
        print("="*70)
    
    # Create results directories
    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create config and train model
    config = build_task_config(N, K, M, seed)
    
    probe_bank = get_probe_bank('continuous', N, K, seed)
    
    if verbose:
        print("Training ML model...")
    train_loader, val_loader, test_loader, metadata = create_dataloaders(config, probe_bank)
    model = create_model(config)
    model, history = train(model, train_loader, val_loader, config, metadata)
    
    # Evaluate
    results = evaluate_model(
        model, test_loader, config,
        metadata['test_powers_full'],
        metadata['test_labels'],
        metadata['test_observed_indices'],
        metadata['test_optimal_powers']
    )
    
    # Create bar plot comparison
    if verbose:
        print("Creating baseline comparison plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Random\n1/K', f'Random\n{M}/{K}', 'Best\nObserved', 'ML\nTop-1']
    values = [results.eta_random_1, results.eta_random_M, 
              results.eta_best_observed, results.eta_top1]
    colors = ['red', 'purple', 'orange', 'steelblue']
    
    bars = ax.bar(methods, values, color=colors, edgecolor='black', alpha=0.8)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('η (Power Ratio)', fontsize=12)
    ax.set_title(f'ML vs Baselines (N={N}, K={K}, M={M})', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Oracle')
    ax.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "baseline_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"  Saved: {plot_path}")
    plt.close()
    
    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Task B3: Baseline Comparison Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Configuration: N={N}, K={K}, M={M}, Seed={seed}\n\n")
        f.write("Performance (η):\n")
        for method, val in zip(methods, values):
            f.write(f"  {method.replace(chr(10), ' ')}: {val:.4f}\n")
        
        f.write("\nML Improvement over Baselines:\n")
        for i, method in enumerate(methods[:-1]):
            improvement = (results.eta_top1 - values[i]) / values[i] * 100
            f.write(f"  vs {method.replace(chr(10), ' ')}: +{improvement:.1f}%\n")
    
    if verbose:
        print(f"  Saved: {metrics_path}")
    
    return {
        'methods': methods,
        'values': values,
        'eval_results': results,
        'plots': [plot_path],
        'metrics_file': metrics_path
    }
