"""
Task B2: Top-m Selection

Evaluate Top-1, 2, 4, 8 performance to understand how much better we can do
by considering multiple predictions instead of just the top-1.
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


def run_task_b2(N: int = 32, K: int = 64, M: int = 8, seed: int = 42,
                results_dir: str = "results/B2_top_m_selection", verbose: bool = True) -> Dict:
    """
    Run Task B2: Top-m Selection Analysis.
    
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
        print("Task B2: Top-m Selection")
        print("="*70)
    
    # Create results directories
    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create config
    config = build_task_config(N, K, M, seed)
    
    # Generate probe bank (use continuous)
    probe_bank = get_probe_bank('continuous', N, K, seed)
    
    # Create dataloaders
    if verbose:
        print("Generating dataset...")
    train_loader, val_loader, test_loader, metadata = create_dataloaders(config, probe_bank)
    
    # Create and train model
    if verbose:
        print("Training model...")
    model = create_model(config)
    model, history = train(model, train_loader, val_loader, config, metadata)
    
    # Evaluate with different top-m values
    if verbose:
        print("Evaluating model...")
    results = evaluate_model(
        model, test_loader, config,
        metadata['test_powers_full'],
        metadata['test_labels'],
        metadata['test_observed_indices'],
        metadata['test_optimal_powers']
    )
    
    # Extract top-m results
    top_m_values = [1, 2, 4, 8]
    eta_values = [results.eta_top1, results.eta_top2, results.eta_top4, results.eta_top8]
    accuracy_values = [results.accuracy_top1, results.accuracy_top2, 
                       results.accuracy_top4, results.accuracy_top8]
    
    # Plot η vs top-m
    if verbose:
        print("Creating η vs top-m plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # η plot
    ax = axes[0]
    ax.plot(top_m_values, eta_values, marker='o', linewidth=2, 
            markersize=10, color='steelblue')
    ax.axhline(y=results.eta_best_observed, color='orange', linestyle='--',
               linewidth=2, label=f'Best Observed: {results.eta_best_observed:.3f}')
    ax.axhline(y=results.eta_random_M, color='purple', linestyle=':',
               linewidth=2, label=f'Random M/K: {results.eta_random_M:.3f}')
    ax.set_xlabel('Top-m')
    ax.set_ylabel('η (Power Ratio)')
    ax.set_title(f'Power Ratio vs Top-m Selection (M={M}, K={K})')
    ax.set_xticks(top_m_values)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Accuracy plot
    ax = axes[1]
    ax.plot(top_m_values, accuracy_values, marker='s', linewidth=2,
            markersize=10, color='coral')
    ax.axhline(y=1/K, color='red', linestyle='--',
               linewidth=2, label=f'Random: {1/K:.4f}')
    ax.set_xlabel('Top-m')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Top-m Accuracy (Oracle in Top-m)')
    ax.set_xticks(top_m_values)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "top_m_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"  Saved: {plot_path}")
    plt.close()
    
    # Create summary table plot
    if verbose:
        print("Creating summary table...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    table_data = []
    for m, eta, acc in zip(top_m_values, eta_values, accuracy_values):
        table_data.append([f'Top-{m}', f'{eta:.4f}', f'{acc:.4f}'])
    
    table = ax.table(cellText=table_data,
                     colLabels=['Selection', 'η (Power Ratio)', 'Accuracy'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(table_data) + 1):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
    
    ax.set_title(f'Top-m Performance Summary (N={N}, K={K}, M={M})', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    table_path = os.path.join(plots_dir, "top_m_summary_table.png")
    plt.savefig(table_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"  Saved: {table_path}")
    plt.close()
    
    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Task B2: Top-m Selection Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  N (RIS elements): {N}\n")
        f.write(f"  K (probes): {K}\n")
        f.write(f"  M (sensing budget): {M}\n")
        f.write(f"  Seed: {seed}\n\n")
        
        f.write("Top-m Performance:\n")
        for m, eta, acc in zip(top_m_values, eta_values, accuracy_values):
            f.write(f"  Top-{m}: η = {eta:.4f}, accuracy = {acc:.4f}\n")
        
        f.write("\nBaselines:\n")
        f.write(f"  Best Observed: η = {results.eta_best_observed:.4f}\n")
        f.write(f"  Random M/K: η = {results.eta_random_M:.4f}\n")
        f.write(f"  Random 1/K: η = {results.eta_random_1:.4f}\n")
        
        f.write("\nKey Observations:\n")
        f.write(f"  - Top-1 achieves η = {results.eta_top1:.4f}\n")
        f.write(f"  - Top-8 achieves η = {results.eta_top8:.4f}\n")
        improvement = (results.eta_top8 - results.eta_top1) / results.eta_top1 * 100
        f.write(f"  - Improvement from top-1 to top-8: {improvement:.1f}%\n")
    
    if verbose:
        print(f"  Saved: {metrics_path}")
        print("\nTop-m Summary:")
        for m, eta, acc in zip(top_m_values, eta_values, accuracy_values):
            print(f"  Top-{m}: η={eta:.4f}, acc={acc:.4f}")
    
    return {
        'top_m_values': top_m_values,
        'eta_values': eta_values,
        'accuracy_values': accuracy_values,
        'eval_results': results,
        'plots': [plot_path, table_path],
        'metrics_file': metrics_path
    }
