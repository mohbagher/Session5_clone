"""
Task A3: Probe Diversity Analysis

Compare diversity across all probe types: continuous, binary, 2-bit, and Hadamard.
Creates comparison plots and summary tables.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import Dict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from experiments.probe_generators import get_probe_bank
from experiments.diversity_analysis import (
    compute_diversity_metrics,
    compute_cosine_similarity_matrix,
    compute_hamming_distance_matrix
)


def run_task_a3(N: int = 32, K: int = 64, M: int = 8, seed: int = 42,
                results_dir: str = "results/A3_diversity_analysis", verbose: bool = True) -> Dict:
    """
    Run Task A3: Probe Diversity Analysis.
    
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
        print("Task A3: Probe Diversity Analysis")
        print("="*70)
    
    # Create results directories
    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate all probe types
    probe_types = ['continuous', 'binary', '2bit', 'hadamard']
    probe_banks = {}
    diversity_metrics = {}
    
    if verbose:
        print(f"Generating probe banks for all types...")
    
    for probe_type in probe_types:
        if verbose:
            print(f"  - {probe_type}")
        probe_banks[probe_type] = get_probe_bank(probe_type, N, K, seed)
        diversity_metrics[probe_type] = compute_diversity_metrics(probe_banks[probe_type])
    
    # Create summary table
    if verbose:
        print("Creating summary table...")
    
    summary_data = []
    for probe_type in probe_types:
        metrics = diversity_metrics[probe_type]
        summary_data.append({
            'Probe Type': probe_type.capitalize(),
            'Metric': metrics['metric_type'],
            'Mean': f"{metrics['mean']:.4f}",
            'Std': f"{metrics['std']:.4f}",
            'Min': f"{metrics['min']:.4f}",
            'Max': f"{metrics['max']:.4f}",
            'Median': f"{metrics['median']:.4f}"
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Save summary table
    summary_path = os.path.join(results_dir, "diversity_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    if verbose:
        print(f"  Saved: {summary_path}")
        print("\nSummary Table:")
        print(df_summary.to_string(index=False))
    
    # Plot pairwise similarity/distance distributions
    if verbose:
        print("\nCreating pairwise distance distributions...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, probe_type in enumerate(probe_types):
        ax = axes[idx]
        bank = probe_banks[probe_type]
        metrics = diversity_metrics[probe_type]
        
        if metrics['metric_type'] == 'cosine_similarity':
            # Compute similarity matrix
            matrix = compute_cosine_similarity_matrix(bank)
            xlabel = 'Cosine Similarity'
            title_suffix = 'Similarity'
        else:
            # Compute distance matrix
            matrix = compute_hamming_distance_matrix(bank)
            xlabel = 'Normalized Hamming Distance'
            title_suffix = 'Distance'
        
        # Extract upper triangle
        mask = np.triu(np.ones((K, K), dtype=bool), k=1)
        values = matrix[mask]
        
        # Plot histogram
        ax.hist(values, bins=40, color=f'C{idx}', edgecolor='black', alpha=0.7)
        ax.axvline(x=metrics['mean'], color='red', linestyle='--',
                   linewidth=2, label=f"Mean: {metrics['mean']:.3f}")
        ax.axvline(x=metrics['median'], color='green', linestyle=':',
                   linewidth=2, label=f"Median: {metrics['median']:.3f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.set_title(f'{probe_type.capitalize()}: Pairwise {title_suffix}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    dist_path = os.path.join(plots_dir, "pairwise_distributions.png")
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"  Saved: {dist_path}")
    plt.close()
    
    # Create comparison bar chart
    if verbose:
        print("Creating comparison bar chart...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for bar chart
    probe_names = [p.capitalize() for p in probe_types]
    means = [diversity_metrics[p]['mean'] for p in probe_types]
    stds = [diversity_metrics[p]['std'] for p in probe_types]
    
    x = np.arange(len(probe_names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=['steelblue', 'coral', 'lightgreen', 'gold'],
                  edgecolor='black', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(probe_names)
    ax.set_ylabel('Diversity Metric Value')
    ax.set_title(f'Probe Diversity Comparison (N={N}, K={K})')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Add note about different metrics
    ax.text(0.5, 0.02, 'Note: Continuous uses cosine similarity; others use Hamming distance',
            ha='center', transform=ax.transAxes, fontsize=9, style='italic')
    
    plt.tight_layout()
    comparison_path = os.path.join(plots_dir, "diversity_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"  Saved: {comparison_path}")
    plt.close()
    
    # Create phase distribution comparison
    if verbose:
        print("Creating phase distribution comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, probe_type in enumerate(probe_types):
        ax = axes[idx]
        bank = probe_banks[probe_type]
        phases_flat = bank.phases.flatten()
        
        if probe_type == 'continuous':
            bins = 50
        elif probe_type == '2bit':
            bins = 10
        else:
            bins = 5
        
        ax.hist(phases_flat / np.pi, bins=bins, color=f'C{idx}',
                edgecolor='black', alpha=0.7)
        ax.set_xlabel('Phase (π)')
        ax.set_ylabel('Count')
        ax.set_title(f'{probe_type.capitalize()} Phase Distribution')
        ax.grid(True, alpha=0.3)
        
        # Add vertical lines for discrete phases
        if probe_type == 'binary':
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=1, color='red', linestyle='--', alpha=0.5)
        elif probe_type == '2bit':
            for v in [0, 0.5, 1, 1.5]:
                ax.axvline(x=v, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    phase_dist_path = os.path.join(plots_dir, "phase_distributions.png")
    plt.savefig(phase_dist_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"  Saved: {phase_dist_path}")
    plt.close()
    
    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Task A3: Probe Diversity Analysis Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  N (RIS elements): {N}\n")
        f.write(f"  K (probes): {K}\n")
        f.write(f"  Seed: {seed}\n\n")
        
        f.write("Diversity Metrics for All Probe Types:\n")
        f.write(df_summary.to_string(index=False))
        f.write("\n\n")
        
        f.write("Key Observations:\n")
        f.write("  - Continuous probes show variable similarity due to random phases\n")
        f.write("  - Binary probes have discrete phase values, leading to specific distance patterns\n")
        f.write("  - 2-bit probes offer more resolution than binary while maintaining structure\n")
        f.write("  - Hadamard probes provide structured patterns with controlled diversity\n")
    
    if verbose:
        print(f"  Saved: {metrics_path}")
    
    return {
        'probe_banks': probe_banks,
        'diversity_metrics': diversity_metrics,
        'summary_df': df_summary,
        'plots': [dist_path, comparison_path, phase_dist_path],
        'metrics_file': metrics_path
    }
