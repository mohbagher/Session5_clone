"""
Task A2: Hadamard Probes

Generate Hadamard-based structured probes and compare with random binary probes.
Creates heatmaps showing structured patterns and Hamming distance distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from experiments.probe_generators import generate_probe_bank_hadamard, generate_probe_bank_binary
from experiments.diversity_analysis import compute_diversity_metrics, compute_hamming_distance_matrix


def run_task_a2(N: int = 32, K: int = 64, M: int = 8, seed: int = 42,
                results_dir: str = "results/A2_hadamard_probes", verbose: bool = True) -> Dict:
    """
    Run Task A2: Hadamard Probe Analysis.
    
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
        print("Task A2: Hadamard Probes")
        print("="*70)
    
    # Create results directories
    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate Hadamard probes
    if verbose:
        print(f"Generating {K} Hadamard probes with N={N} elements...")
    hadamard_bank = generate_probe_bank_hadamard(N, K)
    
    # Generate random binary probes for comparison
    binary_bank = generate_probe_bank_binary(N, K, seed)
    
    # Compute diversity metrics
    if verbose:
        print("Computing diversity metrics...")
    hadamard_metrics = compute_diversity_metrics(hadamard_bank)
    binary_metrics = compute_diversity_metrics(binary_bank)
    
    # Compute Hamming distance matrices
    hadamard_distances = compute_hamming_distance_matrix(hadamard_bank)
    binary_distances = compute_hamming_distance_matrix(binary_bank)
    
    # Create phase heatmap comparison
    if verbose:
        print("Creating phase heatmap...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Hadamard heatmap
    ax = axes[0]
    sns.heatmap(hadamard_bank.phases / np.pi, ax=ax, cmap='RdYlBu_r',
                cbar_kws={'label': 'Phase (π)'}, vmin=0, vmax=1)
    ax.set_xlabel('RIS Element Index')
    ax.set_ylabel('Probe Index')
    ax.set_title(f'Hadamard Probes (Structured)')
    
    # Random binary heatmap
    ax = axes[1]
    sns.heatmap(binary_bank.phases / np.pi, ax=ax, cmap='RdYlBu_r',
                cbar_kws={'label': 'Phase (π)'}, vmin=0, vmax=1)
    ax.set_xlabel('RIS Element Index')
    ax.set_ylabel('Probe Index')
    ax.set_title(f'Random Binary Probes')
    
    plt.tight_layout()
    heatmap_path = os.path.join(plots_dir, "phase_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"  Saved: {heatmap_path}")
    plt.close()
    
    # Create Hamming distance distribution
    if verbose:
        print("Creating Hamming distance distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract upper triangular values (excluding diagonal)
    K_probes = hadamard_bank.K
    mask = np.triu(np.ones((K_probes, K_probes), dtype=bool), k=1)
    hadamard_dist_values = hadamard_distances[mask]
    binary_dist_values = binary_distances[mask]
    
    # Hadamard histogram
    ax = axes[0]
    ax.hist(hadamard_dist_values, bins=30, color='steelblue',
            edgecolor='black', alpha=0.7)
    ax.set_xlabel('Normalized Hamming Distance')
    ax.set_ylabel('Count')
    ax.set_title('Hadamard Probes: Pairwise Hamming Distance')
    ax.axvline(x=hadamard_metrics['mean'], color='red', linestyle='--',
               linewidth=2, label=f"Mean: {hadamard_metrics['mean']:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Random binary histogram
    ax = axes[1]
    ax.hist(binary_dist_values, bins=30, color='coral',
            edgecolor='black', alpha=0.7)
    ax.set_xlabel('Normalized Hamming Distance')
    ax.set_ylabel('Count')
    ax.set_title('Random Binary Probes: Pairwise Hamming Distance')
    ax.axvline(x=binary_metrics['mean'], color='red', linestyle='--',
               linewidth=2, label=f"Mean: {binary_metrics['mean']:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    histogram_path = os.path.join(plots_dir, "hamming_distance_distribution.png")
    plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"  Saved: {histogram_path}")
    plt.close()
    
    # Create distance matrix heatmaps
    if verbose:
        print("Creating distance matrix heatmaps...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Hadamard distance matrix
    ax = axes[0]
    sns.heatmap(hadamard_distances, ax=ax, cmap='viridis',
                cbar_kws={'label': 'Normalized Hamming Distance'})
    ax.set_xlabel('Probe Index')
    ax.set_ylabel('Probe Index')
    ax.set_title('Hadamard: Pairwise Distance Matrix')
    
    # Binary distance matrix
    ax = axes[1]
    sns.heatmap(binary_distances, ax=ax, cmap='viridis',
                cbar_kws={'label': 'Normalized Hamming Distance'})
    ax.set_xlabel('Probe Index')
    ax.set_ylabel('Probe Index')
    ax.set_title('Random Binary: Pairwise Distance Matrix')
    
    plt.tight_layout()
    distance_matrix_path = os.path.join(plots_dir, "distance_matrices.png")
    plt.savefig(distance_matrix_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"  Saved: {distance_matrix_path}")
    plt.close()
    
    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Task A2: Hadamard Probes Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  N (RIS elements): {N}\n")
        f.write(f"  K (probes): {K}\n")
        f.write(f"  Seed: {seed}\n\n")
        
        f.write("Hadamard Probe Diversity Metrics:\n")
        for key, val in hadamard_metrics.items():
            f.write(f"  {key}: {val}\n")
        
        f.write("\nRandom Binary Probe Diversity Metrics (for comparison):\n")
        for key, val in binary_metrics.items():
            f.write(f"  {key}: {val}\n")
        
        f.write("\nObservations:\n")
        f.write(f"  Hadamard probes provide structured patterns with controlled diversity.\n")
        f.write(f"  Mean Hamming distance: Hadamard={hadamard_metrics['mean']:.4f}, "
                f"Binary={binary_metrics['mean']:.4f}\n")
    
    if verbose:
        print(f"  Saved: {metrics_path}")
        print("\nHadamard Probe Diversity:")
        print(f"  Mean Hamming Distance: {hadamard_metrics['mean']:.4f}")
        print(f"  Std: {hadamard_metrics['std']:.4f}")
        print(f"  Range: [{hadamard_metrics['min']:.4f}, {hadamard_metrics['max']:.4f}]")
        print("\nComparison:")
        print(f"  Hadamard mean: {hadamard_metrics['mean']:.4f}")
        print(f"  Random binary mean: {binary_metrics['mean']:.4f}")
    
    return {
        'hadamard_bank': hadamard_bank,
        'binary_bank': binary_bank,
        'hadamard_metrics': hadamard_metrics,
        'binary_metrics': binary_metrics,
        'plots': [heatmap_path, histogram_path, distance_matrix_path],
        'metrics_file': metrics_path
    }
