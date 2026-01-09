"""
Utility functions for visualization and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os

from training import TrainingHistory
from evaluation import EvaluationResults


def set_plot_style():
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 11


def plot_training_history(history: TrainingHistory,
                          save_path: Optional[str] = None):
    """Plot training curves."""
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history.train_loss) + 1)
    
    # Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history.train_loss, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history.val_loss, 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax = axes[0, 1]
    ax.plot(epochs, history.train_acc, 'b-', label='Train Acc', linewidth=2)
    ax.plot(epochs, history.val_acc, 'r-', label='Val Acc', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation eta
    ax = axes[1, 0]
    ax.plot(epochs, history.val_eta, 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('η (Power Ratio)')
    ax.set_title('Validation η_top1')
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Optimal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[1, 1]
    ax.plot(epochs, history.learning_rates, 'm-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved:  {save_path}")
    plt.show()


def plot_eta_distribution(results: EvaluationResults,
                          save_path: Optional[str] = None):
    """Plot distribution of eta values with baselines."""
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    eta_ml = results.eta_top1_distribution
    eta_obs = results.eta_best_observed_distribution
    
    # Histogram comparison
    ax = axes[0]
    ax.hist(eta_ml, bins=50, alpha=0.7, color='steelblue', 
            label=f'ML Model (mean={np.mean(eta_ml):.3f})', edgecolor='black')
    ax.hist(eta_obs, bins=50, alpha=0.5, color='orange',
            label=f'Best Observed (mean={np.mean(eta_obs):.3f})', edgecolor='black')
    ax.axvline(x=results.eta_random_1, color='red', linestyle='--',
               linewidth=2, label=f'Random 1/K:  {results.eta_random_1:.3f}')
    ax.axvline(x=1.0, color='green', linestyle='-',
               linewidth=2, label='Oracle (η=1)')
    ax.set_xlabel('η (Power Ratio)')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of η (M={results.M}, K={results.K})')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # CDF comparison
    ax = axes[1]
    sorted_ml = np.sort(eta_ml)
    sorted_obs = np.sort(eta_obs)
    cdf = np.arange(1, len(sorted_ml) + 1) / len(sorted_ml)
    
    ax.plot(sorted_ml, cdf, 'b-', linewidth=2, label='ML Model')
    ax.plot(sorted_obs, cdf, 'orange', linewidth=2, label='Best Observed')
    ax.axvline(x=results.eta_random_1, color='red', linestyle='--',
               linewidth=2, label=f'Random 1/K')
    ax.axvline(x=results.eta_random_M, color='purple', linestyle='--',
               linewidth=2, label=f'Random M/K')
    ax.set_xlabel('η (Power Ratio)')
    ax.set_ylabel('CDF')
    ax.set_title('Cumulative Distribution of η')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_top_m_comparison(results: EvaluationResults,
                          save_path:  Optional[str] = None):
    """Plot top-m accuracy and eta comparison with baselines."""
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    m_values = [1, 2, 4, 8]
    accuracies = [results.accuracy_top1, results.accuracy_top2,
                  results.accuracy_top4, results.accuracy_top8]
    etas = [results.eta_top1, results.eta_top2,
            results.eta_top4, results.eta_top8]
    
    # Accuracy bar chart
    ax = axes[0]
    x = np.arange(len(m_values))
    bars = ax.bar(x, accuracies, color='steelblue', edgecolor='black', label='ML Model')
    ax.axhline(y=1/results.K, color='red', linestyle='--',
               linewidth=2, label=f'Random (1/K={1/results.K:.4f})')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Top-{m}' for m in m_values])
    ax.set_ylabel('Accuracy')
    ax.set_title('Top-m Accuracy (Oracle in Top-m Predictions)')
    ax.set_ylim(0, 1.05)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=11)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Eta bar chart with baselines
    ax = axes[1]
    bars = ax.bar(x, etas, color='coral', edgecolor='black', label='ML Model')
    
    # Baseline lines
    ax.axhline(y=results.eta_random_1, color='red', linestyle='--',
               linewidth=2, label=f'Random 1/K: {results.eta_random_1:.3f}')
    ax.axhline(y=results.eta_random_M, color='purple', linestyle=':',
               linewidth=2, label=f'Random M/K: {results.eta_random_M:.3f}')
    ax.axhline(y=results.eta_best_observed, color='orange', linestyle='-.',
               linewidth=2, label=f'Best Observed: {results.eta_best_observed:.3f}')
    ax.axhline(y=1.0, color='green', linestyle='-',
               linewidth=2, label='Oracle (η=1)')
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'Top-{m}' for m in m_values])
    ax.set_ylabel('η (Power Ratio)')
    ax.set_title(f'Power Ratio η (M={results.M} observed, K={results.K} total)')
    ax.set_ylim(0, 1.1)
    for bar, eta in zip(bars, etas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{eta:.3f}', ha='center', va='bottom', fontsize=11)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_baseline_comparison(results: EvaluationResults,
                             save_path: Optional[str] = None):
    """Bar chart comparing ML model vs all baselines."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Random\n1/K', f'Random\nM={results.M}', f'Best\nObserved', 
               'ML\nTop-1', 'ML\nTop-2', 'ML\nTop-4', 'Oracle']
    values = [results.eta_random_1, results.eta_random_M, results.eta_best_observed,
              results.eta_top1, results.eta_top2, results.eta_top4, results.eta_oracle]
    colors = ['red', 'purple', 'orange', 'steelblue', 'steelblue', 'steelblue', 'green']
    
    bars = ax.bar(methods, values, color=colors, edgecolor='black', alpha=0.8)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('η (Power Ratio)')
    ax.set_title(f'Method Comparison (M={results.M} probes, K={results.K} total)')
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path: 
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def save_results(results: EvaluationResults,
                 history: TrainingHistory,
                 config,
                 save_dir: str = "results"):
    """Save all results to files."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics as text
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write("=" * 70 + "\n")
        f.write("RIS PROBE-BASED CONTROL - LIMITED PROBING RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  N (RIS elements):    {config.system.N}\n")
        f.write(f"  K (total probes):    {config.system.K}\n")
        f.write(f"  M (sensing budget):  {config.system.M}\n")
        f.write(f"  M/K ratio:           {config.system.M/config.system.K:.2%}\n")
        f.write(f"  Training samples:    {config.data.n_train}\n")
        f.write(f"  Architecture:        {config.model.hidden_sizes}\n\n")
        
        f.write("Results:\n")
        for key, value in results.to_dict().items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")
    
    # Save numpy arrays
    np.savez(
        os.path.join(save_dir, "results.npz"),
        eta_top1_distribution=results.eta_top1_distribution,
        eta_best_observed_distribution=results.eta_best_observed_distribution,
        train_loss=history.train_loss,
        val_loss=history.val_loss,
        val_eta=history.val_eta
    )
    
    print(f"Results saved to {save_dir}/")