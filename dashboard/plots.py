"""
Extended Plotting Functions for RIS PhD Ultimate Dashboard.

Provides 25+ plot types for comprehensive visualization and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
import sys
import os
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def set_plot_style(color_palette='viridis'):
    """Set consistent plot style with specified color palette."""
    plt.style.use('seaborn-v0_8-whitegrid')
    if color_palette == 'seaborn':
        sns.set_palette("deep")
    else:
        plt.rcParams['image.cmap'] = color_palette


# ============================================================================
# 1. TRAINING CURVES
# ============================================================================

def plot_training_curves(history, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """Plot training curves: loss, accuracy, eta, learning rate."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history.train_loss) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history.train_loss, 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history.val_loss, 'r-', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history.train_acc, 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history.val_acc, 'r-', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Eta
    axes[1, 0].plot(epochs, history.val_eta, 'g-', linewidth=2)
    axes[1, 0].axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('η')
    axes[1, 0].set_title('Validation η_top1')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 1].plot(epochs, history.learning_rates, 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


# ============================================================================
# 2-3. ETA DISTRIBUTION & CDF
# ============================================================================

def plot_eta_distribution(results, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """Plot histogram of eta values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    eta_ml = results.eta_top1_distribution
    eta_obs = results.eta_best_observed_distribution
    
    ax.hist(eta_ml, bins=50, alpha=0.7, color='steelblue', 
            label=f'ML (mean={np.mean(eta_ml):.3f})', edgecolor='black')
    ax.hist(eta_obs, bins=50, alpha=0.5, color='orange',
            label=f'Best Obs (mean={np.mean(eta_obs):.3f})', edgecolor='black')
    ax.axvline(x=results.eta_random_1, color='red', linestyle='--',
               linewidth=2, label=f'Random: {results.eta_random_1:.3f}')
    ax.axvline(x=1.0, color='green', linestyle='-',
               linewidth=2, label='Oracle')
    
    ax.set_xlabel('η (Power Ratio)')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of η (M={results.M}, K={results.K})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_cdf(results, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """Plot CDF of eta values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    eta = results.eta_top1_distribution
    sorted_eta = np.sort(eta)
    cdf = np.arange(1, len(sorted_eta) + 1) / len(sorted_eta)
    
    ax.plot(sorted_eta, cdf, linewidth=2, label='ML Model')
    ax.axvline(x=1.0, color='green', linestyle='--', label='Oracle')
    ax.set_xlabel('η (Power Ratio)')
    ax.set_ylabel('CDF')
    ax.set_title('Cumulative Distribution Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


# ============================================================================
# 4-5. TOP-M & BASELINE COMPARISON
# ============================================================================

def plot_top_m_comparison(results, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """Bar chart of top-m accuracies."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['accuracy_top1', 'accuracy_top2', 'accuracy_top4', 'accuracy_top8']
    values = [getattr(results, m, 0) for m in metrics]
    labels = ['Top-1', 'Top-2', 'Top-4', 'Top-8']
    
    bars = ax.bar(labels, values, color='steelblue', alpha=0.8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Top-m Selection Accuracy')
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_baseline_comparison(results, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """Compare ML model against baselines."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Random 1', 'Random M', 'Best Obs', 'ML Model', 'Oracle']
    eta_values = [
        results.eta_random_1,
        results.eta_random_M,
        results.eta_best_observed,
        results.eta_top1,
        results.eta_oracle
    ]
    colors = ['red', 'orange', 'yellow', 'steelblue', 'green']
    
    bars = ax.bar(methods, eta_values, color=colors, alpha=0.8)
    ax.set_ylabel('η (Power Ratio)')
    ax.set_title('Baseline Comparison')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


# ============================================================================
# 6-8. VIOLIN, BOX, SCATTER COMPARISON
# ============================================================================

def plot_violin(results_dict, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """Violin plot comparing multiple models."""
    import pandas as pd
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data = []
    for name, res in results_dict.items():
        for eta in res.evaluation.eta_top1_distribution:
            data.append({'Model': name, 'η': eta})
    
    df = pd.DataFrame(data)
    sns.violinplot(data=df, x='Model', y='η', ax=ax)
    ax.set_title('Model Performance Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_box(results_dict, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """Box plot comparing multiple models."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    labels = list(results_dict.keys())
    data = [res.evaluation.eta_top1_distribution for res in results_dict.values()]
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.7))
    ax.set_ylabel('η (Power Ratio)')
    ax.set_title('Model Performance Distribution (Box Plot)')
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_scatter(results_list, labels, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """Scatter plot comparing configurations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (results, label) in enumerate(zip(results_list, labels)):
        x = np.random.normal(i, 0.1, len(results.evaluation.eta_top1_distribution))
        ax.scatter(x, results.evaluation.eta_top1_distribution, alpha=0.6, label=label)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('η (Power Ratio)')
    ax.set_title('Performance Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


# ============================================================================
# 9-10. HEATMAP & CORRELATION MATRIX
# ============================================================================

def plot_heatmap(probe_bank, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """Plot phase heatmap of probe configurations."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    im = ax.imshow(probe_bank.phases, aspect='auto', cmap='viridis')
    ax.set_title(f'{probe_bank.probe_type.title()} Probe Bank Heatmap')
    ax.set_xlabel('RIS Elements')
    ax.set_ylabel('Probe Index')
    plt.colorbar(im, ax=ax, label='Phase (radians)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_correlation_matrix(probe_bank, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """Plot correlation matrix of probe phases."""
    from experiments.diversity_analysis import compute_cosine_similarity_matrix
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    similarity = compute_cosine_similarity_matrix(probe_bank)
    im = ax.imshow(similarity, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_title('Probe Similarity Matrix')
    ax.set_xlabel('Probe Index')
    ax.set_ylabel('Probe Index')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


# ============================================================================
# 11-12. CONFUSION MATRIX & LEARNING CURVE
# ============================================================================

def plot_confusion_matrix(predictions, labels, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """Plot confusion matrix (for classification analysis)."""
    from sklearn.metrics import confusion_matrix
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cm = confusion_matrix(labels, predictions)
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_learning_curve(history, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """Plot train vs validation learning curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history.train_loss) + 1)
    
    # Loss curve
    axes[0].plot(epochs, history.train_loss, 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history.val_loss, 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Learning Curve (Loss)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(epochs, history.train_acc, 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history.val_acc, 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Learning Curve (Accuracy)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


# ============================================================================
# 13-15. PARAMETER SWEEP PLOTS (ETA VS M/K/N)
# ============================================================================

def plot_eta_vs_M(results_list, M_values, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """Plot performance vs sensing budget M."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    eta_means = [np.mean(r.evaluation.eta_top1_distribution) for r in results_list]
    eta_stds = [np.std(r.evaluation.eta_top1_distribution) for r in results_list]
    
    ax.errorbar(M_values, eta_means, yerr=eta_stds, marker='o', 
                linewidth=2, capsize=5, label='ML Model')
    ax.set_xlabel('M (Sensing Budget)')
    ax.set_ylabel('η (Power Ratio)')
    ax.set_title('Performance vs Sensing Budget')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_eta_vs_K(results_list, K_values, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """Plot performance vs codebook size K."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    eta_means = [np.mean(r.evaluation.eta_top1_distribution) for r in results_list]
    eta_stds = [np.std(r.evaluation.eta_top1_distribution) for r in results_list]
    
    ax.errorbar(K_values, eta_means, yerr=eta_stds, marker='s', 
                linewidth=2, capsize=5, label='ML Model')
    ax.set_xlabel('K (Codebook Size)')
    ax.set_ylabel('η (Power Ratio)')
    ax.set_title('Performance vs Codebook Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_eta_vs_N(results_list, N_values, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """Plot performance vs RIS elements N."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    eta_means = [np.mean(r.evaluation.eta_top1_distribution) for r in results_list]
    eta_stds = [np.std(r.evaluation.eta_top1_distribution) for r in results_list]
    
    ax.errorbar(N_values, eta_means, yerr=eta_stds, marker='^', 
                linewidth=2, capsize=5, label='ML Model')
    ax.set_xlabel('N (RIS Elements)')
    ax.set_ylabel('η (Power Ratio)')
    ax.set_title('Performance vs RIS Elements')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


# ============================================================================
# 16-18. PROBE & CONVERGENCE ANALYSIS
# ============================================================================

def plot_probe_type_comparison(results_dict, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """Compare different probe types."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    probe_types = list(results_dict.keys())
    eta_means = [np.mean(r.evaluation.eta_top1_distribution) for r in results_dict.values()]
    eta_stds = [np.std(r.evaluation.eta_top1_distribution) for r in results_dict.values()]
    
    x_pos = np.arange(len(probe_types))
    bars = ax.bar(x_pos, eta_means, yerr=eta_stds, capsize=5, 
                  color='steelblue', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(probe_types, rotation=45)
    ax.set_ylabel('η (Power Ratio)')
    ax.set_title('Probe Type Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_phase_bits_comparison(results_dict, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """Compare different phase quantization levels."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    phase_bits = sorted(results_dict.keys())
    eta_means = [np.mean(results_dict[b].evaluation.eta_top1_distribution) for b in phase_bits]
    eta_stds = [np.std(results_dict[b].evaluation.eta_top1_distribution) for b in phase_bits]
    
    ax.errorbar(phase_bits, eta_means, yerr=eta_stds, marker='o', 
                linewidth=2, capsize=5)
    ax.set_xlabel('Phase Bits')
    ax.set_ylabel('η (Power Ratio)')
    ax.set_title('Phase Quantization Comparison')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_convergence_analysis(histories_dict, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """Compare convergence of different models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, history in histories_dict.items():
        epochs = range(1, len(history.val_loss) + 1)
        axes[0].plot(epochs, history.val_loss, label=name, linewidth=2)
        axes[1].plot(epochs, history.val_eta, label=name, linewidth=2)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Loss')
    axes[0].set_title('Convergence: Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation η')
    axes[1].set_title('Convergence: η')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


# ============================================================================
# 19-20. MODEL COMPLEXITY PLOTS
# ============================================================================

def plot_model_size_vs_performance(results_dict, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """Plot model parameters vs performance."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = []
    param_counts = []
    eta_means = []
    
    for name, results in results_dict.items():
        model_names.append(name)
        # Get parameter count from config
        config = results.config
        hidden_sizes = config.get('hidden_sizes', [256, 128])
        K = config.get('K', 64)
        
        # Calculate params
        params = 0
        prev = 2 * K
        for h in hidden_sizes:
            params += prev * h + h
            prev = h
        params += prev * K + K
        
        param_counts.append(params / 1e6)  # In millions
        eta_means.append(np.mean(results.evaluation.eta_top1_distribution))
    
    scatter = ax.scatter(param_counts, eta_means, s=100, alpha=0.7)
    
    for i, name in enumerate(model_names):
        ax.annotate(name, (param_counts[i], eta_means[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Model Parameters (millions)')
    ax.set_ylabel('η (Power Ratio)')
    ax.set_title('Model Size vs Performance')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_radar_chart(results_dict, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """Multi-metric radar chart."""
    from math import pi
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Metrics to plot
    metrics = ['Top-1 Acc', 'Top-2 Acc', 'Top-4 Acc', 'Top-8 Acc', 'η_top1']
    num_metrics = len(metrics)
    
    angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
    angles += angles[:1]
    
    for name, results in results_dict.items():
        values = [
            results.evaluation.accuracy_top1,
            results.evaluation.accuracy_top2,
            results.evaluation.accuracy_top4,
            results.evaluation.accuracy_top8,
            results.evaluation.eta_top1
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=name)
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('Multi-Metric Comparison', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


# ============================================================================
# 21-23. ERROR & POWER ANALYSIS
# ============================================================================

def plot_error_analysis(results, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """Analyze prediction errors."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    eta_ml = results.eta_top1_distribution
    eta_oracle = np.ones_like(eta_ml)
    errors = eta_oracle - eta_ml
    
    # Error histogram
    axes[0].hist(errors, bins=50, color='coral', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0, color='k', linestyle='--')
    axes[0].set_xlabel('Error (η_oracle - η_ML)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Error Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # QQ plot (quantile-quantile)
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Normality Check)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_power_distribution(powers_full, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """Plot power distribution statistics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Power histogram
    axes[0].hist(powers_full.flatten(), bins=100, color='skyblue', 
                 alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Received Power')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Power Distribution')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Per-probe statistics
    probe_means = powers_full.mean(axis=0)
    probe_stds = powers_full.std(axis=0)
    probe_indices = np.arange(len(probe_means))
    
    axes[1].errorbar(probe_indices, probe_means, yerr=probe_stds, 
                     fmt='o', alpha=0.6, capsize=3)
    axes[1].set_xlabel('Probe Index')
    axes[1].set_ylabel('Mean Power')
    axes[1].set_title('Per-Probe Power Statistics')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_channel_statistics(metadata, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """Plot channel statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Channel magnitude distribution
    h_mag = np.abs(metadata['h_samples'])
    g_mag = np.abs(metadata['g_samples'])
    
    axes[0, 0].hist(h_mag.flatten(), bins=50, alpha=0.7, color='blue', 
                    label='|h|', edgecolor='black')
    axes[0, 0].set_xlabel('Magnitude')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('BS-RIS Channel Magnitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(g_mag.flatten(), bins=50, alpha=0.7, color='green', 
                    label='|g|', edgecolor='black')
    axes[0, 1].set_xlabel('Magnitude')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('RIS-UE Channel Magnitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Phase distribution
    h_phase = np.angle(metadata['h_samples'])
    g_phase = np.angle(metadata['g_samples'])
    
    axes[1, 0].hist(h_phase.flatten(), bins=50, alpha=0.7, color='blue',
                    label='∠h', edgecolor='black')
    axes[1, 0].set_xlabel('Phase (radians)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('BS-RIS Channel Phase')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(g_phase.flatten(), bins=50, alpha=0.7, color='green',
                    label='∠g', edgecolor='black')
    axes[1, 1].set_xlabel('Phase (radians)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('RIS-UE Channel Phase')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


# ============================================================================
# 24-25. 3D & PARETO PLOTS
# ============================================================================

def plot_3d_surface(results, param1_vals, param2_vals, param1_name='M', param2_name='K',
                   save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """3D surface plot of performance vs two parameters."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid
    P1, P2 = np.meshgrid(param1_vals, param2_vals)
    
    # Extract eta values
    Z = np.zeros_like(P1, dtype=float)
    for i, p1 in enumerate(param1_vals):
        for j, p2 in enumerate(param2_vals):
            key = f"{param1_name}={p1}_{param2_name}={p2}"
            if key in results:
                Z[j, i] = np.mean(results[key].evaluation.eta_top1_distribution)
    
    # Plot surface
    surf = ax.plot_surface(P1, P2, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel(param1_name)
    ax.set_ylabel(param2_name)
    ax.set_zlabel('η (Power Ratio)')
    ax.set_title('3D Parameter Surface')
    fig.colorbar(surf, ax=ax, shrink=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_pareto_front(results_dict, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """Pareto front: complexity vs performance."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    complexities = []
    performances = []
    names = []
    
    for name, results in results_dict.items():
        # Calculate complexity (parameter count)
        config = results.config
        hidden_sizes = config.get('hidden_sizes', [256, 128])
        K = config.get('K', 64)
        
        params = 0
        prev = 2 * K
        for h in hidden_sizes:
            params += prev * h + h
            prev = h
        params += prev * K + K
        
        complexities.append(params / 1e6)
        performances.append(np.mean(results.evaluation.eta_top1_distribution))
        names.append(name)
    
    # Plot all points
    scatter = ax.scatter(complexities, performances, s=100, alpha=0.7, c=performances, 
                        cmap='viridis')
    
    # Find and plot Pareto front
    from scipy.spatial import ConvexHull
    points = np.column_stack([complexities, performances])
    
    # Simple Pareto: maximize performance, minimize complexity
    pareto_indices = []
    for i, (c, p) in enumerate(zip(complexities, performances)):
        dominated = False
        for j, (c2, p2) in enumerate(zip(complexities, performances)):
            if i != j and c2 <= c and p2 >= p and (c2 < c or p2 > p):
                dominated = True
                break
        if not dominated:
            pareto_indices.append(i)
    
    if pareto_indices:
        pareto_c = [complexities[i] for i in pareto_indices]
        pareto_p = [performances[i] for i in pareto_indices]
        # Sort by complexity
        sorted_indices = np.argsort(pareto_c)
        pareto_c = [pareto_c[i] for i in sorted_indices]
        pareto_p = [pareto_p[i] for i in sorted_indices]
        ax.plot(pareto_c, pareto_p, 'r--', linewidth=2, label='Pareto Front')
    
    # Annotate points
    for i, name in enumerate(names):
        ax.annotate(name, (complexities[i], performances[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Model Complexity (M parameters)')
    ax.set_ylabel('Performance (η)')
    ax.set_title('Pareto Front: Complexity vs Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Performance')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


# ============================================================================
# PLOT REGISTRY
# ============================================================================

EXTENDED_PLOT_REGISTRY = {
    'training_curves': plot_training_curves,
    'eta_distribution': plot_eta_distribution,
    'cdf': plot_cdf,
    'top_m_comparison': plot_top_m_comparison,
    'baseline_comparison': plot_baseline_comparison,
    'violin': plot_violin,
    'box': plot_box,
    'scatter': plot_scatter,
    'heatmap': plot_heatmap,
    'correlation_matrix': plot_correlation_matrix,
    'confusion_matrix': plot_confusion_matrix,
    'learning_curve': plot_learning_curve,
    'eta_vs_M': plot_eta_vs_M,
    'eta_vs_K': plot_eta_vs_K,
    'eta_vs_N': plot_eta_vs_N,
    'probe_type_comparison': plot_probe_type_comparison,
    'phase_bits_comparison': plot_phase_bits_comparison,
    'convergence_analysis': plot_convergence_analysis,
    'model_size_vs_performance': plot_model_size_vs_performance,
    'radar_chart': plot_radar_chart,
    'error_analysis': plot_error_analysis,
    'power_distribution': plot_power_distribution,
    'channel_statistics': plot_channel_statistics,
    '3d_surface': plot_3d_surface,
    'pareto_front': plot_pareto_front,
}
