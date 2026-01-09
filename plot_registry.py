"""
Plot Registry for RIS Probe-Based ML System.

Provides a unified interface to all visualization functions.
"""

from typing import Dict, Callable, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Import existing plot functions
from utils import (
    plot_training_history,
    plot_eta_distribution,
    plot_top_m_comparison,
    plot_baseline_comparison
)


def plot_cdf(results, save_path: Optional[str] = None):
    """Plot CDF of eta values."""
    eta = results.eta_top1_distribution
    sorted_eta = np.sort(eta)
    cdf = np.arange(1, len(sorted_eta) + 1) / len(sorted_eta)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_eta, cdf, linewidth=2)
    plt.xlabel('η (Power Ratio)')
    plt.ylabel('CDF')
    plt.title('Cumulative Distribution Function')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_violin(results_dict, save_path: Optional[str] = None):
    """Plot violin plot comparing multiple models."""
    import pandas as pd
    
    data = []
    for name, res in results_dict.items():
        for eta in res.eta_top1_distribution:
            data.append({'Model': name, 'η': eta})
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x='Model', y='η')
    plt.title('Model Performance Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_heatmap(probe_bank, save_path: Optional[str] = None):
    """Plot phase heatmap of probe configurations."""
    plt.figure(figsize=(14, 8))
    sns.heatmap(probe_bank.phases, cmap="viridis", cbar_kws={'label': 'Phase (radians)'})
    plt.title(f'{probe_bank.probe_type.title()} Probe Bank Heatmap')
    plt.xlabel('RIS Elements')
    plt.ylabel('Probe Index')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_scatter_comparison(results_list, labels, save_path: Optional[str] = None):
    """Scatter plot comparing multiple configurations."""
    plt.figure(figsize=(10, 6))
    
    for i, (results, label) in enumerate(zip(results_list, labels)):
        x = np.random.normal(i, 0.1, len(results.eta_top1_distribution))
        plt.scatter(x, results.eta_top1_distribution, alpha=0.6, label=label)
    
    plt.xticks(range(len(labels)), labels)
    plt.ylabel('η (Power Ratio)')
    plt.title('Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_box_comparison(results_dict, save_path: Optional[str] = None):
    """Box plot comparing multiple models."""
    labels = list(results_dict.keys())
    data = [res.eta_top1_distribution for res in results_dict.values()]
    
    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=labels, patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.7))
    plt.ylabel('η (Power Ratio)')
    plt.title('Model Performance Distribution (Box Plot)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_correlation_matrix(probe_bank, save_path: Optional[str] = None):
    """Plot correlation matrix of probe phases."""
    from experiments.diversity_analysis import compute_cosine_similarity_matrix
    
    similarity = compute_cosine_similarity_matrix(probe_bank)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity, cmap="coolwarm", center=0, square=True)
    plt.title('Probe Similarity Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# Plot Registry
PLOT_REGISTRY: Dict[str, Callable] = {
    "training_curves": plot_training_history,
    "eta_distribution": plot_eta_distribution,
    "top_m_comparison": plot_top_m_comparison,
    "baseline_comparison": plot_baseline_comparison,
    "cdf": plot_cdf,
    "violin": plot_violin,
    "heatmap": plot_heatmap,
    "scatter": plot_scatter_comparison,
    "box": plot_box_comparison,
    "correlation_matrix": plot_correlation_matrix,
}


def register_plot(name: str, plot_function: Callable) -> None:
    """Register a new plot type."""
    PLOT_REGISTRY[name] = plot_function


def get_plot_function(name: str) -> Callable:
    """Get plot function by name."""
    if name not in PLOT_REGISTRY:
        raise ValueError(f"Plot '{name}' not found. Available: {list(PLOT_REGISTRY.keys())}")
    return PLOT_REGISTRY[name]


def list_plots() -> List[str]:
    """List all available plot types."""
    return list(PLOT_REGISTRY.keys())
