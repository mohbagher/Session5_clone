"""
Simple run script for quick experiments.
"""

from config import get_config
from main import run_experiment
from utils import (
    plot_training_history,
    plot_eta_distribution,
    plot_top_m_comparison,
    plot_baseline_comparison,
    save_results
)
import os


def run_default():
    """Run with default parameters."""
    config = get_config()
    results = run_experiment(config)
    return results


def run_custom(N=32, K=64, M=8, n_train=50000):
    """Run with custom parameters."""
    config = get_config(
        system={'N': N, 'K':  K, 'M': M},
        data={'n_train':  n_train}
    )
    results = run_experiment(config)
    return results


def run_and_save(save_dir='results', **kwargs):
    """Run experiment and save all outputs."""
    config = get_config(**kwargs) if kwargs else get_config()
    experiment = run_experiment(config)
    
    os.makedirs(save_dir, exist_ok=True)
    
    save_results(
        experiment['results'],
        experiment['history'],
        config,
        save_dir
    )
    
    plot_training_history(
        experiment['history'],
        save_path=os.path.join(save_dir, 'training_history.png')
    )
    
    plot_eta_distribution(
        experiment['results'],
        save_path=os.path.join(save_dir, 'eta_distribution.png')
    )
    
    plot_top_m_comparison(
        experiment['results'],
        save_path=os.path.join(save_dir, 'top_m_comparison.png')
    )
    
    plot_baseline_comparison(
        experiment['results'],
        save_path=os.path.join(save_dir, 'baseline_comparison.png')
    )
    
    return experiment


if __name__ == "__main__":
    # Run with defaults
    results = run_and_save()