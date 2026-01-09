"""
Task D2: Sanity Checks

Verify training loss decreases and validation eta improves.
Saves training curves and a summary of sanity check flags.
"""

import os
from typing import Dict

import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import get_config
from data_generation import create_dataloaders
from evaluation import evaluate_model
from experiments.probe_generators import get_probe_bank
from model import create_model
from training import train
from utils import plot_training_history


def run_task_d2(N: int = 32, K: int = 64, M: int = 8, seed: int = 42,
                results_dir: str = "results/D2_sanity_checks", verbose: bool = True) -> Dict:
    """
    Run Task D2: Sanity Checks.

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
        print("Task D2: Sanity Checks")
        print("="*70)

    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    config = get_config(
        system={'N': N, 'K': K, 'M': M},
        data={'n_train': 20000, 'n_val': 2000, 'n_test': 2000, 'seed': seed},
        training={'n_epochs': 80, 'batch_size': 128}
    )

    probe_bank = get_probe_bank('continuous', N, K, seed)
    train_loader, val_loader, test_loader, metadata = create_dataloaders(config, probe_bank)

    if verbose:
        print("Training model for sanity checks...")
    model = create_model(config)
    model, history = train(model, train_loader, val_loader, config, metadata)

    eval_results = evaluate_model(
        model, test_loader, config,
        metadata['test_powers_full'],
        metadata['test_labels'],
        metadata['test_observed_indices'],
        metadata['test_optimal_powers']
    )

    # Sanity checks
    train_loss_start = history.train_loss[0]
    train_loss_end = history.train_loss[-1]
    val_eta_start = history.val_eta[0]
    val_eta_end = history.val_eta[-1]
    train_acc_end = history.train_acc[-1]
    val_acc_end = history.val_acc[-1]

    loss_decreasing = train_loss_end < train_loss_start
    val_eta_increasing = val_eta_end > val_eta_start
    generalization_gap = train_acc_end - val_acc_end

    # Save training history plot
    if verbose:
        print("Saving training curves...")
    history_plot_path = os.path.join(plots_dir, "training_history.png")
    plot_training_history(history, save_path=history_plot_path)

    metrics_path = os.path.join(results_dir, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Task D2: Sanity Checks Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  N (RIS elements): {N}\n")
        f.write(f"  K (probes): {K}\n")
        f.write(f"  M (sensing budget): {M}\n")
        f.write(f"  Seed: {seed}\n\n")
        f.write("Sanity Check Metrics:\n")
        f.write(f"  Train loss start: {train_loss_start:.4f}\n")
        f.write(f"  Train loss end:   {train_loss_end:.4f}\n")
        f.write(f"  Val η start:      {val_eta_start:.4f}\n")
        f.write(f"  Val η end:        {val_eta_end:.4f}\n")
        f.write(f"  Train acc end:    {train_acc_end:.4f}\n")
        f.write(f"  Val acc end:      {val_acc_end:.4f}\n")
        f.write(f"  Generalization gap (train-val): {generalization_gap:.4f}\n\n")
        f.write("Sanity Flags:\n")
        f.write(f"  Loss decreasing:  {loss_decreasing}\n")
        f.write(f"  Val η improving:  {val_eta_increasing}\n")

        f.write("\nTest Performance:\n")
        f.write(f"  η_top1: {eval_results.eta_top1:.4f}\n")

    if verbose:
        print(f"  Saved: {metrics_path}")

    return {
        'history': history,
        'eval_results': eval_results,
        'loss_decreasing': loss_decreasing,
        'val_eta_increasing': val_eta_increasing,
        'generalization_gap': generalization_gap,
        'plots': [history_plot_path],
        'metrics_file': metrics_path
    }
