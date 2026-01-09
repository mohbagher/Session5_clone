"""
Main execution script for RIS probe-based control with limited probing.
"""

import torch
import numpy as np
import argparse
import os

from config import Config, get_config
from data_generation import create_dataloaders
from experiments.probe_generators import get_probe_bank
from model import create_model, count_parameters
from training import train
from evaluation import evaluate_model, compute_baselines_only
from utils import (
    plot_training_history,
    plot_eta_distribution,
    plot_top_m_comparison,
    plot_baseline_comparison,
    save_results
)


def run_experiment(config: Config, verbose: bool = True) -> dict:
    """Run complete experiment pipeline."""
    
    if verbose: 
        config.print_config()
    
    # Set seeds for reproducibility
    torch.manual_seed(config.data.seed)
    np.random.seed(config.data.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.data.seed)
    
    # Step 1: Generate fixed probe bank
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 1: Generating fixed probe bank")
        print("=" * 70)
    
    # Determine probe type (for backward compatibility)
    probe_type = getattr(config.system, 'probe_type', 'continuous')
    if not probe_type or probe_type == 'random':
        probe_type = 'continuous'
    
    probe_bank = get_probe_bank(
        probe_type=probe_type,
        N=config.system.N,
        K=config.system.K,
        seed=config.data.seed
    )
    
    if verbose:
        print(f"  Created {config.system.K} {probe_type} phase configurations")
        print(f"  Each probe has {config.system.N} phase values in [0, 2π)")
        print(f"  Probe bank shape: {probe_bank.phases.shape}")
    
    # Step 2: Create datasets with limited probing
    if verbose: 
        print("\n" + "=" * 70)
        print("STEP 2: Generating datasets with limited probing")
        print("=" * 70)
        print(f"  Sensing budget: M={config.system.M} probes observed per sample")
        print(f"  Total probes: K={config.system.K}")
        print(f"  Observation ratio: {config.system.M/config.system.K:.1%}")
    
    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        config, probe_bank
    )
    
    if verbose:
        print(f"\n  Train samples: {config.data.n_train}")
        print(f"  Val samples: {config.data.n_val}")
        print(f"  Test samples: {config.data.n_test}")
        print(f"  Train batches: {len(train_loader)}")
    
    # Step 3: Compute baselines before training
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 3: Computing baselines (no ML)")
        print("=" * 70)
    
    baselines = compute_baselines_only(
        metadata['test_powers_full'],
        metadata['test_labels'],
        metadata['test_observed_indices'],
        config.system.K,
        config.system.M,
        seed=config.data.seed
    )
    
    if verbose: 
        print(f"  Random 1/K baseline η: {baselines['eta_random_1_mean']:.4f} ± {baselines['eta_random_1_std']:.4f}")
        print(f"  Random M/K baseline η:  {baselines['eta_random_M_mean']:.4f} ± {baselines['eta_random_M_std']:.4f}")
        print(f"  Best observed baseline η: {baselines['eta_best_observed_mean']:.4f} ± {baselines['eta_best_observed_std']:.4f}")
        print(f"  Expected random accuracy: {baselines['expected_random_accuracy']:.4f}")
    
    # Step 4: Create model
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 4: Creating model")
        print("=" * 70)
    
    model = create_model(config)
    n_params = count_parameters(model)
    
    if verbose:
        print(f"  Architecture:  Masked Vector MLP")
        print(f"  Input size: {2 * config.system.K} (masked_powers + binary_mask)")
        print(f"  Hidden layers: {config.model.hidden_sizes}")
        print(f"  Output size: {config.system.K} (logits for each probe)")
        print(f"  Total parameters: {n_params:,}")
    
    # Step 5: Train model
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 5: Training model")
        print("=" * 70)
    
    model, history = train(model, train_loader, val_loader, config, metadata)
    
    # Step 6: Evaluate model
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 6: Evaluating model")
        print("=" * 70)
    
    results = evaluate_model(
        model,
        test_loader,
        config,
        metadata['test_powers_full'],
        metadata['test_labels'],
        metadata['test_observed_indices'],
        metadata['test_optimal_powers']
    )
    
    # Print results
    results.print_summary()
    
    return {
        'model': model,
        'history':  history,
        'results': results,
        'config': config,
        'probe_bank': probe_bank,
        'metadata': metadata,
        'baselines': baselines
    }


def main():
    """Main entry point with command line arguments."""
    parser = argparse.ArgumentParser(
        description='RIS Probe-Based Control with Limited Probing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # System parameters
    parser.add_argument('--N', type=int, default=32,
                        help='Number of RIS elements')
    parser.add_argument('--K', type=int, default=64,
                        help='Total number of probes in bank')
    parser.add_argument('--M', type=int, default=8,
                        help='Sensing budget (probes measured per sample)')
    parser.add_argument('--phase_mode', type=str, default='continuous',
                        choices=['continuous', 'discrete'],
                        help='Phase configuration mode')
    parser.add_argument('--phase_bits', type=int, default=3,
                        help='Phase quantization bits (discrete mode only)')
    
    # Data parameters
    parser.add_argument('--n_train', type=int, default=50000,
                        help='Number of training samples')
    parser.add_argument('--n_val', type=int, default=5000,
                        help='Number of validation samples')
    parser.add_argument('--n_test', type=int, default=5000,
                        help='Number of test samples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    
    # Model parameters
    parser.add_argument('--hidden', type=int, nargs='+', default=[512, 256, 128],
                        help='Hidden layer sizes')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--no_plots', action='store_true',
                        help='Disable plotting')
    
    args = parser.parse_args()
    
    # Validate M <= K
    if args.M > args.K:
        parser.error(f"M ({args.M}) cannot be greater than K ({args.K})")
    
    # Create configuration
    config = get_config(
        system={
            'N': args.N,
            'K': args.K,
            'M': args.M,
            'phase_mode': args.phase_mode,
            'phase_bits': args.phase_bits
        },
        data={
            'n_train': args.n_train,
            'n_val': args.n_val,
            'n_test': args.n_test,
            'seed': args.seed
        },
        training={
            'n_epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr
        },
        model={
            'hidden_sizes': args.hidden,
            'dropout_prob': args.dropout
        }
    )
    
    # Run experiment
    experiment = run_experiment(config)
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    
    save_results(
        experiment['results'],
        experiment['history'],
        config,
        args.save_dir
    )
    
    # Save model
    model_path = os.path.join(args.save_dir, 'model.pt')
    torch.save({
        'model_state_dict': experiment['model'].state_dict(),
        'config': {
            'N': config.system.N,
            'K': config.system.K,
            'M': config.system.M,
            'hidden_sizes': config.model.hidden_sizes
        },
        'results': experiment['results'].to_dict()
    }, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        
        plot_training_history(
            experiment['history'],
            save_path=os.path.join(args.save_dir, 'training_history.png')
        )
        
        plot_eta_distribution(
            experiment['results'],
            save_path=os.path.join(args.save_dir, 'eta_distribution.png')
        )
        
        plot_top_m_comparison(
            experiment['results'],
            save_path=os.path.join(args.save_dir, 'top_m_comparison.png')
        )
        
        plot_baseline_comparison(
            experiment['results'],
            save_path=os.path.join(args.save_dir, 'baseline_comparison.png')
        )
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    
    return experiment


if __name__ == "__main__":
    main()
