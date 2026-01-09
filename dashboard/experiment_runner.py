"""
Experiment Runner for RIS PhD Ultimate Dashboard.

Main experiment execution engine with support for single runs,
multi-model comparison, and multi-seed experiments.
"""

import sys
import os
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config, SystemConfig, DataConfig, ModelConfig, TrainingConfig, EvalConfig
from experiments.probe_generators import get_probe_bank
from data_generation import generate_dataset, create_dataloaders
from model import LimitedProbingMLP
from training import train
from evaluation import evaluate_model
from model_registry import get_model_architecture


@dataclass
class ExperimentResults:
    """Container for experiment results."""
    config: Dict[str, Any]
    evaluation: Any  # EvaluationResults object
    training_history: Any  # TrainingHistory object
    model_state: Optional[Dict] = None
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for saving."""
        result_dict = {
            'config': self.config,
            'execution_time': self.execution_time,
        }
        
        if self.evaluation:
            result_dict['evaluation'] = self.evaluation.to_dict()
        
        if self.training_history:
            result_dict['training_history'] = {
                'train_loss': self.training_history.train_loss,
                'val_loss': self.training_history.val_loss,
                'train_acc': self.training_history.train_acc,
                'val_acc': self.training_history.val_acc,
                'val_eta': self.training_history.val_eta,
                'learning_rates': self.training_history.learning_rates,
            }
        
        return result_dict


def create_config_from_dict(config_dict: Dict[str, Any]) -> Config:
    """
    Create Config object from configuration dictionary.
    
    Args:
        config_dict: Dictionary with all configuration parameters
        
    Returns:
        Config object
    """
    # System configuration
    system_config = SystemConfig(
        N=config_dict.get('N', 32),
        K=config_dict.get('K', 64),
        M=config_dict.get('M', 8),
        P_tx=config_dict.get('P_tx', 1.0),
        sigma_h_sq=config_dict.get('sigma_h_sq', 1.0),
        sigma_g_sq=config_dict.get('sigma_g_sq', 1.0),
        phase_mode=config_dict.get('phase_mode', 'continuous'),
        phase_bits=config_dict.get('phase_bits', 3),
        probe_type=config_dict.get('probe_type', 'continuous')
    )
    
    # Data configuration
    data_config = DataConfig(
        n_train=config_dict.get('n_train', 50000),
        n_val=config_dict.get('n_val', 5000),
        n_test=config_dict.get('n_test', 5000),
        seed=config_dict.get('seed', 42),
        normalize_input=config_dict.get('normalize_input', True),
        normalization_type=config_dict.get('normalization_type', 'mean')
    )
    
    # Model configuration
    model_config = ModelConfig(
        hidden_sizes=config_dict.get('hidden_sizes', [256, 128]),
        dropout_prob=config_dict.get('dropout_prob', 0.1),
        use_batch_norm=config_dict.get('use_batch_norm', True)
    )
    
    # Training configuration
    training_config = TrainingConfig(
        batch_size=config_dict.get('batch_size', 128),
        learning_rate=config_dict.get('learning_rate', 1e-3),
        weight_decay=config_dict.get('weight_decay', 1e-4),
        n_epochs=config_dict.get('n_epochs', 50),
        early_stop_patience=config_dict.get('early_stop_patience', 10)
    )
    
    # Evaluation configuration
    top_m_values = config_dict.get('top_m_values', [1, 2, 4, 8])
    eval_config = EvalConfig(
        top_m_values=[int(m) for m in top_m_values]
    )
    
    # Create and return config
    config = Config(
        system=system_config,
        data=data_config,
        model=model_config,
        training=training_config,
        eval=eval_config
    )
    
    return config


def run_single_experiment(config_dict: Dict[str, Any], 
                         progress_callback=None,
                         verbose: bool = True) -> ExperimentResults:
    """
    Run a single experiment with given configuration.
    
    Args:
        config_dict: Dictionary with all configuration parameters
        progress_callback: Optional callback function(epoch, total_epochs, metrics)
        verbose: Whether to print progress messages
        
    Returns:
        ExperimentResults object
    """
    start_time = time.time()
    
    try:
        # Create config object
        config = create_config_from_dict(config_dict)
        
        if verbose:
            print("=" * 70)
            print("STARTING EXPERIMENT")
            print("=" * 70)
            config.print_config()
        
        # Generate probe bank
        if verbose:
            print("\n[1/5] Generating probe bank...")
        probe_bank = get_probe_bank(
            probe_type=config.system.probe_type,
            N=config.system.N,
            K=config.system.K,
            seed=config.data.seed
        )
        
        # Generate dataset
        if verbose:
            print("[2/5] Generating dataset...")
        dataset_train, dataset_val, dataset_test = generate_dataset(
            config=config,
            probe_bank=probe_bank
        )
        
        # Create dataloaders
        if verbose:
            print("[3/5] Creating dataloaders...")
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_train, dataset_val, dataset_test,
            batch_size=config.training.batch_size
        )
        
        # Create model
        if verbose:
            print("[4/5] Training model...")
        model = LimitedProbingMLP(
            K=config.system.K,
            hidden_sizes=config.model.hidden_sizes,
            dropout_prob=config.model.dropout_prob,
            use_batch_norm=config.model.use_batch_norm
        )
        
        # Train model
        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            probe_bank=probe_bank,
            progress_callback=progress_callback,
            verbose=verbose
        )
        
        # Evaluate model
        if verbose:
            print("[5/5] Evaluating model...")
        results = evaluate_model(
            model=model,
            test_loader=test_loader,
            probe_bank=probe_bank,
            config=config,
            verbose=verbose
        )
        
        execution_time = time.time() - start_time
        
        if verbose:
            print(f"\n✅ Experiment completed in {execution_time:.1f}s")
            print(f"   Top-1 Accuracy: {results.accuracy_top1:.3f}")
            print(f"   η_top1: {results.eta_top1:.4f}")
        
        return ExperimentResults(
            config=config_dict,
            evaluation=results,
            training_history=history,
            model_state=model.state_dict(),
            execution_time=execution_time
        )
        
    except Exception as e:
        if verbose:
            print(f"\n❌ Experiment failed: {str(e)}")
        raise


def run_multi_model_comparison(base_config_dict: Dict[str, Any],
                               model_names: List[str],
                               progress_callback=None,
                               verbose: bool = True) -> Dict[str, ExperimentResults]:
    """
    Run experiments comparing multiple model architectures.
    
    Args:
        base_config_dict: Base configuration dictionary
        model_names: List of model names to compare
        progress_callback: Optional progress callback
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping model names to ExperimentResults
    """
    results_dict = {}
    
    if verbose:
        print("\n" + "=" * 70)
        print(f"MULTI-MODEL COMPARISON: {len(model_names)} models")
        print("=" * 70)
    
    for i, model_name in enumerate(model_names, 1):
        if verbose:
            print(f"\n[Model {i}/{len(model_names)}] {model_name}")
            print("-" * 70)
        
        # Create config for this model
        config_dict = base_config_dict.copy()
        
        # Get model architecture
        try:
            hidden_sizes = get_model_architecture(model_name)
            config_dict['hidden_sizes'] = hidden_sizes
            config_dict['model_preset'] = model_name
            
            # Run experiment
            results = run_single_experiment(
                config_dict=config_dict,
                progress_callback=progress_callback,
                verbose=verbose
            )
            
            results_dict[model_name] = results
            
        except Exception as e:
            if verbose:
                print(f"❌ Failed to run model {model_name}: {str(e)}")
            continue
    
    if verbose:
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        for model_name, results in results_dict.items():
            print(f"{model_name:20s}: η_top1={results.evaluation.eta_top1:.4f}, "
                  f"acc={results.evaluation.accuracy_top1:.3f}, "
                  f"time={results.execution_time:.1f}s")
    
    return results_dict


def run_multi_seed_experiment(config_dict: Dict[str, Any],
                              seeds: List[int],
                              progress_callback=None,
                              verbose: bool = True) -> List[ExperimentResults]:
    """
    Run experiments with multiple random seeds for statistical analysis.
    
    Args:
        config_dict: Base configuration dictionary
        seeds: List of random seeds to use
        progress_callback: Optional progress callback
        verbose: Whether to print progress
        
    Returns:
        List of ExperimentResults, one per seed
    """
    results_list = []
    
    if verbose:
        print("\n" + "=" * 70)
        print(f"MULTI-SEED EXPERIMENT: {len(seeds)} seeds")
        print("=" * 70)
    
    for i, seed in enumerate(seeds, 1):
        if verbose:
            print(f"\n[Seed {i}/{len(seeds)}] seed={seed}")
            print("-" * 70)
        
        # Create config for this seed
        config_dict_seed = config_dict.copy()
        config_dict_seed['seed'] = seed
        
        try:
            # Run experiment
            results = run_single_experiment(
                config_dict=config_dict_seed,
                progress_callback=progress_callback,
                verbose=verbose
            )
            
            results_list.append(results)
            
        except Exception as e:
            if verbose:
                print(f"❌ Failed with seed {seed}: {str(e)}")
            continue
    
    # Compute statistics
    if results_list and verbose:
        eta_values = [r.evaluation.eta_top1 for r in results_list]
        acc_values = [r.evaluation.accuracy_top1 for r in results_list]
        
        print("\n" + "=" * 70)
        print("MULTI-SEED SUMMARY")
        print("=" * 70)
        print(f"η_top1:  mean={np.mean(eta_values):.4f}, std={np.std(eta_values):.4f}")
        print(f"Accuracy: mean={np.mean(acc_values):.3f}, std={np.std(acc_values):.3f}")
    
    return results_list


def aggregate_results(results_list: List[ExperimentResults]) -> Dict[str, Any]:
    """
    Aggregate statistics from multiple experiment runs.
    
    Args:
        results_list: List of ExperimentResults
        
    Returns:
        Dictionary with aggregated statistics
    """
    if not results_list:
        return {}
    
    # Extract metrics
    eta_top1 = [r.evaluation.eta_top1 for r in results_list]
    eta_top2 = [r.evaluation.eta_top2 for r in results_list]
    eta_top4 = [r.evaluation.eta_top4 for r in results_list]
    eta_top8 = [r.evaluation.eta_top8 for r in results_list]
    
    acc_top1 = [r.evaluation.accuracy_top1 for r in results_list]
    acc_top2 = [r.evaluation.accuracy_top2 for r in results_list]
    acc_top4 = [r.evaluation.accuracy_top4 for r in results_list]
    acc_top8 = [r.evaluation.accuracy_top8 for r in results_list]
    
    execution_times = [r.execution_time for r in results_list]
    
    # Compute statistics
    aggregated = {
        'n_runs': len(results_list),
        
        'eta_top1': {
            'mean': float(np.mean(eta_top1)),
            'std': float(np.std(eta_top1)),
            'min': float(np.min(eta_top1)),
            'max': float(np.max(eta_top1)),
            'median': float(np.median(eta_top1)),
        },
        
        'eta_top2': {
            'mean': float(np.mean(eta_top2)),
            'std': float(np.std(eta_top2)),
        },
        
        'eta_top4': {
            'mean': float(np.mean(eta_top4)),
            'std': float(np.std(eta_top4)),
        },
        
        'eta_top8': {
            'mean': float(np.mean(eta_top8)),
            'std': float(np.std(eta_top8)),
        },
        
        'accuracy_top1': {
            'mean': float(np.mean(acc_top1)),
            'std': float(np.std(acc_top1)),
            'min': float(np.min(acc_top1)),
            'max': float(np.max(acc_top1)),
            'median': float(np.median(acc_top1)),
        },
        
        'accuracy_top2': {
            'mean': float(np.mean(acc_top2)),
            'std': float(np.std(acc_top2)),
        },
        
        'accuracy_top4': {
            'mean': float(np.mean(acc_top4)),
            'std': float(np.std(acc_top4)),
        },
        
        'accuracy_top8': {
            'mean': float(np.mean(acc_top8)),
            'std': float(np.std(acc_top8)),
        },
        
        'execution_time': {
            'mean': float(np.mean(execution_times)),
            'total': float(np.sum(execution_times)),
        },
    }
    
    # Confidence intervals (95%)
    if len(results_list) > 1:
        from scipy import stats
        
        def compute_ci(values):
            mean = np.mean(values)
            sem = stats.sem(values)
            ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=sem)
            return {'lower': float(ci[0]), 'upper': float(ci[1])}
        
        aggregated['eta_top1']['ci_95'] = compute_ci(eta_top1)
        aggregated['accuracy_top1']['ci_95'] = compute_ci(acc_top1)
    
    return aggregated
