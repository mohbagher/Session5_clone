"""
Experiment Toolkit - Unified High-Level API for RIS Probe-Based ML Research.

Provides clean, fluent interface for configuring and running experiments.
"""

from typing import Dict, List, Optional, Union
import numpy as np

from config import Config, get_config
from experiments.probe_generators import get_probe_bank, ProbeBank
from data_generation import create_dataloaders
from model import create_model
from training import train
from evaluation import evaluate_model
from model_registry import MODEL_REGISTRY, get_model_architecture
from plot_registry import PLOT_REGISTRY, get_plot_function


class ConfigBuilder:
    """Fluent interface for building experiment configurations."""
    
    def __init__(self):
        self._system = {}
        self._data = {}
        self._model = {}
        self._training = {}
        self._eval = {}
    
    def system(self, N: int = 32, K: int = 64, M: int = 8, 
               phase_mode: str = "continuous", phase_bits: int = 3,
               probe_type: str = "continuous") -> 'ConfigBuilder':
        """Configure system parameters."""
        self._system = {
            'N': N, 'K': K, 'M': M,
            'phase_mode': phase_mode,
            'phase_bits': phase_bits,
            'probe_type': probe_type
        }
        return self
    
    def data(self, n_train: int = 50000, n_val: int = 5000, n_test: int = 5000,
             seed: int = 42) -> 'ConfigBuilder':
        """Configure data generation."""
        self._data = {
            'n_train': n_train,
            'n_val': n_val,
            'n_test': n_test,
            'seed': seed
        }
        return self
    
    def model(self, hidden_sizes: List[int], dropout: float = 0.1,
              batch_norm: bool = True) -> 'ConfigBuilder':
        """Configure model architecture."""
        self._model = {
            'hidden_sizes': hidden_sizes,
            'dropout_prob': dropout,
            'use_batch_norm': batch_norm
        }
        return self
    
    def model_by_name(self, model_name: str, dropout: float = 0.1,
                      batch_norm: bool = True) -> 'ConfigBuilder':
        """Configure model by registry name."""
        hidden_sizes = get_model_architecture(model_name)
        return self.model(hidden_sizes, dropout, batch_norm)
    
    def training(self, epochs: int = 50, batch_size: int = 128,
                 lr: float = 1e-3, patience: int = 10) -> 'ConfigBuilder':
        """Configure training parameters."""
        self._training = {
            'n_epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'early_stop_patience': patience
        }
        return self
    
    def build(self) -> Config:
        """Build final configuration."""
        return get_config(
            system=self._system,
            data=self._data,
            model=self._model,
            training=self._training
        )


class ExperimentRunner:
    """High-level experiment orchestration."""
    
    @staticmethod
    def run(config: Config, probe_type: Optional[str] = None, verbose: bool = True) -> Dict:
        """Run complete experiment with given configuration."""
        
        # Generate probe bank
        if probe_type is None:
            probe_type = getattr(config.system, 'probe_type', 'continuous')
        
        probe_bank = get_probe_bank(
            probe_type, 
            config.system.N, 
            config.system.K, 
            config.data.seed
        )
        
        # Create data
        train_loader, val_loader, test_loader, metadata = create_dataloaders(
            config, probe_bank
        )
        
        # Create and train model
        model = create_model(config)
        trained_model, history = train(model, train_loader, val_loader, config, metadata)
        
        # Evaluate
        results = evaluate_model(
            trained_model, test_loader, config,
            metadata['test_powers_full'],
            metadata['test_labels'],
            metadata['test_observed_indices'],
            metadata['test_optimal_powers']
        )
        
        if verbose:
            results.print_summary()
        
        return {
            'model': trained_model,
            'history': history,
            'results': results,
            'config': config,
            'probe_bank': probe_bank,
            'metadata': metadata
        }
    
    @staticmethod
    def compare_models(config_base: Config, model_names: List[str], 
                       verbose: bool = True) -> Dict[str, Dict]:
        """Compare multiple model architectures."""
        results = {}
        
        for model_name in model_names:
            if verbose:
                print(f"\n{'='*70}")
                print(f"Training: {model_name}")
                print('='*70)
            
            # Create a new config based on config_base but with different model architecture
            from copy import deepcopy
            config = deepcopy(config_base)
            config.model.hidden_sizes = get_model_architecture(model_name)
            
            result = ExperimentRunner.run(config, verbose=verbose)
            results[model_name] = result
        
        return results


class PlotGallery:
    """Unified plotting interface."""
    
    @staticmethod
    def show(plot_types: Union[str, List[str]], *args, **kwargs):
        """Show one or more plots."""
        if isinstance(plot_types, str):
            plot_types = [plot_types]
        
        for plot_type in plot_types:
            plot_func = get_plot_function(plot_type)
            plot_func(*args, **kwargs)
    
    @staticmethod
    def list_available() -> List[str]:
        """List all available plot types."""
        return list(PLOT_REGISTRY.keys())
