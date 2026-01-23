"""
Bayesian Optimizer
==================
Optuna-based Bayesian optimization with TPE algorithm.

This enables hyperparameter tuning and system optimization.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
import logging
from src.ris_platform.core.interfaces import Optimizer

logger = logging.getLogger(__name__)

# Conditional import for Optuna
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None


class BayesianOptimizer(Optimizer):
    """
    Bayesian optimization using Optuna's TPE algorithm.
    
    Features:
    - Tree-structured Parzen Estimator (TPE)
    - SQLite persistence for resumable experiments
    - Parallel trial support
    - Pruning for early stopping
    
    Use cases:
    - Hyperparameter tuning (learning rate, batch size, etc.)
    - Probe strategy optimization
    - Physics parameter fitting
    """
    
    def __init__(
        self,
        study_name: str = "ris_optimization",
        storage: Optional[str] = None,
        direction: str = "minimize",
        n_jobs: int = 1
    ):
        """
        Initialize Bayesian optimizer.
        
        Args:
            study_name: Name for the optimization study
            storage: SQLite database path (None = in-memory)
            direction: "minimize" or "maximize"
            n_jobs: Number of parallel jobs (1 = sequential)
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna not installed. Install with: pip install optuna>=3.0.0"
            )
        
        self.study_name = study_name
        self.storage = storage
        self.direction = direction
        self.n_jobs = n_jobs
        self.study = None
        self._history = []
    
    def optimize(
        self,
        objective_fn: Callable,
        search_space: Dict[str, Any],
        n_trials: int,
        **kwargs
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run Bayesian optimization.
        
        Args:
            objective_fn: Function to optimize (takes params dict, returns scalar)
            search_space: Parameter search space specification
                Example: {
                    'learning_rate': ('float', 1e-4, 1e-2, 'log'),
                    'batch_size': ('int', 16, 128),
                    'dropout': ('float', 0.0, 0.5)
                }
            n_trials: Number of optimization trials
            **kwargs: Additional Optuna parameters
            
        Returns:
            Tuple of (best_params, best_value)
        """
        logger.info(f"Starting Bayesian optimization: {n_trials} trials")
        
        # Create study
        if self.storage is not None:
            storage_url = f"sqlite:///{self.storage}"
        else:
            storage_url = None
        
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=storage_url,
            direction=self.direction,
            load_if_exists=True
        )
        
        # Create objective wrapper that handles search space
        def wrapped_objective(trial):
            # Sample parameters from search space
            params = {}
            for param_name, param_spec in search_space.items():
                if param_spec[0] == 'float':
                    if len(param_spec) == 4 and param_spec[3] == 'log':
                        params[param_name] = trial.suggest_float(
                            param_name, param_spec[1], param_spec[2], log=True
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, param_spec[1], param_spec[2]
                        )
                elif param_spec[0] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_spec[1], param_spec[2]
                    )
                elif param_spec[0] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_spec[1]
                    )
                else:
                    raise ValueError(f"Unknown parameter type: {param_spec[0]}")
            
            # Evaluate objective
            value = objective_fn(params)
            
            # Store in history
            self._history.append({
                'trial': trial.number,
                'params': params.copy(),
                'value': value
            })
            
            return value
        
        # Run optimization
        self.study.optimize(
            wrapped_objective,
            n_trials=n_trials,
            n_jobs=self.n_jobs,
            **kwargs
        )
        
        # Get best results
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        logger.info(f"Optimization complete. Best value: {best_value:.6f}")
        logger.info(f"Best params: {best_params}")
        
        return best_params, best_value
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """
        Get history of optimization trials.
        
        Returns:
            List of trial results with params and values
        """
        return self._history
    
    def save_state(self, filepath: str) -> bool:
        """
        Save optimization state.
        
        Note: If using SQLite storage, state is automatically saved.
        This method saves the in-memory history.
        
        Args:
            filepath: Path to save state (JSON)
            
        Returns:
            True if successful
        """
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump({
                    'study_name': self.study_name,
                    'direction': self.direction,
                    'history': self._history
                }, f, indent=2)
            logger.info(f"Optimization state saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False
    
    def load_state(self, filepath: str) -> bool:
        """
        Load optimization state.
        
        Args:
            filepath: Path to load state from (JSON)
            
        Returns:
            True if successful
        """
        try:
            import json
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.study_name = state['study_name']
            self.direction = state['direction']
            self._history = state['history']
            
            logger.info(f"Optimization state loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Plot optimization history.
        
        Args:
            save_path: Optional path to save plot
        """
        if not OPTUNA_AVAILABLE or self.study is None:
            logger.warning("No study available for plotting")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig = optuna.visualization.matplotlib.plot_optimization_history(self.study)
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Plot saved to {save_path}")
            else:
                plt.show()
        except Exception as e:
            logger.error(f"Failed to plot: {e}")


__all__ = ['BayesianOptimizer']
