"""
Grid Search Optimizer
=====================
Exhaustive grid search for small parameter spaces.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Callable, Optional
import itertools
import logging
from src.ris_platform.core.interfaces import Optimizer

logger = logging.getLogger(__name__)


class GridSearchOptimizer(Optimizer):
    """
    Exhaustive grid search optimizer.
    
    Features:
    - Complete coverage of parameter space
    - Deterministic results
    - Best for small parameter spaces
    
    Use cases:
    - Validation of Bayesian optimization results
    - Small parameter spaces (< 1000 combinations)
    - When computational cost is low
    """
    
    def __init__(self, direction: str = "minimize"):
        """
        Initialize grid search optimizer.
        
        Args:
            direction: "minimize" or "maximize"
        """
        self.direction = direction
        self._history = []
    
    def optimize(
        self,
        objective_fn: Callable,
        search_space: Dict[str, Any],
        n_trials: int,
        **kwargs
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run grid search optimization.
        
        Args:
            objective_fn: Function to optimize (takes params dict, returns scalar)
            search_space: Parameter grid specification
                Example: {
                    'learning_rate': [1e-4, 1e-3, 1e-2],
                    'batch_size': [16, 32, 64, 128],
                    'dropout': [0.0, 0.1, 0.2, 0.3]
                }
            n_trials: Maximum number of trials (may be less if grid is smaller)
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (best_params, best_value)
        """
        logger.info("Starting grid search optimization")
        
        # Generate all combinations
        param_names = list(search_space.keys())
        param_values = [search_space[name] for name in param_names]
        
        all_combinations = list(itertools.product(*param_values))
        
        logger.info(f"Grid size: {len(all_combinations)} combinations")
        
        # Limit to n_trials if specified
        if n_trials < len(all_combinations):
            logger.warning(
                f"Grid has {len(all_combinations)} points but n_trials={n_trials}. "
                f"Will evaluate all {len(all_combinations)} points."
            )
        
        # Evaluate all combinations
        best_value = np.inf if self.direction == "minimize" else -np.inf
        best_params = None
        
        for i, combination in enumerate(all_combinations):
            # Create params dict
            params = dict(zip(param_names, combination))
            
            # Evaluate objective
            value = objective_fn(params)
            
            # Store in history
            self._history.append({
                'trial': i,
                'params': params.copy(),
                'value': value
            })
            
            # Update best
            is_better = (
                (self.direction == "minimize" and value < best_value) or
                (self.direction == "maximize" and value > best_value)
            )
            
            if is_better:
                best_value = value
                best_params = params.copy()
            
            # Log progress
            if (i + 1) % 10 == 0 or (i + 1) == len(all_combinations):
                logger.info(
                    f"Progress: {i+1}/{len(all_combinations)} "
                    f"(best so far: {best_value:.6f})"
                )
        
        logger.info(f"Grid search complete. Best value: {best_value:.6f}")
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
        Save optimization state to JSON.
        
        Args:
            filepath: Path to save state
            
        Returns:
            True if successful
        """
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump({
                    'direction': self.direction,
                    'history': self._history
                }, f, indent=2)
            logger.info(f"Grid search state saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False
    
    def load_state(self, filepath: str) -> bool:
        """
        Load optimization state from JSON.
        
        Args:
            filepath: Path to load state from
            
        Returns:
            True if successful
        """
        try:
            import json
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.direction = state['direction']
            self._history = state['history']
            
            logger.info(f"Grid search state loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False
    
    def plot_results(
        self,
        param_x: str,
        param_y: str,
        save_path: Optional[str] = None
    ):
        """
        Plot 2D grid search results.
        
        Args:
            param_x: Parameter for x-axis
            param_y: Parameter for y-axis
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            
            # Extract data
            x_vals = [h['params'][param_x] for h in self._history]
            y_vals = [h['params'][param_y] for h in self._history]
            z_vals = [h['value'] for h in self._history]
            
            # Create scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(x_vals, y_vals, c=z_vals, cmap=cm.viridis, s=100)
            ax.set_xlabel(param_x)
            ax.set_ylabel(param_y)
            ax.set_title('Grid Search Results')
            fig.colorbar(scatter, ax=ax, label='Objective Value')
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Plot saved to {save_path}")
            else:
                plt.show()
        except Exception as e:
            logger.error(f"Failed to plot: {e}")


__all__ = ['GridSearchOptimizer']
