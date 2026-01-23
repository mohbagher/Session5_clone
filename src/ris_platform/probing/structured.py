"""
Structured Probing Strategies
=============================
Intelligent probe selection strategies.
"""

import numpy as np
from typing import Dict, Any, Optional
from scipy.stats import qmc  # Quasi-Monte Carlo
from src.ris_platform.core.interfaces import ProbingStrategy


class RandomProbing(ProbingStrategy):
    """
    Uniform random probe selection.
    
    Baseline strategy that selects M probes uniformly at random
    from K available probes.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random probing.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def select_probes(
        self,
        K: int,
        M: int,
        feedback: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Select M random probes from K total.
        
        Args:
            K: Total number of available probes
            M: Number of probes to select
            feedback: Optional feedback (ignored for random)
            **kwargs: Additional parameters
            
        Returns:
            Array of selected probe indices (M,)
        """
        if M > K:
            raise ValueError(f"Cannot select M={M} probes from K={K} total")
        
        # Uniform random selection without replacement
        indices = self.rng.choice(K, size=M, replace=False)
        return indices
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': 'RandomProbing',
            'description': 'Uniform random selection',
            'seed': self.seed,
            'adaptive': False
        }


class SobolProbing(ProbingStrategy):
    """
    Low-discrepancy quasi-random sampling using Sobol sequences.
    
    Sobol sequences provide better coverage of the probe space
    than uniform random sampling, potentially improving channel
    estimation quality.
    
    Reference: Sobol, USSR Computational Mathematics, 1967
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize Sobol probing.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self._sampler = None
    
    def select_probes(
        self,
        K: int,
        M: int,
        feedback: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Select M probes using Sobol sequence.
        
        Args:
            K: Total number of available probes
            M: Number of probes to select
            feedback: Optional feedback (ignored)
            **kwargs: Additional parameters
            
        Returns:
            Array of selected probe indices (M,)
        """
        if M > K:
            raise ValueError(f"Cannot select M={M} probes from K={K} total")
        
        # Create Sobol sampler
        sampler = qmc.Sobol(d=1, scramble=True, seed=self.seed)
        
        # Generate M samples in [0, 1)
        samples = sampler.random(M)
        
        # Scale to [0, K) and convert to integer indices
        indices = (samples[:, 0] * K).astype(int)
        
        # Ensure unique indices (shouldn't be an issue for Sobol)
        indices = np.unique(indices)
        
        # If we lost some due to duplicates, fill with remaining
        if len(indices) < M:
            remaining = list(set(range(K)) - set(indices))
            rng = np.random.RandomState(self.seed)
            extra = rng.choice(remaining, size=M - len(indices), replace=False)
            indices = np.concatenate([indices, extra])
        
        return indices[:M]
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': 'SobolProbing',
            'description': 'Low-discrepancy quasi-random sampling',
            'seed': self.seed,
            'adaptive': False,
            'reference': 'Sobol, 1967'
        }


class HadamardProbing(ProbingStrategy):
    """
    Structured orthogonal probing using Hadamard matrices.
    
    Hadamard probes are mutually orthogonal, which can improve
    channel estimation under certain conditions.
    
    Note: K must be a power of 2 for standard Hadamard matrices.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize Hadamard probing.
        
        Args:
            seed: Random seed for row selection
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def select_probes(
        self,
        K: int,
        M: int,
        feedback: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Select M probes corresponding to Hadamard matrix rows.
        
        Args:
            K: Total number of available probes
            M: Number of probes to select
            feedback: Optional feedback (ignored)
            **kwargs: Additional parameters
            
        Returns:
            Array of selected probe indices (M,)
        """
        if M > K:
            raise ValueError(f"Cannot select M={M} probes from K={K} total")
        
        # Find nearest power of 2 >= K
        n = int(2**np.ceil(np.log2(K)))
        
        # Generate Hadamard matrix
        H = self._hadamard_matrix(n)
        
        # Select M random rows (corresponding to probes)
        # We map these back to [0, K) range
        row_indices = self.rng.choice(n, size=M, replace=False)
        
        # Map to [0, K) by taking modulo
        probe_indices = row_indices % K
        
        # Ensure uniqueness
        probe_indices = np.unique(probe_indices)
        
        # If we lost some, fill with random
        if len(probe_indices) < M:
            remaining = list(set(range(K)) - set(probe_indices))
            extra = self.rng.choice(remaining, size=M - len(probe_indices), replace=False)
            probe_indices = np.concatenate([probe_indices, extra])
        
        return probe_indices[:M]
    
    def _hadamard_matrix(self, n: int) -> np.ndarray:
        """
        Generate Hadamard matrix of size n x n (Sylvester construction).
        
        Args:
            n: Size (must be power of 2)
            
        Returns:
            Hadamard matrix (n, n)
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        
        # Check power of 2
        if n & (n - 1) != 0:
            raise ValueError("n must be power of 2")
        
        # Recursive construction
        if n == 1:
            return np.array([[1]])
        else:
            H_half = self._hadamard_matrix(n // 2)
            return np.block([
                [H_half, H_half],
                [H_half, -H_half]
            ])
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': 'HadamardProbing',
            'description': 'Structured orthogonal probes',
            'seed': self.seed,
            'adaptive': False,
            'note': 'Best for K = power of 2'
        }


__all__ = ['RandomProbing', 'SobolProbing', 'HadamardProbing']
