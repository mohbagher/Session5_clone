"""
No Coupling Model
=================
Baseline model with no mutual electromagnetic coupling.
"""

import numpy as np
from typing import Dict, Any, Optional
from src.ris_platform.core.interfaces import CouplingModel


class NoCoupling(CouplingModel):
    """
    No coupling baseline model.
    
    Γ_coupled = Γ_uncoupled (identity transformation)
    
    This is the baseline assumption where each RIS element operates
    independently without affecting its neighbors.
    """
    
    def __init__(self):
        """Initialize no coupling model."""
        pass
    
    def apply_coupling(
        self,
        gamma_uncoupled: np.ndarray,
        geometry: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Pass through without coupling (identity).
        
        Args:
            gamma_uncoupled: Uncoupled reflection coefficients (N,) or (N, K)
            geometry: RIS geometry (ignored)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Same as input (no coupling applied)
        """
        return gamma_uncoupled
    
    def get_coupling_matrix(
        self,
        geometry: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Get identity coupling matrix.
        
        Args:
            geometry: RIS geometry
            **kwargs: Additional parameters
            
        Returns:
            Identity matrix (N, N)
        """
        N = kwargs.get('N', 1)
        if geometry is not None and 'N' in geometry:
            N = geometry['N']
        return np.eye(N)


__all__ = ['NoCoupling']
