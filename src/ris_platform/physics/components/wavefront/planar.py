"""
Planar Wavefront Model
======================
Far-field assumption with planar wave propagation.

This is the standard model used in most RIS literature.
"""

import numpy as np
from typing import Dict, Any, Optional
from src.ris_platform.core.interfaces import WavefrontModel


class PlanarWavefront(WavefrontModel):
    """
    Planar wavefront model for far-field propagation.
    
    Model:
        y = g^H @ (Γ ⊙ h)
    
    Where:
    - ⊙ denotes element-wise (Hadamard) product
    - Assumes far-field conditions
    - No distance-dependent phase or path loss
    
    This is the simplified model valid when all distances are
    much greater than the Rayleigh distance.
    """
    
    def __init__(self):
        """Initialize planar wavefront model."""
        pass
    
    def compute_signal(
        self,
        h: np.ndarray,
        g: np.ndarray,
        gamma: np.ndarray,
        geometry: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Compute received signal using planar wavefront.
        
        y = g^H @ (Γ ⊙ h)
        
        Args:
            h: Base station to RIS channel (N,) or (N, K)
            g: RIS to user channel (N,) or (N, K)
            gamma: RIS reflection coefficients (N,) or (N, K)
            geometry: Spatial geometry (ignored for planar)
            **kwargs: Additional parameters
            
        Returns:
            Received signal(s)
        """
        # Element-wise product
        reflected = gamma * h
        
        # Inner product with g
        if h.ndim == 1:
            # Single snapshot
            return np.vdot(g, reflected)
        else:
            # Multiple snapshots: (N, K)
            # Compute for each column
            return np.sum(np.conj(g) * reflected, axis=0)
    
    def compute_rayleigh_distance(
        self,
        geometry: Dict[str, Any],
        **kwargs
    ) -> float:
        """
        Compute Rayleigh distance (not used for planar).
        
        For planar wavefront, this always returns infinity since
        we assume far-field conditions everywhere.
        
        Args:
            geometry: Spatial geometry
            **kwargs: Additional parameters
            
        Returns:
            np.inf (always far-field)
        """
        return np.inf


__all__ = ['PlanarWavefront']
