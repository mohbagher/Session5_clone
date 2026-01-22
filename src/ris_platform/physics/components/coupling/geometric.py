"""
Geometric Coupling Model
========================
Bessel function-based mutual electromagnetic coupling.

Reference: Pozar, Microwave Engineering, Ch. 8
"""

import numpy as np
from typing import Dict, Any, Optional
from scipy.special import jv  # Bessel function
from src.ris_platform.core.interfaces import CouplingModel


class GeometricCoupling(CouplingModel):
    """
    Geometric mutual coupling using Bessel functions.
    
    Model:
        C_ij = J₀(k * d_ij) for i ≠ j
        C_ii = 1
    
    Where:
        - J₀: Bessel function of the first kind, order 0
        - k: Wave number (2π/λ)
        - d_ij: Distance between elements i and j
    
    Properties:
    - Distance-dependent coupling
    - Physically-motivated (spherical wave expansion)
    - Includes coupling matrix caching
    - Optional distance cutoff for large arrays
    """
    
    def __init__(
        self,
        coupling_strength: float = 1.0,
        wavelength: float = 0.125,  # 2.4 GHz default
        distance_cutoff: Optional[float] = None,
        cache_enabled: bool = True
    ):
        """
        Initialize geometric coupling model.
        
        Args:
            coupling_strength: Scaling factor for coupling
            wavelength: Operating wavelength in meters
            distance_cutoff: Maximum distance for coupling (None = no cutoff)
            cache_enabled: Enable coupling matrix caching
        """
        self.strength = coupling_strength
        self.wavelength = wavelength
        self.k = 2.0 * np.pi / wavelength  # Wave number
        self.distance_cutoff = distance_cutoff
        self.cache_enabled = cache_enabled
        self._cache = {}
    
    def apply_coupling(
        self,
        gamma_uncoupled: np.ndarray,
        geometry: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Apply mutual coupling via matrix multiplication.
        
        Γ_coupled = C @ Γ_uncoupled
        
        Args:
            gamma_uncoupled: Uncoupled reflection coefficients (N,) or (N, K)
            geometry: RIS geometry with positions
            **kwargs: Additional parameters
            
        Returns:
            Coupled reflection coefficients
        """
        C = self.get_coupling_matrix(geometry, **kwargs)
        
        # Handle different input shapes
        if gamma_uncoupled.ndim == 1:
            # Single snapshot: (N,)
            return C @ gamma_uncoupled
        elif gamma_uncoupled.ndim == 2:
            # Multiple snapshots: (N, K)
            return C @ gamma_uncoupled
        else:
            raise ValueError(f"Unsupported gamma shape: {gamma_uncoupled.shape}")
    
    def get_coupling_matrix(
        self,
        geometry: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Compute or retrieve cached coupling matrix.
        
        Args:
            geometry: Dict with 'positions' (N, 2) or (N, 3) array
            **kwargs: Additional parameters
            
        Returns:
            Coupling matrix C (N, N)
        """
        # Get geometry hash for caching
        if geometry is not None and 'positions' in geometry:
            positions = geometry['positions']
            N = len(positions)
            
            # Create cache key
            if self.cache_enabled:
                cache_key = self._get_geometry_hash(positions)
                if cache_key in self._cache:
                    return self._cache[cache_key]
        else:
            # Default: uniform linear array
            N = kwargs.get('N', 1)
            spacing = kwargs.get('spacing', self.wavelength / 2)
            positions = np.column_stack([np.arange(N) * spacing, np.zeros(N)])
            cache_key = None
        
        # Compute coupling matrix
        C = self._compute_coupling_matrix(positions)
        
        # Cache if enabled
        if self.cache_enabled and cache_key is not None:
            self._cache[cache_key] = C
        
        return C
    
    def _compute_coupling_matrix(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute coupling matrix from element positions.
        
        Args:
            positions: Element positions (N, 2) or (N, 3)
            
        Returns:
            Coupling matrix (N, N)
        """
        N = len(positions)
        C = np.eye(N, dtype=complex)
        
        for i in range(N):
            for j in range(i + 1, N):
                # Compute distance
                d_ij = np.linalg.norm(positions[i] - positions[j])
                
                # Apply distance cutoff if specified
                if self.distance_cutoff is not None and d_ij > self.distance_cutoff:
                    coupling = 0.0
                else:
                    # Bessel function coupling
                    coupling = self.strength * jv(0, self.k * d_ij)
                
                # Symmetric coupling
                C[i, j] = coupling
                C[j, i] = coupling
        
        return C
    
    def _get_geometry_hash(self, positions: np.ndarray) -> str:
        """
        Create hash key for geometry caching.
        
        Args:
            positions: Element positions
            
        Returns:
            Hash string
        """
        # Simple hash based on positions
        return str(hash(positions.tobytes()))
    
    def clear_cache(self):
        """Clear coupling matrix cache."""
        self._cache = {}


__all__ = ['GeometricCoupling']
