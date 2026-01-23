"""
Ideal Unit Cell Model
=====================
Perfect reflection with |Γ| = 1.

This is the baseline model used for ideal simulations.
"""

import numpy as np
from typing import Dict, Any, Optional
from src.ris_platform.core.interfaces import UnitCellModel


class IdealUnitCell(UnitCellModel):
    """
    Ideal unit cell with perfect reflection.
    
    Γ(θ) = exp(jθ)
    
    Properties:
    - Perfect amplitude: |Γ| = 1
    - No thermal drift
    - No amplitude-phase coupling
    """
    
    def __init__(self):
        """Initialize ideal unit cell."""
        pass
    
    def compute_reflection(
        self,
        phases: np.ndarray,
        temperature: Optional[float] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Compute perfect reflection coefficients.
        
        Args:
            phases: Desired phase shifts (N,) or (N, K)
            temperature: Operating temperature (ignored for ideal)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Complex reflection coefficients with |Γ| = 1
        """
        return np.exp(1j * phases)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return unit cell parameters."""
        return {
            'type': 'ideal',
            'amplitude': 1.0,
            'coupling_strength': 0.0,
            'thermal_drift': 0.0,
            'description': 'Perfect reflection (|Γ| = 1)'
        }


__all__ = ['IdealUnitCell']
