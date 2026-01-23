"""
Varactor Unit Cell Model
========================
Realistic varactor-based unit cell with amplitude-phase coupling
and thermal drift.

Reference: Dai et al., IEEE Access 2020
"""

import numpy as np
from typing import Dict, Any, Optional
from src.ris_platform.core.interfaces import UnitCellModel


class VaractorUnitCell(UnitCellModel):
    """
    Varactor-based unit cell with hardware realism.
    
    Model:
        A(θ) = 1 - α(1 + cos(θ))
        Γ(θ, T) = A(θ) * exp(j(θ + β*(T - T0)))
    
    Where:
        - α: Coupling strength (0.3 typical)
        - β: Thermal drift coefficient (0.02 rad/°C)
        - T0: Reference temperature (25°C)
    
    Properties:
    - Amplitude-phase coupling: amplitude varies with phase
    - Thermal drift: phase shifts with temperature
    - Hardware-realistic impedance matching
    """
    
    def __init__(
        self,
        coupling_strength: float = 0.3,
        thermal_drift_coeff: float = 0.02,
        reference_temp: float = 25.0
    ):
        """
        Initialize varactor unit cell.
        
        Args:
            coupling_strength: α parameter (amplitude coupling)
            thermal_drift_coeff: β parameter (rad/°C)
            reference_temp: T0 reference temperature (°C)
        """
        self.alpha = coupling_strength
        self.beta = thermal_drift_coeff
        self.T0 = reference_temp
    
    def compute_reflection(
        self,
        phases: np.ndarray,
        temperature: Optional[float] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Compute varactor reflection coefficients.
        
        Args:
            phases: Desired phase shifts (N,) or (N, K)
            temperature: Operating temperature in Celsius
            **kwargs: Additional parameters
            
        Returns:
            Complex reflection coefficients with coupling and drift
        """
        # Amplitude-phase coupling
        amplitude = 1.0 - self.alpha * (1.0 + np.cos(phases))
        
        # Thermal drift
        if temperature is not None:
            phase_shift = self.beta * (temperature - self.T0)
            effective_phases = phases + phase_shift
        else:
            effective_phases = phases
        
        # Complex reflection coefficient
        return amplitude * np.exp(1j * effective_phases)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return unit cell parameters."""
        return {
            'type': 'varactor',
            'coupling_strength': self.alpha,
            'thermal_drift_coeff': self.beta,
            'reference_temp': self.T0,
            'description': 'Varactor with amplitude-phase coupling and thermal drift'
        }
    
    def get_amplitude(self, phases: np.ndarray) -> np.ndarray:
        """
        Get amplitude response for given phases.
        
        Args:
            phases: Phase shifts (N,) or (N, K)
            
        Returns:
            Amplitude values
        """
        return 1.0 - self.alpha * (1.0 + np.cos(phases))
    
    def get_thermal_shift(self, temperature: float) -> float:
        """
        Get phase shift due to temperature.
        
        Args:
            temperature: Operating temperature in Celsius
            
        Returns:
            Phase shift in radians
        """
        return self.beta * (temperature - self.T0)


__all__ = ['VaractorUnitCell']
