"""
Physics Models
==============
High-level physics simulation models that compose multiple components.
"""

import numpy as np
from typing import Dict, Any, Optional
from src.ris_platform.core.interfaces import (
    PhysicsModel,
    UnitCellModel,
    CouplingModel,
    WavefrontModel,
    ChannelAgingModel
)
from src.ris_platform.physics.components.unit_cell import IdealUnitCell
from src.ris_platform.physics.components.coupling import NoCoupling
from src.ris_platform.physics.components.wavefront import PlanarWavefront


class IdealPhysicsModel(PhysicsModel):
    """
    Ideal baseline physics model.
    
    Properties:
    - Perfect reflection (|Γ| = 1)
    - No coupling
    - Planar wavefront
    - No aging
    
    This is the standard textbook model used in most literature.
    """
    
    def __init__(self):
        """Initialize ideal physics model with no impairments."""
        self.unit_cell = IdealUnitCell()
        self.coupling = NoCoupling()
        self.wavefront = PlanarWavefront()
        self.aging = None
    
    def compute_received_power(
        self,
        h: np.ndarray,
        g: np.ndarray,
        phases: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Compute received power using ideal physics.
        
        Pipeline:
        1. Unit cell: Γ = exp(jθ)
        2. Coupling: Γ_coupled = Γ (no coupling)
        3. Wavefront: y = g^H @ (Γ ⊙ h)
        4. Power: |y|²
        
        Args:
            h: Base station to RIS channel (N,) or (N, K)
            g: RIS to user channel (N,) or (N, K)
            phases: RIS phase configuration (N,) or (N, K)
            **kwargs: Additional parameters (ignored for ideal)
            
        Returns:
            Received power(s)
        """
        # Unit cell reflection
        gamma = self.unit_cell.compute_reflection(phases)
        
        # No coupling applied
        gamma_coupled = self.coupling.apply_coupling(gamma)
        
        # Compute signal
        signal = self.wavefront.compute_signal(h, g, gamma_coupled)
        
        # Compute power
        power = np.abs(signal)**2
        
        return power
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about ideal model."""
        return {
            'name': 'IdealPhysicsModel',
            'unit_cell': self.unit_cell.get_parameters(),
            'coupling': 'none',
            'wavefront': 'planar',
            'aging': 'none',
            'description': 'Baseline ideal model with no impairments'
        }


class RealisticPhysicsModel(PhysicsModel):
    """
    Realistic physics model with composable components.
    
    This model allows injecting different components for:
    - Unit cell (ideal, varactor, etc.)
    - Coupling (none, geometric, etc.)
    - Wavefront (planar, spherical, etc.)
    - Aging (none, Jakes, etc.)
    
    This enables ablation studies by swapping components.
    """
    
    def __init__(
        self,
        unit_cell: Optional[UnitCellModel] = None,
        coupling: Optional[CouplingModel] = None,
        wavefront: Optional[WavefrontModel] = None,
        aging: Optional[ChannelAgingModel] = None
    ):
        """
        Initialize realistic physics model with dependency injection.
        
        Args:
            unit_cell: Unit cell model (default: IdealUnitCell)
            coupling: Coupling model (default: NoCoupling)
            wavefront: Wavefront model (default: PlanarWavefront)
            aging: Channel aging model (default: None)
        """
        self.unit_cell = unit_cell if unit_cell is not None else IdealUnitCell()
        self.coupling = coupling if coupling is not None else NoCoupling()
        self.wavefront = wavefront if wavefront is not None else PlanarWavefront()
        self.aging = aging
    
    def compute_received_power(
        self,
        h: np.ndarray,
        g: np.ndarray,
        phases: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Compute received power using realistic physics pipeline.
        
        Pipeline:
        1. Unit cell: Γ = f(θ, T) (may include coupling, thermal drift)
        2. Coupling: Γ_coupled = C @ Γ
        3. Wavefront: y = f(h, g, Γ_coupled) (planar or spherical)
        4. Power: |y|²
        
        Args:
            h: Base station to RIS channel (N,) or (N, K)
            g: RIS to user channel (N,) or (N, K)
            phases: RIS phase configuration (N,) or (N, K)
            **kwargs: Additional parameters (temperature, geometry, etc.)
            
        Returns:
            Received power(s)
        """
        # Extract optional parameters
        temperature = kwargs.get('temperature', None)
        geometry = kwargs.get('geometry', None)
        
        # Unit cell reflection
        gamma = self.unit_cell.compute_reflection(
            phases,
            temperature=temperature
        )
        
        # Apply coupling
        gamma_coupled = self.coupling.apply_coupling(
            gamma,
            geometry=geometry
        )
        
        # Compute signal with wavefront model
        signal = self.wavefront.compute_signal(
            h, g, gamma_coupled,
            geometry=geometry
        )
        
        # Compute power
        power = np.abs(signal)**2
        
        return power
    
    def apply_aging(
        self,
        h: np.ndarray,
        g: np.ndarray,
        time_delta: float,
        rng: Optional[np.random.RandomState] = None
    ):
        """
        Apply channel aging if enabled.
        
        Args:
            h: Current h channel
            g: Current g channel
            time_delta: Time step in seconds
            rng: Random number generator
            
        Returns:
            Tuple of (h_aged, g_aged) or (h, g) if no aging
        """
        if self.aging is not None:
            return self.aging.age_channel(h, g, time_delta, rng)
        else:
            return h, g
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about realistic model and its components."""
        metadata = {
            'name': 'RealisticPhysicsModel',
            'unit_cell': self.unit_cell.get_parameters(),
            'coupling': type(self.coupling).__name__,
            'wavefront': type(self.wavefront).__name__,
            'aging': type(self.aging).__name__ if self.aging else 'none',
            'description': 'Composable realistic model with injected components'
        }
        
        # Add component-specific metadata
        if hasattr(self.coupling, 'get_parameters'):
            metadata['coupling_params'] = self.coupling.get_parameters()
        
        if hasattr(self.wavefront, 'wavelength'):
            metadata['wavelength'] = self.wavefront.wavelength
        
        if self.aging and hasattr(self.aging, 'doppler_hz'):
            metadata['doppler_hz'] = self.aging.doppler_hz
        
        return metadata


__all__ = ['IdealPhysicsModel', 'RealisticPhysicsModel']
