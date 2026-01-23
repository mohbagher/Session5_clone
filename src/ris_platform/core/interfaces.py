"""
Core Interfaces for RIS Platform
=================================
Abstract Base Classes for dependency injection and composable architecture.

This module defines the contracts for:
- Physics models and components
- Backend abstractions
- Probing strategies
- Hardware drivers
- Optimization engines
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime
import numpy as np


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class HardwareStatus:
    """Hardware monitoring data."""
    timestamp: datetime
    battery_level: Optional[float] = None  # 0.0 to 1.0
    temperature: Optional[float] = None  # Celsius
    link_quality: Optional[float] = None  # dB
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class ChannelMeasurement:
    """Timestamped channel measurement."""
    h: np.ndarray  # Base station to RIS channel
    g: np.ndarray  # RIS to user channel
    timestamp: datetime
    metadata: Dict[str, Any]


# ============================================================================
# PHYSICS INTERFACES
# ============================================================================

class PhysicsModel(ABC):
    """
    High-level physics simulation interface.
    
    Computes end-to-end system behavior given channels and RIS configuration.
    Implementations can be simple (ideal) or complex (realistic with impairments).
    """
    
    @abstractmethod
    def compute_received_power(
        self,
        h: np.ndarray,
        g: np.ndarray,
        phases: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Compute received power at user.
        
        Args:
            h: Base station to RIS channel (N,) or (N, K)
            g: RIS to user channel (N,) or (N, K)
            phases: RIS phase configuration (N,) or (N, K)
            **kwargs: Additional parameters (geometry, temperature, etc.)
            
        Returns:
            Received power(s) (scalar or array)
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return metadata about the physics model.
        
        Returns:
            Dictionary with model name, parameters, impairments, etc.
        """
        pass


class UnitCellModel(ABC):
    """
    RIS element reflection characteristics.
    
    Models the relationship between applied voltage/phase and the
    resulting complex reflection coefficient Γ.
    """
    
    @abstractmethod
    def compute_reflection(
        self,
        phases: np.ndarray,
        temperature: Optional[float] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Compute complex reflection coefficients.
        
        Args:
            phases: Desired phase shifts (N,) or (N, K)
            temperature: Operating temperature in Celsius
            **kwargs: Additional parameters
            
        Returns:
            Complex reflection coefficients Γ (same shape as phases)
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Return unit cell parameters."""
        pass


class CouplingModel(ABC):
    """
    Mutual electromagnetic coupling between RIS elements.
    
    Models how the reflection coefficient of one element affects
    its neighbors due to electromagnetic coupling.
    """
    
    @abstractmethod
    def apply_coupling(
        self,
        gamma_uncoupled: np.ndarray,
        geometry: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Apply mutual coupling to reflection coefficients.
        
        Args:
            gamma_uncoupled: Uncoupled reflection coefficients (N,) or (N, K)
            geometry: RIS geometry information (positions, spacing, etc.)
            **kwargs: Additional parameters
            
        Returns:
            Coupled reflection coefficients (same shape as input)
        """
        pass
    
    @abstractmethod
    def get_coupling_matrix(
        self,
        geometry: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Get coupling matrix C.
        
        Args:
            geometry: RIS geometry information
            **kwargs: Additional parameters
            
        Returns:
            Coupling matrix (N, N)
        """
        pass


class WavefrontModel(ABC):
    """
    Near-field/far-field propagation model.
    
    Models how the signal propagates from base station through RIS to user,
    accounting for spatial effects.
    """
    
    @abstractmethod
    def compute_signal(
        self,
        h: np.ndarray,
        g: np.ndarray,
        gamma: np.ndarray,
        geometry: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Compute received signal.
        
        Args:
            h: Base station to RIS channel (N,) or (N, K)
            g: RIS to user channel (N,) or (N, K)
            gamma: RIS reflection coefficients (N,) or (N, K)
            geometry: Spatial geometry information
            **kwargs: Additional parameters
            
        Returns:
            Received signal(s) (scalar or array)
        """
        pass
    
    @abstractmethod
    def compute_rayleigh_distance(
        self,
        geometry: Dict[str, Any],
        **kwargs
    ) -> float:
        """
        Compute Rayleigh distance for near-field determination.
        
        Args:
            geometry: Spatial geometry information
            **kwargs: Additional parameters
            
        Returns:
            Rayleigh distance in meters
        """
        pass


class ChannelAgingModel(ABC):
    """
    Temporal correlation (Doppler) model.
    
    Models how channels evolve over time due to mobility.
    """
    
    @abstractmethod
    def age_channel(
        self,
        h_current: np.ndarray,
        g_current: np.ndarray,
        time_delta: float,
        rng: Optional[np.random.RandomState] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evolve channels forward in time.
        
        Args:
            h_current: Current h channel (N,) or (N, K)
            g_current: Current g channel (N,) or (N, K)
            time_delta: Time step in seconds
            rng: Random number generator
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (h_aged, g_aged)
        """
        pass
    
    @abstractmethod
    def get_correlation_coefficient(
        self,
        time_delta: float,
        **kwargs
    ) -> float:
        """
        Get temporal correlation coefficient for time delta.
        
        Args:
            time_delta: Time step in seconds
            **kwargs: Additional parameters
            
        Returns:
            Correlation coefficient ρ ∈ [0, 1]
        """
        pass


# ============================================================================
# BACKEND INTERFACES
# ============================================================================

class ChannelBackend(ABC):
    """
    Multi-backend support (MATLAB, Python, Sionna).
    
    Abstracts channel generation across different simulation engines.
    """
    
    @abstractmethod
    def generate_channels(
        self,
        N: int,
        K: int,
        num_samples: int,
        seed: Optional[int] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generate channel realizations.
        
        Args:
            N: Number of RIS elements
            K: Number of subcarriers (or 1 for narrowband)
            num_samples: Number of channel realizations
            seed: Random seed for reproducibility
            **kwargs: Backend-specific parameters
            
        Returns:
            Tuple of (h, g, metadata) where:
            - h: (N, num_samples) or (N, K, num_samples)
            - g: (N, num_samples) or (N, K, num_samples)
            - metadata: Dict with generation info
        """
        pass
    
    @abstractmethod
    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get backend information.
        
        Returns:
            Dictionary with backend name, version, capabilities, etc.
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if backend is available and functional.
        
        Returns:
            True if backend can be used
        """
        pass


# ============================================================================
# PROBING INTERFACES
# ============================================================================

class ProbingStrategy(ABC):
    """
    Intelligent probe selection strategy.
    
    Determines which probes to use for channel estimation given
    limited feedback budget.
    """
    
    @abstractmethod
    def select_probes(
        self,
        K: int,
        M: int,
        feedback: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Select M probes from K total probes.
        
        Args:
            K: Total number of available probes
            M: Number of probes to select
            feedback: Optional feedback from previous iterations
            **kwargs: Strategy-specific parameters
            
        Returns:
            Array of selected probe indices (M,)
        """
        pass
    
    @abstractmethod
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy information.
        
        Returns:
            Dictionary with strategy name, parameters, etc.
        """
        pass


# ============================================================================
# HARDWARE INTERFACES
# ============================================================================

class RISDriver(ABC):
    """
    Hardware deployment interface (ESP32, Jetson, FPGA).
    
    Abstracts communication with physical RIS hardware.
    """
    
    @abstractmethod
    def configure_phases(
        self,
        phases: np.ndarray,
        **kwargs
    ) -> bool:
        """
        Configure RIS phase shifts.
        
        Args:
            phases: Phase configuration (N,)
            **kwargs: Hardware-specific parameters
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def get_status(self) -> HardwareStatus:
        """
        Get current hardware status.
        
        Returns:
            HardwareStatus object
        """
        pass
    
    @abstractmethod
    def measure_channel(self, **kwargs) -> ChannelMeasurement:
        """
        Measure current channel state.
        
        Args:
            **kwargs: Measurement parameters
            
        Returns:
            ChannelMeasurement object
        """
        pass
    
    @abstractmethod
    def reset(self) -> bool:
        """
        Reset hardware to default state.
        
        Returns:
            True if successful
        """
        pass


# ============================================================================
# OPTIMIZATION INTERFACES
# ============================================================================

class Optimizer(ABC):
    """
    Hyperparameter/system optimization interface.
    
    Abstracts different optimization strategies (Bayesian, grid search, etc.)
    for hyperparameter tuning and system optimization.
    """
    
    @abstractmethod
    def optimize(
        self,
        objective_fn: callable,
        search_space: Dict[str, Any],
        n_trials: int,
        **kwargs
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run optimization.
        
        Args:
            objective_fn: Function to minimize/maximize
            search_space: Parameter search space specification
            n_trials: Number of optimization trials
            **kwargs: Optimizer-specific parameters
            
        Returns:
            Tuple of (best_params, best_value)
        """
        pass
    
    @abstractmethod
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """
        Get history of optimization trials.
        
        Returns:
            List of trial results
        """
        pass
    
    @abstractmethod
    def save_state(self, filepath: str) -> bool:
        """
        Save optimization state for resumption.
        
        Args:
            filepath: Path to save state
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def load_state(self, filepath: str) -> bool:
        """
        Load optimization state.
        
        Args:
            filepath: Path to load state from
            
        Returns:
            True if successful
        """
        pass


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Data structures
    'HardwareStatus',
    'ChannelMeasurement',
    
    # Physics interfaces
    'PhysicsModel',
    'UnitCellModel',
    'CouplingModel',
    'WavefrontModel',
    'ChannelAgingModel',
    
    # Backend interfaces
    'ChannelBackend',
    
    # Probing interfaces
    'ProbingStrategy',
    
    # Hardware interfaces
    'RISDriver',
    
    # Optimization interfaces
    'Optimizer',
]
