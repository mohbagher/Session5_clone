"""
RIS Platform - Physics-First Architecture
=========================================
Composable physics engine for RIS simulation and optimization.

Phase 2: Refactored architecture with dependency injection.
"""

__version__ = "2.0.0"

from src.ris_platform.core.interfaces import (
    # Data structures
    HardwareStatus,
    ChannelMeasurement,
    
    # Physics interfaces
    PhysicsModel,
    UnitCellModel,
    CouplingModel,
    WavefrontModel,
    ChannelAgingModel,
    
    # Backend interfaces
    ChannelBackend,
    
    # Probing interfaces
    ProbingStrategy,
    
    # Hardware interfaces
    RISDriver,
    
    # Optimization interfaces
    Optimizer,
)

__all__ = [
    '__version__',
    # Data structures
    'HardwareStatus',
    'ChannelMeasurement',
    # Interfaces
    'PhysicsModel',
    'UnitCellModel',
    'CouplingModel',
    'WavefrontModel',
    'ChannelAgingModel',
    'ChannelBackend',
    'ProbingStrategy',
    'RISDriver',
    'Optimizer',
]
