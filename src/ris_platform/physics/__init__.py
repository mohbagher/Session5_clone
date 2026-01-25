"""
Physics Module
==============
Composable physics models and components.
"""

from src.ris_platform.physics.models import IdealPhysicsModel, RealisticPhysicsModel, PhysicsEngine
from src.ris_platform.physics.components import *

__all__ = [
    'IdealPhysicsModel',
    'RealisticPhysicsModel',
    'PhysicsEngine',  # <--- CRITICAL ADDITION
    # Components
    'IdealUnitCell',
    'VaractorUnitCell',
    'NoCoupling',
    'GeometricCoupling',
    'PlanarWavefront',
    'SphericalWavefront',
    'JakesAging',
]