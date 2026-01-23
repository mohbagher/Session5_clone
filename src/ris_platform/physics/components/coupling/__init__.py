"""
Coupling Models
===============
Mutual electromagnetic coupling between RIS elements.
"""

from src.ris_platform.physics.components.coupling.none import NoCoupling
from src.ris_platform.physics.components.coupling.geometric import GeometricCoupling

__all__ = ['NoCoupling', 'GeometricCoupling']
