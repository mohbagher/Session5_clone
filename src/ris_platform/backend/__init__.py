"""
Backend Abstractions
====================
Multi-backend support for channel generation.
"""

from src.ris_platform.backend.matlab import MATLABBackend
from src.ris_platform.backend.python_synthetic import PythonSyntheticBackend

__all__ = ['MATLABBackend', 'PythonSyntheticBackend']
