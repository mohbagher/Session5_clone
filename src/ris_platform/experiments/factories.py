"""
Backend Factories (Aligned with Tree)
=====================================
Creates the correct backend, physics engine, and probing strategy.
"""
import logging
from typing import Dict, Any

# 1. Backend Imports
try:
    from src.ris_platform.backend.matlab import MATLABBackend
except ImportError:
    MATLABBackend = None

# Python backend is in src/ris_platform/backend/python_synthetic.py based on your tree
from src.ris_platform.backend.python_synthetic import PythonSyntheticBackend as PythonBackend

# 2. Physics & Probing Imports (FIXED PATHS)
from src.ris_platform.physics.models import PhysicsEngine
from src.ris_platform.probing.structured import ProbingStrategy

logger = logging.getLogger(__name__)

def create_backend(config: Dict[str, Any]):
    """
    Factory to create the requested Simulation Backend.
    """
    backend_type = config.get('physics_backend', 'python')

    if backend_type == 'matlab':
        if MATLABBackend is None:
            raise ImportError("MATLAB Backend file not found or failed to import.")

        scenario = config.get('matlab_scenario', 'rayleigh_basic')
        logger.info(f"Creating MATLAB Backend with scenario: {scenario}")
        return MATLABBackend(scenario=scenario)
    else:
        # Default to Python
        logger.info("Creating Python Synthetic Backend")
        # Ensure we pass the scenario if the class expects it, or generic kwargs
        return PythonBackend(scenario=config.get('matlab_scenario', 'rayleigh_basic'))

def create_physics_model(config: Dict[str, Any]):
    """Factory for Physics Engine."""
    logger.info("Initializing Physics Engine...")
    return PhysicsEngine(config)

def create_probing_strategy(config: Dict[str, Any]):
    """Factory for Probing Strategy."""
    logger.info("Initializing Probing Strategy...")
    return ProbingStrategy(config)