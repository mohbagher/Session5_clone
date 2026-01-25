"""
Python Backend Adapter (Phase 2 Compliant)
==========================================
Standard Python-based channel generation implementing the Phase 2 Interface.
"""
import numpy as np
import logging
from typing import Dict, Any, Tuple
from src.ris_platform.core.interfaces import ChannelBackend

logger = logging.getLogger(__name__)

class PythonSyntheticBackend(ChannelBackend):
    def __init__(self, scenario='rayleigh_basic', **kwargs):
        self.scenario = scenario

    # --- REQUIRED INTERFACE METHODS ---
    def is_available(self) -> bool:
        """Python is always available."""
        return True

    def get_backend_info(self) -> Dict[str, Any]:
        return {
            'name': 'python_synthetic',
            'scenario': self.scenario,
            'available': True,
            'library': 'numpy'
        }
    # ----------------------------------

    def generate_channels(self, N: int, K: int, num_samples: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate synthetic Rayleigh fading channels using NumPy.
        """
        # logger.info(f"Generating {num_samples} samples using Python (NumPy)...")

        rng = np.random.RandomState(seed)

        # Standard complex Gaussian (Rayleigh fading magnitude)
        # N elements x num_samples
        h = (rng.randn(N, num_samples) + 1j * rng.randn(N, num_samples)) / np.sqrt(2)
        g = (rng.randn(N, num_samples) + 1j * rng.randn(N, num_samples)) / np.sqrt(2)

        metadata = {
            'backend': 'python_synthetic',
            'scenario': self.scenario,
            'seed': seed
        }

        return h, g, metadata