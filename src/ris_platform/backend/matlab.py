"""
MATLAB Backend Adapter (Robust)
===============================
Handles connection failures gracefully by falling back to internal Python generation.
"""
import logging
import numpy as np
from typing import Dict, Any, Tuple
from src.ris_platform.core.interfaces import ChannelBackend

# Legacy import
try:
    from physics.matlab_backend.matlab_source import MATLABEngineSource
except ImportError:
    MATLABEngineSource = None

logger = logging.getLogger(__name__)

class MATLABBackend(ChannelBackend):
    def __init__(self, scenario='rayleigh_basic', auto_fallback=True):
        self.scenario = scenario
        self.auto_fallback = auto_fallback
        self.source = None

        if MATLABEngineSource:
            try:
                self.source = MATLABEngineSource(scenario=scenario)
            except Exception as e:
                logger.warning(f"MATLAB Init Failed: {e}")

    def is_available(self) -> bool:
        return self.source is not None

    def get_backend_info(self) -> Dict:
        return {'name': 'matlab_engine', 'available': self.is_available()}

    def generate_channels(self, N: int, K: int, num_samples: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
        # 1. Try MATLAB
        if self.source:
            try:
                h, g, meta = self.source.generate_channels(N=N, K=K, num_samples=num_samples, seed=seed)
                # Check explicitly if we got data back
                if h is not None and g is not None:
                    meta['backend'] = 'matlab'
                    return h, g, meta
            except Exception as e:
                logger.error(f"MATLAB Generation Failed: {e}")
                # Don't return None! Proceed to fallback.

        # 2. Fallback (Internal Logic)
        if self.auto_fallback:
            logger.warning("⚠️ MATLAB Failed. Falling back to Python Synthetic.")
            rng = np.random.RandomState(seed)
            # Generate simple Rayleigh fading
            h = (rng.randn(N, num_samples) + 1j * rng.randn(N, num_samples)) / np.sqrt(2)
            g = (rng.randn(N, num_samples) + 1j * rng.randn(N, num_samples)) / np.sqrt(2)

            return h, g, {
                'backend': 'python_fallback',
                'reason': 'matlab_unavailable',
                'N': N,
                'seed': seed
            }

        # If we get here, everything failed
        raise RuntimeError("MATLAB backend failed and fallback is disabled.")