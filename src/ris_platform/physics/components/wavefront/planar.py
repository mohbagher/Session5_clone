import numpy as np
from typing import Dict, Any, Optional
from src.ris_platform.core.interfaces import WavefrontModel

class PlanarWavefront(WavefrontModel):
    def __init__(self):
        pass

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
        Handles broadcasting for Probe Banks (Single Channel vs Multiple RIS Configs).
        """
        # Element-wise product
        # Broadcasts if h is (N,) and gamma is (K, N) -> reflected is (K, N)
        reflected = gamma * h
        
        # Case 1: Single Channel Snapshot (h is 1D)
        if h.ndim == 1:
            # Sub-case: Multiple RIS Configurations (Probe Bank)
            if reflected.ndim == 2:
                # reflected is (K, N), g is (N,)
                # We want a result of shape (K,)
                return np.sum(np.conj(g) * reflected, axis=1)
            else:
                # Sub-case: Single RIS Configuration
                return np.vdot(g, reflected)

        # Case 2: Multiple Channel Snapshots (h is 2D: N x Samples)
        else:
            return np.sum(np.conj(g) * reflected, axis=0)

    def compute_rayleigh_distance(self, geometry: Dict[str, Any], **kwargs) -> float:
        return np.inf