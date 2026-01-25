"""
Structured Probing Strategies
=============================
Defines algorithms for selecting RIS configurations (probes).
"""
import numpy as np
from typing import Dict, Any

class ProbingStrategy:
    """Base class for all probing strategies."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def select_probes(self, K: int, M: int) -> np.ndarray:
        """Default behavior: Random selection."""
        if M > K: M = K
        # Deterministic random selection for reproducibility
        rng = np.random.RandomState(42)
        return rng.choice(K, size=M, replace=False)

# --- Concrete Implementations (Fixes the ImportError) ---

class RandomProbing(ProbingStrategy):
    """Pure random selection from codebook."""
    pass

class SobolProbing(ProbingStrategy):
    """
    Quasi-random Sobol sequence selection.
    (Placeholder: falls back to random for now, ready for expansion)
    """
    pass

class HadamardProbing(ProbingStrategy):
    """
    Hadamard-based deterministic selection.
    (Placeholder: falls back to random for now, ready for expansion)
    """
    pass