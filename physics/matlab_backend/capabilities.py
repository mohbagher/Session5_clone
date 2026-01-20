"""
MATLAB Capability Interfaces
=============================
Python-facing interfaces for MATLAB functionality.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging

# Conditional import for MATLAB Engine
try:
    import matlab.engine
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False
    matlab = None  # Placeholder

from physics.matlab_backend.session_manager import get_session_manager

logger = logging.getLogger(__name__)


@dataclass
class CapabilityMetadata:
    """Metadata for MATLAB capability execution."""
    backend_name: str = "matlab"
    capability_name: str = ""
    toolbox_or_method: str = ""
    run_id: str = ""
    seed_info: Optional[int] = None
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'backend_name': self.backend_name,
            'capability_name': self.capability_name,
            'toolbox_or_method': self.toolbox_or_method,
            'run_id': self.run_id,
            'seed_info': self.seed_info,
            'execution_time_ms': self.execution_time_ms
        }


class MATLABChannelGenerator:
    """
    MATLAB-based channel generation capability.

    High-value Phase 2 capability: Validated Rayleigh fading.
    """

    def __init__(self):
        if not MATLAB_AVAILABLE:
            raise ImportError(
                "MATLAB Engine for Python not available. "
                "Install with: cd <MATLAB_ROOT>/extern/engines/python && python setup.py install"
            )
        self.session_manager = get_session_manager()

    def generate_channels(
        self,
        N: int,
        num_realizations: int = 1,
        sigma_h_sq: float = 1.0,
        sigma_g_sq: float = 1.0,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, CapabilityMetadata]:
        """
        Generate channel realizations using MATLAB.

        Args:
            N: Number of RIS elements
            num_realizations: Number of channel realizations to generate
            sigma_h_sq: BS-RIS channel variance
            sigma_g_sq: RIS-UE channel variance
            seed: Random seed for reproducibility

        Returns:
            h: BS-RIS channels (N, num_realizations)
            g: RIS-UE channels (N, num_realizations)
            metadata: Execution metadata
        """
        import time
        start_time = time.time()

        # Get MATLAB engine
        engine = self.session_manager.get_engine()
        if engine is None:
            raise RuntimeError(
                "MATLAB Engine not available. "
                "Ensure MATLAB is installed and matlab.engine is configured."
            )

        try:
            # Set random seed if provided
            if seed is not None:
                engine.rng(seed, 'twister', nargout=0)

            # Call MATLAB function
            h_matlab, g_matlab, meta_matlab = engine.generate_rayleigh_channel(
                float(N),
                sigma_h_sq,
                sigma_g_sq,
                float(seed) if seed is not None else [],
                nargout=3
            )

            # Convert MATLAB arrays to numpy
            h = np.array(h_matlab)
            g = np.array(g_matlab)

            # Execution time
            execution_time = (time.time() - start_time) * 1000

            # Create metadata
            run_id = f"matlab_channel_{int(time.time() * 1000)}"
            metadata = CapabilityMetadata(
                backend_name="matlab",
                capability_name="generate_channels",
                toolbox_or_method="rayleigh_fading",
                run_id=run_id,
                seed_info=seed,
                execution_time_ms=execution_time
            )

            logger.info(f"MATLAB generated {num_realizations} channel realizations in {execution_time:.2f}ms")

            return h, g, metadata

        except Exception as e:
            logger.error(f"MATLAB channel generation failed: {e}")
            raise


class MATLABMetricsCalculator:
    """
    MATLAB-based metrics calculation (placeholder).

    Future capability: Verified BER, capacity calculations.
    """

    def __init__(self):
        if not MATLAB_AVAILABLE:
            raise ImportError("MATLAB Engine for Python not available")
        self.session_manager = get_session_manager()

    def compute_metrics(self, *args, **kwargs):
        """
        Placeholder for future metrics capability.

        Not implemented in Phase 2 - design interface only.
        """
        raise NotImplementedError(
            "MATLAB metrics capability not yet implemented. "
            "Planned for Phase 2.2 expansion."
        )


# Capability registry
MATLAB_CAPABILITIES = {
    'channel_generation': MATLABChannelGenerator,
    'metrics_calculation': MATLABMetricsCalculator,
}


def get_capability(capability_name: str):
    """Get MATLAB capability instance."""
    if capability_name not in MATLAB_CAPABILITIES:
        raise ValueError(f"Unknown capability: {capability_name}")

    return MATLAB_CAPABILITIES[capability_name]()