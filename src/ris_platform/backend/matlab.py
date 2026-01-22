"""
MATLAB Backend
==============
Wrapper for existing MATLAB Engine session manager.

CRITICAL: This wraps the battle-tested session_manager, does NOT rewrite it.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from src.ris_platform.core.interfaces import ChannelBackend

logger = logging.getLogger(__name__)


class MATLABBackend(ChannelBackend):
    """
    MATLAB channel generation backend.
    
    This wraps the existing physics.matlab_backend.session_manager
    to provide a standardized interface while preserving all
    existing functionality.
    
    Features:
    - Reuses persistent MATLAB session
    - Supports multiple scenarios (rayleigh_basic, cdl_ris, etc.)
    - Adds metadata tracking for reproducibility
    """
    
    def __init__(
        self,
        scenario: str = 'rayleigh_basic',
        auto_fallback: bool = True
    ):
        """
        Initialize MATLAB backend.
        
        Args:
            scenario: MATLAB scenario name
            auto_fallback: Fallback to Python if MATLAB unavailable
        """
        self.scenario = scenario
        self.auto_fallback = auto_fallback
        self._matlab_available = None
        self._session_manager = None
    
    def generate_channels(
        self,
        N: int,
        K: int,
        num_samples: int,
        seed: Optional[int] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generate channels using MATLAB backend.
        
        Args:
            N: Number of RIS elements
            K: Number of subcarriers (or snapshots)
            num_samples: Number of channel realizations
            seed: Random seed
            **kwargs: Additional MATLAB-specific parameters
            
        Returns:
            Tuple of (h, g, metadata) where:
            - h: (N, num_samples) complex channel
            - g: (N, num_samples) complex channel
            - metadata: Generation information
        """
        # Try to import MATLAB source
        try:
            from physics.matlab_backend.matlab_source import MATLABEngineSource
            
            logger.info(f"Generating {num_samples} channels with MATLAB ({self.scenario})")
            
            # Create source
            source = MATLABEngineSource(scenario=self.scenario)
            
            # Generate channels
            h, g, matlab_metadata = source.generate_channels(
                N=N,
                K=K,
                num_samples=num_samples,
                seed=seed if seed is not None else 42
            )
            
            # Enhance metadata
            metadata = {
                'backend': 'matlab',
                'scenario': self.scenario,
                'N': N,
                'K': K,
                'num_samples': num_samples,
                'seed': seed,
                **matlab_metadata
            }
            
            logger.info("MATLAB channel generation successful")
            return h, g, metadata
            
        except ImportError as e:
            logger.warning(f"MATLAB backend not available: {e}")
            if self.auto_fallback:
                logger.info("Falling back to Python synthetic backend")
                from src.ris_platform.backend.python_synthetic import PythonSyntheticBackend
                fallback = PythonSyntheticBackend()
                return fallback.generate_channels(N, K, num_samples, seed, **kwargs)
            else:
                raise RuntimeError(
                    "MATLAB backend not available. Install MATLAB Engine for Python "
                    "or set auto_fallback=True"
                )
        except Exception as e:
            logger.error(f"MATLAB generation failed: {e}")
            if self.auto_fallback:
                logger.info("Falling back to Python synthetic backend")
                from src.ris_platform.backend.python_synthetic import PythonSyntheticBackend
                fallback = PythonSyntheticBackend()
                return fallback.generate_channels(N, K, num_samples, seed, **kwargs)
            else:
                raise
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get MATLAB backend information."""
        info = {
            'name': 'MATLAB Backend',
            'scenario': self.scenario,
            'auto_fallback': self.auto_fallback,
            'available': self.is_available(),
        }
        
        # Try to get session info
        try:
            from physics.matlab_backend.session_manager import get_session_manager
            manager = get_session_manager()
            if manager.is_session_active():
                session_info = manager.get_session_info()
                info['session'] = {
                    'status': session_info.status,
                    'matlab_version': session_info.matlab_version,
                    'toolboxes': session_info.available_toolboxes
                }
        except Exception:
            pass
        
        return info
    
    def is_available(self) -> bool:
        """
        Check if MATLAB backend is available.
        
        Returns:
            True if MATLAB Engine can be imported
        """
        if self._matlab_available is not None:
            return self._matlab_available
        
        try:
            import matlab.engine
            from physics.matlab_backend.session_manager import get_session_manager
            self._matlab_available = True
        except ImportError:
            self._matlab_available = False
        
        return self._matlab_available


__all__ = ['MATLABBackend']
