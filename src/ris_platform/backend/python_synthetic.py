"""
Python Synthetic Backend
========================
Fast numpy-based Rayleigh fading channel generation.

This is the baseline backend for quick simulations and testing.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from src.ris_platform.core.interfaces import ChannelBackend

logger = logging.getLogger(__name__)


class PythonSyntheticBackend(ChannelBackend):
    """
    Python synthetic channel generation backend.
    
    Generates Rayleigh fading channels using numpy:
    - CN(0, σ²) complex Gaussian distribution
    - Fast, analytically verified
    - No external dependencies
    
    This is the default backend for quick simulations.
    """
    
    def __init__(
        self,
        sigma_h_sq: float = 1.0,
        sigma_g_sq: float = 1.0
    ):
        """
        Initialize Python synthetic backend.
        
        Args:
            sigma_h_sq: Variance for h channel
            sigma_g_sq: Variance for g channel
        """
        self.sigma_h_sq = sigma_h_sq
        self.sigma_g_sq = sigma_g_sq
    
    def generate_channels(
        self,
        N: int,
        K: int,
        num_samples: int,
        seed: Optional[int] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generate synthetic Rayleigh fading channels.
        
        h ~ CN(0, σ_h²)
        g ~ CN(0, σ_g²)
        
        Args:
            N: Number of RIS elements
            K: Number of subcarriers (or 1 for narrowband)
            num_samples: Number of channel realizations
            seed: Random seed for reproducibility
            **kwargs: Additional parameters (sigma_h_sq, sigma_g_sq)
            
        Returns:
            Tuple of (h, g, metadata) where:
            - h: (N, num_samples) complex channel
            - g: (N, num_samples) complex channel
            - metadata: Generation information
        """
        # Override variances if provided in kwargs
        sigma_h_sq = kwargs.get('sigma_h_sq', self.sigma_h_sq)
        sigma_g_sq = kwargs.get('sigma_g_sq', self.sigma_g_sq)
        
        # Create RNG
        rng = np.random.RandomState(seed)
        
        logger.info(f"Generating {num_samples} synthetic channels (N={N})")
        
        # Generate complex Gaussian channels
        # CN(0, σ²) = (N(0, σ²/2) + j*N(0, σ²/2))
        h_real = rng.randn(N, num_samples) * np.sqrt(sigma_h_sq / 2)
        h_imag = rng.randn(N, num_samples) * np.sqrt(sigma_h_sq / 2)
        h = h_real + 1j * h_imag
        
        g_real = rng.randn(N, num_samples) * np.sqrt(sigma_g_sq / 2)
        g_imag = rng.randn(N, num_samples) * np.sqrt(sigma_g_sq / 2)
        g = g_real + 1j * g_imag
        
        # Metadata
        metadata = {
            'backend': 'python_synthetic',
            'distribution': 'rayleigh',
            'N': N,
            'K': K,
            'num_samples': num_samples,
            'sigma_h_sq': sigma_h_sq,
            'sigma_g_sq': sigma_g_sq,
            'seed': seed,
        }
        
        logger.info("Python synthetic channel generation successful")
        return h, g, metadata
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get Python backend information."""
        return {
            'name': 'Python Synthetic Backend',
            'description': 'Numpy-based Rayleigh fading',
            'sigma_h_sq': self.sigma_h_sq,
            'sigma_g_sq': self.sigma_g_sq,
            'available': True,
            'fast': True,
            'verified': True,
        }
    
    def is_available(self) -> bool:
        """
        Check if Python backend is available.
        
        Returns:
            Always True (numpy is required)
        """
        return True


__all__ = ['PythonSyntheticBackend']
