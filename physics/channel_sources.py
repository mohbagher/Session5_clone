"""
Channel Source Abstraction Layer
=================================
Plugin-style architecture for different physics generators.

Supported sources:
- PythonSynthetic: Built-in Python channel generator (default)
- MATLABEngine: MATLAB-based channel models (future)
- MATLABVerified: Pre-verified MATLAB data loader (future)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ChannelSourceMetadata:
    """Metadata about the channel source used."""
    source_type: str
    source_version: str
    generation_params: Dict
    verification_status: str = "unverified"

    def to_dict(self) -> Dict:
        return {
            'source_type': self.source_type,
            'source_version': self.source_version,
            'generation_params': self.generation_params,
            'verification_status': self.verification_status
        }


class ChannelSource(ABC):
    """Abstract base class for channel generators."""

    @abstractmethod
    def generate_channel(self,
                        N: int,
                        **kwargs) -> Tuple[np.ndarray, np.ndarray, ChannelSourceMetadata]:
        """
        Generate one channel realization (h, g).

        Args:
            N: Number of RIS elements
            **kwargs: Source-specific parameters

        Returns:
            h: BS-RIS channel (complex, shape N)
            g: RIS-UE channel (complex, shape N)
            metadata: Generation metadata
        """
        pass

    @abstractmethod
    def get_source_info(self) -> Dict:
        """Return information about this channel source."""
        pass


class PythonSyntheticSource(ChannelSource):
    """
    Built-in Python synthetic channel generator.

    Current default - uses numpy for Rayleigh fading.
    """

    def __init__(self):
        self.version = "1.0.0"

    def generate_channel(self,
                        N: int,
                        sigma_h_sq: float = 1.0,
                        sigma_g_sq: float = 1.0,
                        rng: Optional[np.random.RandomState] = None,
                        **kwargs) -> Tuple[np.ndarray, np.ndarray, ChannelSourceMetadata]:
        """Generate Rayleigh fading channels."""
        if rng is None:
            rng = np.random.RandomState()

        # Standard Rayleigh fading
        h = np.sqrt(sigma_h_sq / 2) * (rng.randn(N) + 1j * rng.randn(N))
        g = np.sqrt(sigma_g_sq / 2) * (rng.randn(N) + 1j * rng.randn(N))

        metadata = ChannelSourceMetadata(
            source_type="PythonSynthetic",
            source_version=self.version,
            generation_params={
                'N': N,
                'sigma_h_sq': sigma_h_sq,
                'sigma_g_sq': sigma_g_sq,
                'distribution': 'Rayleigh'
            },
            verification_status="verified_against_theory"
        )

        return h, g, metadata

    def generate_channels(self,
                          N: int,
                          K: int = 1,
                          num_samples: int = 1,
                          sigma_h_sq: float = 1.0,
                          sigma_g_sq: float = 1.0,
                          seed: Optional[int] = None,
                          **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate multiple channel realizations.

        Args:
            N: Number of RIS elements
            K: Number of transmit antennas (unused for simple Rayleigh)
            num_samples: Number of channel realizations
            sigma_h_sq: BS-RIS channel variance
            sigma_g_sq: RIS-UE channel variance
            seed: Random seed
            **kwargs: Additional parameters (ignored)

        Returns:
            h_all: BS-RIS channels (N, num_samples)
            g_all: RIS-UE channels (N, num_samples)
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

        h_all = np.zeros((N, num_samples), dtype=complex)
        g_all = np.zeros((N, num_samples), dtype=complex)

        for i in range(num_samples):
            h, g, _ = self.generate_channel(N, sigma_h_sq, sigma_g_sq, rng)
            h_all[:, i] = h
            g_all[:, i] = g

        return h_all, g_all

    def get_source_info(self) -> Dict:
        return {
            'name': 'Python Synthetic Generator',
            'version': self.version,
            'description': 'Built-in numpy-based Rayleigh fading',
            'validation': 'Analytically verified',
            'dependencies': ['numpy']
        }


class MATLABEngineSource(ChannelSource):
    """
    MATLAB Engine-based channel generator (future implementation).

    Will call MATLAB scripts via MATLAB Engine API.
    """

    def __init__(self):
        self.version = "0.0.0"
        self.engine = None  # Placeholder for future MATLAB engine

    def generate_channel(self,
                        N: int,
                        **kwargs) -> Tuple[np.ndarray, np.ndarray, ChannelSourceMetadata]:
        raise NotImplementedError(
            "MATLAB Engine integration not yet implemented. "
            "Use PythonSynthetic source for now."
        )

    def get_source_info(self) -> Dict:
        return {
            'name': 'MATLAB Engine Generator',
            'version': self.version,
            'description': 'Live MATLAB channel generation via Engine API',
            'validation': 'Pending implementation',
            'dependencies': ['matlab.engine', 'MATLAB R2021b+'],
            'status': 'NOT_IMPLEMENTED'
        }


class MATLABVerifiedSource(ChannelSource):
    """
    Pre-verified MATLAB data loader (future implementation).

    Will load pre-generated channel realizations from MATLAB .mat files.
    """

    def __init__(self, data_path: Optional[str] = None):
        self.version = "0.0.0"
        self.data_path = data_path

    def generate_channel(self,
                        N: int,
                        **kwargs) -> Tuple[np.ndarray, np.ndarray, ChannelSourceMetadata]:
        raise NotImplementedError(
            "MATLAB verified data loading not yet implemented. "
            "Use PythonSynthetic source for now."
        )

    def get_source_info(self) -> Dict:
        return {
            'name': 'MATLAB Verified Data Loader',
            'version': self.version,
            'description': 'Load pre-verified MATLAB channel data',
            'validation': 'Pending implementation',
            'dependencies': ['scipy.io'],
            'status': 'NOT_IMPLEMENTED'
        }


# =============================================================================
# CHANNEL SOURCE REGISTRY & FACTORY
# =============================================================================

_CHANNEL_SOURCES = {
    'python_synthetic': PythonSyntheticSource,
    'matlab_engine': MATLABEngineSource,
    'matlab_verified': MATLABVerifiedSource,
}


def get_channel_source(source_type: str = 'python_synthetic', **init_kwargs) -> ChannelSource:
    """
    Factory function to create channel source.

    Args:
        source_type: One of 'python_synthetic', 'matlab_engine', 'matlab_verified'
        **init_kwargs: Initialization parameters for the source

    Returns:
        ChannelSource instance
    """
    source_type = source_type.lower()

    if source_type not in _CHANNEL_SOURCES:
        raise ValueError(
            f"Unknown channel source: {source_type}. "
            f"Available: {list(_CHANNEL_SOURCES.keys())}"
        )

    source_class = _CHANNEL_SOURCES[source_type]
    return source_class(**init_kwargs)


def list_available_sources() -> Dict[str, Dict]:
    """List all available channel sources and their status."""
    sources_info = {}
    for name, source_class in _CHANNEL_SOURCES.items():
        try:
            source = source_class()
            sources_info[name] = source.get_source_info()
        except:
            sources_info[name] = {
                'name': name,
                'status': 'INITIALIZATION_FAILED'
            }
    return sources_info
