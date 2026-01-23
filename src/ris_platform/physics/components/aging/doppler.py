"""
Jakes' Doppler Channel Aging Model
==================================
Temporal correlation based on Jakes' Doppler model.

Reference: Jakes, Microwave Mobile Communications, 1974
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy.special import jv  # Bessel function
from src.ris_platform.core.interfaces import ChannelAgingModel


class JakesAging(ChannelAgingModel):
    """
    Jakes' Doppler model for channel aging.
    
    Model:
        ρ(τ) = J₀(2π * f_d * τ)
        h(t+Δt) = ρ * h(t) + √(1-ρ²) * innovation
    
    Where:
    - J₀: Bessel function of the first kind, order 0
    - f_d: Doppler frequency (Hz)
    - τ: Time delay
    - AR(1) temporal correlation
    
    Typical Doppler frequencies:
    - Pedestrian (3 km/h @ 2.4 GHz): ~6.7 Hz
    - Vehicular (30 km/h @ 2.4 GHz): ~67 Hz
    - High-speed (120 km/h @ 2.4 GHz): ~267 Hz
    """
    
    def __init__(
        self,
        doppler_hz: float = 10.0,
        velocity: Optional[float] = None,
        carrier_freq: Optional[float] = None
    ):
        """
        Initialize Jakes aging model.
        
        Args:
            doppler_hz: Doppler frequency in Hz
            velocity: Optional velocity in m/s (overrides doppler_hz if provided)
            carrier_freq: Optional carrier frequency in Hz (for velocity calculation)
        """
        if velocity is not None and carrier_freq is not None:
            # Calculate Doppler from velocity
            # f_d = v * f_c / c
            c = 3e8  # Speed of light
            self.doppler_hz = velocity * carrier_freq / c
        else:
            self.doppler_hz = doppler_hz
    
    def age_channel(
        self,
        h_current: np.ndarray,
        g_current: np.ndarray,
        time_delta: float,
        rng: Optional[np.random.RandomState] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evolve channels forward in time using AR(1) model.
        
        h(t+Δt) = ρ * h(t) + √(1-ρ²) * w_h
        g(t+Δt) = ρ * g(t) + √(1-ρ²) * w_g
        
        Args:
            h_current: Current h channel (N,) or (N, K)
            g_current: Current g channel (N,) or (N, K)
            time_delta: Time step in seconds
            rng: Random number generator
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (h_aged, g_aged) with same shape as inputs
        """
        if rng is None:
            rng = np.random.RandomState()
        
        # Compute correlation coefficient
        rho = self.get_correlation_coefficient(time_delta)
        
        # Innovation variance
        innovation_var = 1.0 - rho**2
        
        # Handle different shapes
        if h_current.ndim == 1:
            # Single snapshot (N,)
            h_innovation = (rng.randn(*h_current.shape) + 
                          1j * rng.randn(*h_current.shape)) / np.sqrt(2)
            g_innovation = (rng.randn(*g_current.shape) + 
                          1j * rng.randn(*g_current.shape)) / np.sqrt(2)
        else:
            # Multiple snapshots (N, K)
            h_innovation = (rng.randn(*h_current.shape) + 
                          1j * rng.randn(*h_current.shape)) / np.sqrt(2)
            g_innovation = (rng.randn(*g_current.shape) + 
                          1j * rng.randn(*g_current.shape)) / np.sqrt(2)
        
        # AR(1) update
        h_aged = rho * h_current + np.sqrt(innovation_var) * h_innovation
        g_aged = rho * g_current + np.sqrt(innovation_var) * g_innovation
        
        return h_aged, g_aged
    
    def get_correlation_coefficient(
        self,
        time_delta: float,
        **kwargs
    ) -> float:
        """
        Get temporal correlation coefficient using Jakes' model.
        
        ρ(τ) = J₀(2π * f_d * τ)
        
        Args:
            time_delta: Time step in seconds
            **kwargs: Additional parameters
            
        Returns:
            Correlation coefficient ρ ∈ [0, 1]
        """
        # Jakes' autocorrelation function
        rho = jv(0, 2.0 * np.pi * self.doppler_hz * time_delta)
        
        # Clip to [0, 1] for numerical stability
        rho = np.clip(rho, 0.0, 1.0)
        
        return float(rho)
    
    def get_coherence_time(self) -> float:
        """
        Get channel coherence time.
        
        Coherence time is approximately 1 / f_d.
        
        Returns:
            Coherence time in seconds
        """
        if self.doppler_hz > 0:
            return 1.0 / self.doppler_hz
        else:
            return np.inf
    
    def compute_doppler_spectrum(
        self,
        frequencies: np.ndarray
    ) -> np.ndarray:
        """
        Compute Jakes' Doppler spectrum.
        
        S(f) = 1 / (π * f_d * √(1 - (f/f_d)²)) for |f| < f_d
        
        Args:
            frequencies: Frequency points (Hz)
            
        Returns:
            Spectrum values
        """
        f_d = self.doppler_hz
        
        # Normalized frequency
        f_norm = frequencies / f_d
        
        # Compute spectrum
        spectrum = np.zeros_like(frequencies)
        valid = np.abs(f_norm) < 1.0
        
        if np.any(valid):
            spectrum[valid] = 1.0 / (
                np.pi * f_d * np.sqrt(1.0 - f_norm[valid]**2)
            )
        
        return spectrum


__all__ = ['JakesAging']
