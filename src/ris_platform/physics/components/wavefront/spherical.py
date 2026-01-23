"""
Spherical Wavefront Model
=========================
Near-field 6G propagation with distance-dependent effects.

Reference: Björnson et al., arXiv:2110.06661
"""

import numpy as np
from typing import Dict, Any, Optional
from src.ris_platform.core.interfaces import WavefrontModel


class SphericalWavefront(WavefrontModel):
    """
    Spherical wavefront model for near-field propagation.
    
    Model:
        y = Σ_n g_n * Γ_n * h_n * exp(-jk*r_n) / r_n
    
    Where:
    - r_n: Distance from element n to user
    - k: Wave number (2π/λ)
    - Includes distance-dependent phase and path loss
    
    This model is critical for 6G near-field scenarios where
    distances may be comparable to the Rayleigh distance.
    """
    
    def __init__(
        self,
        wavelength: float = 0.125,  # 2.4 GHz default
        include_path_loss: bool = True,
        reference_distance: float = 1.0
    ):
        """
        Initialize spherical wavefront model.
        
        Args:
            wavelength: Operating wavelength in meters
            include_path_loss: Include 1/r path loss
            reference_distance: Reference distance for normalization (meters)
        """
        self.wavelength = wavelength
        self.k = 2.0 * np.pi / wavelength  # Wave number
        self.include_path_loss = include_path_loss
        self.d_ref = reference_distance
    
    def compute_signal(
        self,
        h: np.ndarray,
        g: np.ndarray,
        gamma: np.ndarray,
        geometry: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Compute received signal with spherical wavefront.
        
        y = Σ_n g_n * Γ_n * h_n * exp(-jk*r_n) / r_n
        
        Args:
            h: Base station to RIS channel (N,) or (N, K)
            g: RIS to user channel (N,) or (N, K)
            gamma: RIS reflection coefficients (N,) or (N, K)
            geometry: Dict with 'ris_positions', 'bs_position', 'user_position'
            **kwargs: Additional parameters
            
        Returns:
            Received signal(s) with near-field effects
        """
        if geometry is None:
            # Fallback to planar wavefront if no geometry
            reflected = gamma * h
            if h.ndim == 1:
                return np.vdot(g, reflected)
            else:
                return np.sum(np.conj(g) * reflected, axis=0)
        
        # Extract positions
        ris_pos = geometry.get('ris_positions')
        bs_pos = geometry.get('bs_position')
        user_pos = geometry.get('user_position')
        
        if ris_pos is None or bs_pos is None or user_pos is None:
            # Missing geometry, fallback to planar
            reflected = gamma * h
            if h.ndim == 1:
                return np.vdot(g, reflected)
            else:
                return np.sum(np.conj(g) * reflected, axis=0)
        
        # Compute distances
        r_bs = self._compute_distances(ris_pos, bs_pos)
        r_user = self._compute_distances(ris_pos, user_pos)
        
        # Distance-dependent phase
        phase_bs = np.exp(-1j * self.k * r_bs)
        phase_user = np.exp(-1j * self.k * r_user)
        
        # Path loss (if enabled)
        if self.include_path_loss:
            path_loss_bs = self.d_ref / r_bs
            path_loss_user = self.d_ref / r_user
        else:
            path_loss_bs = np.ones_like(r_bs)
            path_loss_user = np.ones_like(r_user)
        
        # Combined effect
        h_effective = h * phase_bs * path_loss_bs
        g_effective = g * phase_user * path_loss_user
        
        # Compute signal
        reflected = gamma * h_effective
        
        if h.ndim == 1:
            return np.vdot(g_effective, reflected)
        else:
            return np.sum(np.conj(g_effective) * reflected, axis=0)
    
    def compute_rayleigh_distance(
        self,
        geometry: Dict[str, Any],
        **kwargs
    ) -> float:
        """
        Compute Rayleigh distance for near-field determination.
        
        Rayleigh distance: D_R = 2 * D^2 / λ
        
        Where D is the largest dimension of the RIS.
        
        Args:
            geometry: Dict with 'ris_positions'
            **kwargs: Additional parameters
            
        Returns:
            Rayleigh distance in meters
        """
        ris_pos = geometry.get('ris_positions')
        
        if ris_pos is None:
            # Default assumption
            return 10.0
        
        # Find largest dimension
        if ris_pos.shape[1] == 2:
            # 2D positions
            x_range = np.max(ris_pos[:, 0]) - np.min(ris_pos[:, 0])
            y_range = np.max(ris_pos[:, 1]) - np.min(ris_pos[:, 1])
            D = max(x_range, y_range)
        else:
            # 3D positions
            ranges = np.ptp(ris_pos, axis=0)  # Peak-to-peak
            D = np.max(ranges)
        
        # Rayleigh distance
        D_R = 2.0 * D**2 / self.wavelength
        
        return D_R
    
    def _compute_distances(
        self,
        positions: np.ndarray,
        point: np.ndarray
    ) -> np.ndarray:
        """
        Compute distances from multiple positions to a point.
        
        Args:
            positions: Array of positions (N, d)
            point: Single point (d,)
            
        Returns:
            Distances (N,)
        """
        diffs = positions - point
        distances = np.linalg.norm(diffs, axis=1)
        return distances
    
    def is_near_field(
        self,
        geometry: Dict[str, Any],
        **kwargs
    ) -> bool:
        """
        Check if scenario is in near-field regime.
        
        Near-field if distance < Rayleigh distance.
        
        Args:
            geometry: Spatial geometry
            **kwargs: Additional parameters
            
        Returns:
            True if near-field
        """
        D_R = self.compute_rayleigh_distance(geometry)
        
        # Check user distance
        ris_pos = geometry.get('ris_positions')
        user_pos = geometry.get('user_position')
        
        if ris_pos is None or user_pos is None:
            return False
        
        # Distance to center of RIS
        ris_center = np.mean(ris_pos, axis=0)
        d_user = np.linalg.norm(user_pos - ris_center)
        
        return d_user < D_R


__all__ = ['SphericalWavefront']
