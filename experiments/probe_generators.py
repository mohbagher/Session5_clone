"""
Probe generation for different probe types in RIS probe-based ML system.

Supports:
- Continuous probes: Random phases in [0, 2π)
- Binary probes: Phases {0, π}
- 2-bit probes: Phases {0, π/2, π, 3π/2}
- Hadamard probes: Structured binary using Hadamard matrix
"""

import numpy as np
from scipy.linalg import hadamard
from scipy.stats import qmc
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProbeBank:
    """Fixed probe bank containing K phase configurations."""
    phases: np.ndarray          # Shape: (K, N), phase values in [0, 2π)
    K: int                      # Number of probes
    N: int                      # Number of RIS elements
    probe_type: str = "continuous"  # Type of probe: continuous, binary, 2bit, hadamard, sobol, halton
    
    def get_reflection_coefficients(self) -> np.ndarray:
        """Return complex reflection coefficients exp(j*θ)."""
        return np.exp(1j * self.phases)


def generate_probe_bank_continuous(N: int, K: int, seed: Optional[int] = None) -> ProbeBank:
    """
    Generate probe bank with continuous random phases in [0, 2π).
    
    Args:
        N: Number of RIS elements
        K: Number of probes
        seed: Random seed for reproducibility
        
    Returns:
        ProbeBank with continuous phases
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # Generate random phases uniformly in [0, 2π)
    phases = rng.uniform(0, 2 * np.pi, size=(K, N))
    
    return ProbeBank(phases=phases, K=K, N=N, probe_type="continuous")


def generate_probe_bank_binary(N: int, K: int, seed: Optional[int] = None) -> ProbeBank:
    """
    Generate probe bank with binary phases {0, π}.
    
    Args:
        N: Number of RIS elements
        K: Number of probes
        seed: Random seed for reproducibility
        
    Returns:
        ProbeBank with binary phases
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # Generate random binary values (0 or 1)
    binary_values = rng.randint(0, 2, size=(K, N))
    
    # Map to phases {0, π}
    phases = binary_values * np.pi
    
    return ProbeBank(phases=phases, K=K, N=N, probe_type="binary")


def generate_probe_bank_2bit(N: int, K: int, seed: Optional[int] = None) -> ProbeBank:
    """
    Generate probe bank with 2-bit phases {0, π/2, π, 3π/2}.
    
    Args:
        N: Number of RIS elements
        K: Number of probes
        seed: Random seed for reproducibility
        
    Returns:
        ProbeBank with 2-bit phases
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # Generate random 2-bit values (0, 1, 2, 3)
    two_bit_values = rng.randint(0, 4, size=(K, N))
    
    # Map to phases {0, π/2, π, 3π/2}
    phases = two_bit_values * (np.pi / 2)
    
    return ProbeBank(phases=phases, K=K, N=N, probe_type="2bit")


def generate_probe_bank_hadamard(N: int, K: int) -> ProbeBank:
    """
    Generate probe bank using Hadamard-based structured binary patterns.
    
    Uses scipy.linalg.hadamard to create structured binary probes.
    The Hadamard matrix provides maximally orthogonal patterns.
    
    Args:
        N: Number of RIS elements (will be rounded up to nearest power of 2)
        K: Number of probes (limited by Hadamard matrix size)
        
    Returns:
        ProbeBank with Hadamard-based binary phases
    """
    # Find smallest power of 2 >= N
    hadamard_size = 2 ** int(np.ceil(np.log2(N)))
    
    # Generate Hadamard matrix
    H = hadamard(hadamard_size)
    
    # Normalize to {-1, 1} (already done by scipy)
    # Map to {0, π}: -1 → 0, +1 → π
    H_binary = (H + 1) / 2  # Map to {0, 1}
    H_phases = H_binary * np.pi  # Map to {0, π}
    
    # Take first N columns and first K rows
    phases = H_phases[:K, :N]
    
    # If K > hadamard_size, we need to repeat or generate more
    if K > hadamard_size:
        # Repeat patterns cyclically
        repeats = int(np.ceil(K / hadamard_size))
        phases_repeated = np.tile(H_phases[:, :N], (repeats, 1))
        phases = phases_repeated[:K, :]
    
    return ProbeBank(phases=phases, K=K, N=N, probe_type="hadamard")


def generate_probe_bank_sobol(N: int, K: int, seed: Optional[int] = None) -> ProbeBank:
    """
    Generate probe bank using Sobol low-discrepancy sequences.

    Args:
        N: Number of RIS elements
        K: Number of probes
        seed: Random seed for reproducibility

    Returns:
        ProbeBank with Sobol phases in [0, 2π)
    """
    sampler = qmc.Sobol(d=N, scramble=False, seed=seed)
    samples = sampler.random(n=K)
    phases = samples * (2 * np.pi)
    return ProbeBank(phases=phases, K=K, N=N, probe_type="sobol")


def generate_probe_bank_halton(N: int, K: int, seed: Optional[int] = None) -> ProbeBank:
    """
    Generate probe bank using Halton low-discrepancy sequences.

    Args:
        N: Number of RIS elements
        K: Number of probes
        seed: Random seed for reproducibility

    Returns:
        ProbeBank with Halton phases in [0, 2π)
    """
    sampler = qmc.Halton(d=N, scramble=False, seed=seed)
    samples = sampler.random(n=K)
    phases = samples * (2 * np.pi)
    return ProbeBank(phases=phases, K=K, N=N, probe_type="halton")


def get_probe_bank(probe_type: str, N: int, K: int, seed: Optional[int] = None) -> ProbeBank:
    """
    Factory function to generate probe bank of specified type.
    
    Args:
        probe_type: Type of probe ("continuous", "binary", "2bit", "hadamard", "sobol", "halton")
        N: Number of RIS elements
        K: Number of probes
        seed: Random seed (not used for Hadamard)
        
    Returns:
        ProbeBank of the specified type
    """
    probe_type = probe_type.lower()
    
    if probe_type == "continuous":
        return generate_probe_bank_continuous(N, K, seed)
    elif probe_type == "binary":
        return generate_probe_bank_binary(N, K, seed)
    elif probe_type == "2bit":
        return generate_probe_bank_2bit(N, K, seed)
    elif probe_type == "hadamard":
        return generate_probe_bank_hadamard(N, K)
    elif probe_type == "sobol":
        return generate_probe_bank_sobol(N, K, seed)
    elif probe_type == "halton":
        return generate_probe_bank_halton(N, K, seed)
    else:
        raise ValueError(f"Unknown probe type: {probe_type}. "
                        f"Must be one of: continuous, binary, 2bit, hadamard, sobol, halton")
