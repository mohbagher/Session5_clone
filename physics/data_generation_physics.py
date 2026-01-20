"""
Data generation for RIS probe-based control with Phase 1 physics extensions.

Key Phase 1 changes:
- Uses pluggable channel sources
- Applies modular impairment pipeline
- Logs physics metadata for reproducibility
"""

import numpy as np
from typing import Tuple, Dict, Optional
import sys
import os

# Add physics module to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from physics.channel_sources import get_channel_source, ChannelSourceMetadata
from physics.realism_profiles import create_pipeline_from_profile, create_custom_pipeline
from data.probe_generators import ProbeBank


def generate_channel_realization_with_physics(
        N: int,
        channel_source_type: str = "python_synthetic",
        realism_profile: str = "ideal",
        use_custom_impairments: bool = False,
        custom_impairments_config: Optional[Dict] = None,
        sigma_h_sq: float = 1.0,
        sigma_g_sq: float = 1.0,
        rng: Optional[np.random.RandomState] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate channel realization using Phase 1 physics pipeline.

    Args:
        N: Number of RIS elements
        channel_source_type: Channel generator type
        realism_profile: Realism preset name
        use_custom_impairments: If True, use custom config
        custom_impairments_config: Custom impairment settings
        sigma_h_sq: BS-RIS channel variance
        sigma_g_sq: RIS-UE channel variance
        rng: Random number generator

    Returns:
        h: BS-RIS channel (with impairments applied)
        g: RIS-UE channel (with impairments applied)
        metadata: Complete physics metadata for logging
    """
    if rng is None:
        rng = np.random.RandomState()

    # 1. Generate clean channel from source
    source = get_channel_source(channel_source_type)
    h_clean, g_clean, source_metadata = source.generate_channel(
        N=N,
        sigma_h_sq=sigma_h_sq,
        sigma_g_sq=sigma_g_sq,
        rng=rng
    )

    # 2. Create impairment pipeline
    if use_custom_impairments and custom_impairments_config is not None:
        pipeline = create_custom_pipeline(custom_impairments_config)
    else:
        pipeline = create_pipeline_from_profile(realism_profile)

    # 3. Apply channel impairments
    h_impaired, g_impaired, impairment_metadata = pipeline.apply_channel_impairments(
        h_clean, g_clean, rng
    )

    # 4. Compile complete metadata
    metadata = {
        'channel_source': source_metadata.to_dict(),
        'realism_profile': realism_profile,
        'use_custom_impairments': use_custom_impairments,
        'impairments_applied': impairment_metadata,
        'impairment_configuration': pipeline.get_configuration_summary()
    }

    return h_impaired, g_impaired, metadata


def compute_probe_powers_with_physics(
        h: np.ndarray,
        g: np.ndarray,
        probe_bank: ProbeBank,
        realism_profile: str = "ideal",
        use_custom_impairments: bool = False,
        custom_impairments_config: Optional[Dict] = None,
        P_tx: float = 1.0,
        rng: Optional[np.random.RandomState] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Compute probe powers with hardware impairments.

    Args:
        h, g: Channel vectors
        probe_bank: Probe bank
        realism_profile: Realism preset
        use_custom_impairments: Use custom config
        custom_impairments_config: Custom settings
        P_tx: Transmit power
        rng: Random number generator

    Returns:
        powers: Received powers for all probes
        metadata: Hardware impairment metadata
    """
    if rng is None:
        rng = np.random.RandomState()

    # Create pipeline for hardware impairments
    if use_custom_impairments and custom_impairments_config is not None:
        pipeline = create_custom_pipeline(custom_impairments_config)
    else:
        pipeline = create_pipeline_from_profile(realism_profile)

    # Get probe phases
    probe_phases = probe_bank.phases  # Shape: (K, N)

    # Apply hardware impairments to each probe
    K = len(probe_phases)
    powers = np.zeros(K)
    hardware_metadata = []

    for k in range(K):
        # Get phases for this probe
        phases_k = probe_phases[k]

        # Apply phase quantization
        phases_impaired, phase_meta = pipeline.apply_hardware_impairments(phases_k, rng)

        # Create reflection coefficients
        reflection_coeffs = np.exp(1j * phases_impaired)

        # Apply amplitude impairments
        reflection_impaired, amp_meta = pipeline.apply_amplitude_impairments(reflection_coeffs, rng)

        # Compute effective channel
        c = h * g
        h_eff = np.dot(reflection_impaired, c)

        # Compute power
        powers[k] = P_tx * np.abs(h_eff) ** 2

        # Store metadata (only for first probe to save memory)
        if k == 0:
            hardware_metadata = {
                'phase_impairments': phase_meta,
                'amplitude_impairments': amp_meta
            }

    metadata = {
        'hardware_impairments_applied': hardware_metadata,
        'impairment_configuration': pipeline.get_configuration_summary()
    }

    return powers, metadata


def generate_limited_probing_dataset_with_physics(
        probe_bank: ProbeBank,
        n_samples: int,
        M: int,
        channel_source: str = "python_synthetic",
        realism_profile: str = "ideal",
        use_custom_impairments: bool = False,
        custom_impairments_config: Optional[Dict] = None,
        sigma_h_sq: float = 1.0,
        sigma_g_sq: float = 1.0,
        P_tx: float = 1.0,
        normalize: bool = True,
        seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generate dataset with Phase 1 physics pipeline.

    Returns same structure as original generate_limited_probing_dataset,
    but with added physics_metadata field.
    """
    rng = np.random.RandomState(seed)

    K = probe_bank.K
    N = probe_bank.N

    # Allocate arrays (same as before)
    masked_powers = np.zeros((n_samples, K), dtype=np.float32)
    masks = np.zeros((n_samples, K), dtype=np.float32)
    observed_indices = np.zeros((n_samples, M), dtype=np.int64)
    powers_full = np.zeros((n_samples, K), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int64)
    optimal_powers = np.zeros(n_samples, dtype=np.float32)

    # Track physics metadata for first sample (representative)
    first_sample_metadata = None

    for i in range(n_samples):
        # Generate channel with physics pipeline
        h, g, channel_metadata = generate_channel_realization_with_physics(
            N=N,
            channel_source_type=channel_source,
            realism_profile=realism_profile,
            use_custom_impairments=use_custom_impairments,
            custom_impairments_config=custom_impairments_config,
            sigma_h_sq=sigma_h_sq,
            sigma_g_sq=sigma_g_sq,
            rng=rng
        )

        # Compute probe powers with hardware impairments
        p_full, hardware_metadata = compute_probe_powers_with_physics(
            h, g, probe_bank,
            realism_profile=realism_profile,
            use_custom_impairments=use_custom_impairments,
            custom_impairments_config=custom_impairments_config,
            P_tx=P_tx,
            rng=rng
        )

        powers_full[i] = p_full
        labels[i] = np.argmax(p_full)

        # Theoretical optimal (without impairments)
        c = h * g
        h_eff_opt = np.sum(np.abs(c))
        optimal_powers[i] = P_tx * h_eff_opt ** 2

        # Select observation subset
        obs_idx = rng.choice(K, size=M, replace=False)
        obs_idx = np.sort(obs_idx)
        observed_indices[i] = obs_idx

        # Create masked input
        observed_powers = p_full[obs_idx]
        if normalize and len(observed_powers) > 0:
            mean_power = np.mean(observed_powers)
            if mean_power > 1e-10:
                observed_powers = observed_powers / mean_power

        masked_powers[i, obs_idx] = observed_powers
        masks[i, obs_idx] = 1.0

        # Save metadata for first sample
        if i == 0:
            first_sample_metadata = {
                'channel_metadata': channel_metadata,
                'hardware_metadata': hardware_metadata
            }

    return {
        'masked_powers': masked_powers,
        'masks': masks,
        'observed_indices': observed_indices,
        'powers_full': powers_full,
        'labels': labels,
        'optimal_powers': optimal_powers,
        'physics_metadata': first_sample_metadata  # NEW: Physics tracking
    }


# =============================================================================
# BACKWARD COMPATIBILITY WRAPPER
# =============================================================================

def generate_limited_probing_dataset(
        probe_bank: ProbeBank,
        n_samples: int,
        M: int,
        system_config,  # Can be SystemConfig or dict
        normalize: bool = True,
        seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Backward-compatible wrapper that supports both old and new configs.

    If system_config has Phase 1 fields, uses new physics pipeline.
    Otherwise, falls back to old behavior.
    """
    # Extract parameters
    if hasattr(system_config, '__dict__'):
        # It's a dataclass
        config_dict = system_config.__dict__
    else:
        # It's already a dict
        config_dict = system_config

    # Check if Phase 1 parameters exist
    has_phase1 = 'channel_source' in config_dict

    if has_phase1:
        # Use new physics pipeline
        return generate_limited_probing_dataset_with_physics(
            probe_bank=probe_bank,
            n_samples=n_samples,
            M=M,
            channel_source=config_dict.get('channel_source', 'python_synthetic'),
            realism_profile=config_dict.get('realism_profile', 'ideal'),
            use_custom_impairments=config_dict.get('use_custom_impairments', False),
            custom_impairments_config=config_dict.get('custom_impairments_config'),
            sigma_h_sq=config_dict.get('sigma_h_sq', 1.0),
            sigma_g_sq=config_dict.get('sigma_g_sq', 1.0),
            P_tx=config_dict.get('P_tx', 1.0),
            normalize=normalize,
            seed=seed
        )
    else:
        # Fall back to original implementation (for backward compatibility)
        # Import original function if it exists in another module
        from data.data_generation import generate_limited_probing_dataset as original_func
        return original_func(
            probe_bank=probe_bank,
            n_samples=n_samples,
            M=M,
            system_config=system_config,
            normalize=normalize,
            seed=seed
        )