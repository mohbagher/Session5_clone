"""
Physics Module - Phase 1 Modular Realism Architecture
=====================================================

Provides:
- Pluggable channel sources (Python, MATLAB interfaces)
- Modular impairment pipeline
- Pre-configured realism profiles
- Integrated data generation with physics tracking
"""

__version__ = "1.0.0-phase1"

from physics.channel_sources import (
    ChannelSource,
    PythonSyntheticSource,
    MATLABEngineSource,
    MATLABVerifiedSource,
    get_channel_source,
    list_available_sources
)

from physics.impairments import (
    CSIEstimationError,
    ChannelAging,
    QuantizationNoise,
    PhaseShifterQuantization,
    AmplitudeControl,
    MutualCoupling,
    ImpairmentPipeline
)

from physics.realism_profiles import (
    REALISM_PROFILES,
    get_profile,
    list_profiles,
    create_pipeline_from_profile,
    create_custom_pipeline,
    compare_profiles
)

from physics.data_generation_physics import (
    generate_channel_realization_with_physics,
    compute_probe_powers_with_physics,
    generate_limited_probing_dataset_with_physics,
    generate_limited_probing_dataset  # Backward compatible
)

__all__ = [
    # Channel sources
    'ChannelSource',
    'PythonSyntheticSource',
    'MATLABEngineSource',
    'MATLABVerifiedSource',
    'get_channel_source',
    'list_available_sources',

    # Impairments
    'CSIEstimationError',
    'ChannelAging',
    'QuantizationNoise',
    'PhaseShifterQuantization',
    'AmplitudeControl',
    'MutualCoupling',
    'ImpairmentPipeline',

    # Profiles
    'REALISM_PROFILES',
    'get_profile',
    'list_profiles',
    'create_pipeline_from_profile',
    'create_custom_pipeline',
    'compare_profiles',

    # Data generation
    'generate_channel_realization_with_physics',
    'compute_probe_powers_with_physics',
    'generate_limited_probing_dataset_with_physics',
    'generate_limited_probing_dataset',
]


def print_phase1_info():
    """Print Phase 1 implementation summary."""
    print("="*70)
    print("RIS Dashboard - Phase 1 Physics Extensions")
    print("="*70)
    print(f"\nVersion: {__version__}")
    print("\nAvailable Channel Sources:")
    sources = list_available_sources()
    for name, info in sources.items():
        status = info.get('status', 'AVAILABLE')
        symbol = "[OK]" if status == 'AVAILABLE' or 'verified' in status.lower() else "[WARN]"
        print(f"  {symbol} {name}: {info.get('description', 'N/A')}")

    print("\nAvailable Realism Profiles:")
    profiles = list_profiles()
    for name, info in profiles.items():
        print(f"  * {info['name']}")
        print(f"    {info['description']}")

    print("\n" + "="*70)
    print("For detailed integration guide, see PHASE1_INTEGRATION_GUIDE.py")
    print("="*70)
