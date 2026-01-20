"""
Physics Module
==============
Channel generation, impairments, and realism profiles.

Phase 1: Modular physics with impairment pipelines
Phase 2: MATLAB backend integration
"""

from physics.channel_sources import (
    ChannelSource,
    PythonSyntheticSource,
    MATLABEngineSource,
    MATLABVerifiedSource
)

from physics.impairments import (
    ImpairmentPipeline
)

from physics.realism_profiles import (
    list_profiles,
    create_pipeline_from_profile
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def list_available_sources():
    """
    List all available channel sources.

    Returns:
        list: Available source names
    """
    return [
        'python_synthetic',
        'matlab_engine',
        'matlab_verified'
    ]


def create_source_from_name(source_name: str, **kwargs) -> ChannelSource:
    """
    Create a channel source instance from name.

    Args:
        source_name: Name of source ('python_synthetic', 'matlab_engine', 'matlab_verified')
        **kwargs: Additional parameters for the source

    Returns:
        ChannelSource instance

    Raises:
        ValueError: If source_name is unknown

    Examples:
        >>> source = create_source_from_name('python_synthetic')
        >>> source = create_source_from_name('matlab_engine', scenario='cdl_ris')
    """

    source_registry = {
        'python_synthetic': PythonSyntheticSource,
        'matlab_engine': MATLABEngineSource,
        'matlab_verified': MATLABVerifiedSource
    }

    if source_name not in source_registry:
        raise ValueError(
            f"Unknown source: {source_name}. "
            f"Available sources: {list(source_registry.keys())}"
        )

    source_class = source_registry[source_name]

    return source_class(**kwargs)


def get_source_info(source_name: str) -> dict:
    """
    Get information about a channel source.

    Args:
        source_name: Name of source

    Returns:
        dict with source information
    """

    source_info_registry = {
        'python_synthetic': {
            'name': 'Python Synthetic',
            'description': 'Built-in numpy-based Rayleigh fading',
            'backend': 'python',
            'verified': True,
            'fast': True,
            'requires': []
        },
        'matlab_engine': {
            'name': 'MATLAB Engine',
            'description': 'Live MATLAB channel generation using verified toolboxes',
            'backend': 'matlab',
            'verified': True,
            'fast': False,
            'requires': ['MATLAB R2021b+', 'Communications Toolbox', 'MATLAB Engine for Python']
        },
        'matlab_verified': {
            'name': 'MATLAB Verified Data',
            'description': 'Load pre-verified .mat files',
            'backend': 'matlab',
            'verified': True,
            'fast': True,
            'requires': ['Pre-generated .mat files']
        }
    }

    return source_info_registry.get(source_name, {})


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Channel sources
    'ChannelSource',
    'PythonSyntheticSource',
    'MATLABEngineSource',
    'MATLABVerifiedSource',
    'list_available_sources',
    'create_source_from_name',
    'get_source_info',

    # Impairments
    'ImpairmentPipeline',

    # Realism profiles
    'list_profiles',
    'create_pipeline_from_profile'
]