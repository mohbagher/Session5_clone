"""
Realism Profiles
================
Pre-configured bundles of impairments for different scenarios.
"""

from physics.impairments import ImpairmentPipeline
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# PROFILE DEFINITIONS
# ============================================================================

REALISM_PROFILES = {
    'ideal': {
        'name': 'Ideal Conditions',
        'description': 'No impairments - theoretical upper bound',
        'impairments': []
    },

    'mild_impairments': {
        'name': 'Mild Impairments (Lab)',
        'description': 'High-quality lab equipment',
        'impairments': [
            {
                'type': 'csi_error',
                'enabled': True,
                'error_variance_db': -30
            },
            {
                'type': 'channel_aging',
                'enabled': True,
                'doppler_hz': 5,
                'feedback_delay_ms': 10
            },
            {
                'type': 'phase_quantization',
                'enabled': True,
                'phase_bits': 6
            }
        ]
    },

    'moderate_impairments': {
        'name': 'Moderate Impairments (Indoor)',
        'description': 'Typical indoor deployment',
        'impairments': [
            {
                'type': 'csi_error',
                'enabled': True,
                'error_variance_db': -20
            },
            {
                'type': 'channel_aging',
                'enabled': True,
                'doppler_hz': 10,
                'feedback_delay_ms': 20
            },
            {
                'type': 'phase_quantization',
                'enabled': True,
                'phase_bits': 4
            },
            {
                'type': 'quantization',
                'enabled': True,
                'adc_bits': 10
            }
        ]
    },

    'severe_impairments': {
        'name': 'Severe Impairments (Outdoor)',
        'description': 'Outdoor/vehicular scenarios',
        'impairments': [
            {
                'type': 'csi_error',
                'enabled': True,
                'error_variance_db': -15
            },
            {
                'type': 'channel_aging',
                'enabled': True,
                'doppler_hz': 50,
                'feedback_delay_ms': 50
            },
            {
                'type': 'phase_quantization',
                'enabled': True,
                'phase_bits': 3
            },
            {
                'type': 'quantization',
                'enabled': True,
                'adc_bits': 8
            }
        ]
    },

    'worst_case': {
        'name': 'Worst Case (Stress Test)',
        'description': 'Robustness testing',
        'impairments': [
            {
                'type': 'csi_error',
                'enabled': True,
                'error_variance_db': -10
            },
            {
                'type': 'channel_aging',
                'enabled': True,
                'doppler_hz': 100,
                'feedback_delay_ms': 100
            },
            {
                'type': 'phase_quantization',
                'enabled': True,
                'phase_bits': 2
            },
            {
                'type': 'quantization',
                'enabled': True,
                'adc_bits': 6
            }
        ]
    }
}


# ============================================================================
# PUBLIC API
# ============================================================================

def list_profiles() -> List[str]:
    """
    List available realism profiles.

    Returns:
        List of profile names
    """
    return list(REALISM_PROFILES.keys())


def create_pipeline_from_profile(profile_name: str) -> ImpairmentPipeline:
    """
    Create an impairment pipeline from a realism profile.

    Args:
        profile_name: Name of profile ('ideal', 'mild_impairments', etc.)

    Returns:
        ImpairmentPipeline configured with profile's impairments

    Raises:
        ValueError: If profile_name is unknown
    """

    if profile_name not in REALISM_PROFILES:
        raise ValueError(
            f"Unknown profile: {profile_name}. "
            f"Available profiles: {list(REALISM_PROFILES.keys())}"
        )

    profile = REALISM_PROFILES[profile_name]

    pipeline = ImpairmentPipeline()

    for impairment in profile['impairments']:
        pipeline.add_block(impairment['type'], impairment)

    logger.info(f"Created pipeline from profile: {profile_name}")

    return pipeline


def create_custom_pipeline(impairment_configs: list) -> ImpairmentPipeline:
    """
    Create custom impairment pipeline from list of configs.

    Args:
        impairment_configs: List of dicts with 'type' and parameters

    Returns:
        ImpairmentPipeline configured with custom impairments
    """

    pipeline = ImpairmentPipeline()

    for config in impairment_configs:
        imp_type = config.get('type')
        pipeline.add_block(imp_type, config)

    logger.info(f"Created custom pipeline with {len(impairment_configs)} impairments")

    return pipeline