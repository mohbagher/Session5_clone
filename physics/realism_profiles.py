"""
Realism Profile Presets
========================
Pre-configured impairment bundles for different scenarios.
"""

from typing import Dict
from physics.impairments import *


REALISM_PROFILES = {
    'ideal': {
        'name': 'Ideal (No Impairments)',
        'description': 'Perfect CSI, infinite precision, no hardware limits',
        'use_case': 'Theoretical upper bound, algorithm development',
        'config': {
            'csi_error': {'enabled': False},
            'channel_aging': {'enabled': False},
            'quantization': {'enabled': False},
            'phase_quantization': {'enabled': False},
            'amplitude_control': {'enabled': False},
            'mutual_coupling': {'enabled': False}
        }
    },
    'mild_impairments': {
        'name': 'Mild Impairments',
        'description': 'Good hardware, near-ideal conditions',
        'use_case': 'High-quality RIS with controlled environment',
        'config': {
            'csi_error': {'enabled': True, 'error_variance_db': -30},
            'channel_aging': {'enabled': True, 'doppler_hz': 5, 'feedback_delay_ms': 10},
            'quantization': {'enabled': True, 'adc_bits': 14},
            'phase_quantization': {'enabled': True, 'phase_bits': 6},
            'amplitude_control': {'enabled': True, 'insertion_loss_db': 0.3, 'amplitude_variation_db': 0.1},
            'mutual_coupling': {'enabled': True, 'coupling_strength': 0.05}
        }
    },
    'moderate_impairments': {
        'name': 'Moderate Impairments',
        'description': 'Typical indoor RIS deployment',
        'use_case': 'Realistic indoor scenario (office, home)',
        'config': {
            'csi_error': {'enabled': True, 'error_variance_db': -20},
            'channel_aging': {'enabled': True, 'doppler_hz': 10, 'feedback_delay_ms': 20},
            'quantization': {'enabled': True, 'adc_bits': 10},
            'phase_quantization': {'enabled': True, 'phase_bits': 4},
            'amplitude_control': {'enabled': True, 'insertion_loss_db': 0.5, 'amplitude_variation_db': 0.2},
            'mutual_coupling': {'enabled': True, 'coupling_strength': 0.1}
        }
    },
    'severe_impairments': {
        'name': 'Severe Impairments',
        'description': 'Challenging outdoor/mobile environment',
        'use_case': 'Vehicular, high mobility, budget hardware',
        'config': {
            'csi_error': {'enabled': True, 'error_variance_db': -15},
            'channel_aging': {'enabled': True, 'doppler_hz': 50, 'feedback_delay_ms': 50},
            'quantization': {'enabled': True, 'adc_bits': 8},
            'phase_quantization': {'enabled': True, 'phase_bits': 3},
            'amplitude_control': {'enabled': True, 'insertion_loss_db': 1.0, 'amplitude_variation_db': 0.5},
            'mutual_coupling': {'enabled': True, 'coupling_strength': 0.2}
        }
    },
    'worst_case': {
        'name': 'Worst Case',
        'description': 'Extreme conditions for robustness testing',
        'use_case': 'Stress testing, reliability validation',
        'config': {
            'csi_error': {'enabled': True, 'error_variance_db': -10},
            'channel_aging': {'enabled': True, 'doppler_hz': 100, 'feedback_delay_ms': 100},
            'quantization': {'enabled': True, 'adc_bits': 6},
            'phase_quantization': {'enabled': True, 'phase_bits': 2},
            'amplitude_control': {'enabled': True, 'insertion_loss_db': 2.0, 'amplitude_variation_db': 1.0},
            'mutual_coupling': {'enabled': True, 'coupling_strength': 0.3}
        }
    }
}


def get_profile(profile_name: str) -> Dict:
    """Get realism profile configuration."""
    if profile_name not in REALISM_PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}. Available: {list(REALISM_PROFILES.keys())}")
    return REALISM_PROFILES[profile_name]


def list_profiles() -> Dict[str, Dict]:
    """List all available profiles with descriptions."""
    return {name: {'name': prof['name'], 'description': prof['description'], 'use_case': prof['use_case']}
            for name, prof in REALISM_PROFILES.items()}


def create_pipeline_from_profile(profile_name: str):
    """Create ImpairmentPipeline from profile name."""
    profile = get_profile(profile_name)
    config = profile['config']
    return ImpairmentPipeline(
        csi_error=CSIEstimationError(**config['csi_error']),
        channel_aging=ChannelAging(**config['channel_aging']),
        quantization=QuantizationNoise(**config['quantization']),
        phase_quantization=PhaseShifterQuantization(**config['phase_quantization']),
        amplitude_control=AmplitudeControl(**config['amplitude_control']),
        mutual_coupling=MutualCoupling(**config['mutual_coupling'])
    )


def create_custom_pipeline(custom_config: Dict):
    """Create ImpairmentPipeline from custom configuration."""
    return ImpairmentPipeline(
        csi_error=CSIEstimationError(**custom_config.get('csi_error', {'enabled': False})),
        channel_aging=ChannelAging(**custom_config.get('channel_aging', {'enabled': False})),
        quantization=QuantizationNoise(**custom_config.get('quantization', {'enabled': False})),
        phase_quantization=PhaseShifterQuantization(**custom_config.get('phase_quantization', {'enabled': False})),
        amplitude_control=AmplitudeControl(**custom_config.get('amplitude_control', {'enabled': False})),
        mutual_coupling=MutualCoupling(**custom_config.get('mutual_coupling', {'enabled': False}))
    )


def compare_profiles() -> str:
    """Generate comparison table of all profiles."""
    output = "\n" + "="*80 + "\n"
    output += "REALISM PROFILE COMPARISON\n"
    output += "="*80 + "\n\n"
    for name, profile in REALISM_PROFILES.items():
        output += f"Profile: {profile['name']}\n"
        output += f"  Description: {profile['description']}\n"
        output += f"  Use Case: {profile['use_case']}\n"
        cfg = profile['config']
        output += f"    CSI Error: {cfg['csi_error']['error_variance_db'] if cfg['csi_error']['enabled'] else 'OFF'} dB\n"
        output += f"    Doppler: {cfg['channel_aging']['doppler_hz'] if cfg['channel_aging']['enabled'] else 'OFF'} Hz\n"
        output += f"    Phase Bits: {cfg['phase_quantization']['phase_bits'] if cfg['phase_quantization']['enabled'] else 'Infinite'}\n\n"
    output += "="*80 + "\n"
    return output
