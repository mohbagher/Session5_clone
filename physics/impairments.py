"""
Modular Impairment Blocks
=========================
Plugin-style realism pipeline that can be toggled from dashboard.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class ImpairmentMetadata:
    """Track which impairments were applied."""
    impairment_name: str
    enabled: bool
    parameters: Dict

    def to_dict(self) -> Dict:
        return {
            'name': self.impairment_name,
            'enabled': self.enabled,
            'parameters': self.parameters
        }


class CSIEstimationError:
    """Imperfect Channel State Information (CSI)."""

    def __init__(self, error_variance_db: float = -20, enabled: bool = False):
        self.error_variance_db = error_variance_db
        self.enabled = enabled

    def apply(self, h: np.ndarray, g: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray, ImpairmentMetadata]:
        if not self.enabled:
            metadata = ImpairmentMetadata("CSI_Estimation_Error", False, {})
            return h, g, metadata

        error_variance_linear = 10 ** (self.error_variance_db / 10)
        h_power = np.mean(np.abs(h) ** 2)
        g_power = np.mean(np.abs(g) ** 2)
        h_error_var = error_variance_linear * h_power
        g_error_var = error_variance_linear * g_power
        h_error = np.sqrt(h_error_var / 2) * (rng.randn(len(h)) + 1j * rng.randn(len(h)))
        g_error = np.sqrt(g_error_var / 2) * (rng.randn(len(g)) + 1j * rng.randn(len(g)))
        h_impaired = h + h_error
        g_impaired = g + g_error

        metadata = ImpairmentMetadata(
            "CSI_Estimation_Error", True,
            {'error_variance_db': self.error_variance_db, 'error_variance_linear': error_variance_linear}
        )
        return h_impaired, g_impaired, metadata


class ChannelAging:
    """Outdated CSI due to mobility/time variation."""

    def __init__(self, doppler_hz: float = 0, feedback_delay_ms: float = 0, enabled: bool = False):
        self.doppler_hz = doppler_hz
        self.feedback_delay_ms = feedback_delay_ms
        self.enabled = enabled

    def apply(self, h: np.ndarray, g: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray, ImpairmentMetadata]:
        if not self.enabled or self.doppler_hz == 0:
            return h, g, ImpairmentMetadata("Channel_Aging", False, {})

        tau = self.feedback_delay_ms / 1000
        from scipy.special import j0
        rho = j0(2 * np.pi * self.doppler_hz * tau)

        innovation_h = np.sqrt((1 - rho**2) / 2) * (rng.randn(len(h)) + 1j * rng.randn(len(h)))
        innovation_g = np.sqrt((1 - rho**2) / 2) * (rng.randn(len(g)) + 1j * rng.randn(len(g)))

        h_power = np.mean(np.abs(h) ** 2)
        g_power = np.mean(np.abs(g) ** 2)
        innovation_h *= np.sqrt(h_power)
        innovation_g *= np.sqrt(g_power)

        h_aged = rho * h + innovation_h
        g_aged = rho * g + innovation_g

        metadata = ImpairmentMetadata("Channel_Aging", True, {
            'doppler_hz': self.doppler_hz,
            'feedback_delay_ms': self.feedback_delay_ms,
            'correlation_coefficient': float(rho)
        })
        return h_aged, g_aged, metadata


class QuantizationNoise:
    """ADC/DAC quantization effects."""

    def __init__(self, adc_bits: int = 16, enabled: bool = False):
        self.adc_bits = adc_bits
        self.enabled = enabled

    def apply(self, h: np.ndarray, g: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray, ImpairmentMetadata]:
        if not self.enabled or self.adc_bits >= 16:
            return h, g, ImpairmentMetadata("Quantization_Noise", False, {})

        def quantize_complex(x, bits):
            max_val = np.max(np.abs(x))
            levels = 2 ** bits
            step = (2 * max_val) / levels
            x_real_q = np.round(x.real / step) * step
            x_imag_q = np.round(x.imag / step) * step
            return x_real_q + 1j * x_imag_q

        h_quantized = quantize_complex(h, self.adc_bits)
        g_quantized = quantize_complex(g, self.adc_bits)

        metadata = ImpairmentMetadata("Quantization_Noise", True, {
            'adc_bits': self.adc_bits,
            'quantization_levels': 2 ** self.adc_bits
        })
        return h_quantized, g_quantized, metadata


class PhaseShifterQuantization:
    """Finite phase shifter resolution."""

    def __init__(self, phase_bits: int = 8, enabled: bool = False):
        self.phase_bits = phase_bits
        self.enabled = enabled

    def apply_to_phases(self, phases: np.ndarray) -> Tuple[np.ndarray, ImpairmentMetadata]:
        if not self.enabled:
            return phases, ImpairmentMetadata("PhaseShifter_Quantization", False, {})

        levels = 2 ** self.phase_bits
        step = 2 * np.pi / levels
        phases_quantized = np.round(phases / step) * step
        phases_quantized = np.mod(phases_quantized, 2 * np.pi)

        metadata = ImpairmentMetadata("PhaseShifter_Quantization", True, {
            'phase_bits': self.phase_bits,
            'phase_levels': levels,
            'phase_step_degrees': float(np.degrees(step))
        })
        return phases_quantized, metadata


class AmplitudeControl:
    """Non-ideal amplitude control."""

    def __init__(self, insertion_loss_db: float = 0.5, amplitude_variation_db: float = 0.2, enabled: bool = False):
        self.insertion_loss_db = insertion_loss_db
        self.amplitude_variation_db = amplitude_variation_db
        self.enabled = enabled

    def apply_to_reflection(self, reflection_coeffs: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, ImpairmentMetadata]:
        if not self.enabled:
            return reflection_coeffs, ImpairmentMetadata("Amplitude_Control", False, {})

        loss_linear = 10 ** (-self.insertion_loss_db / 20)
        variation_std_linear = 10 ** (self.amplitude_variation_db / 20) - 1
        amplitude_errors = 1 + variation_std_linear * rng.randn(len(reflection_coeffs))
        reflection_impaired = reflection_coeffs * loss_linear * amplitude_errors

        metadata = ImpairmentMetadata("Amplitude_Control", True, {
            'insertion_loss_db': self.insertion_loss_db,
            'amplitude_variation_db': self.amplitude_variation_db,
            'average_amplitude': float(np.mean(np.abs(reflection_impaired)))
        })
        return reflection_impaired, metadata


class MutualCoupling:
    """Electromagnetic coupling between RIS elements."""

    def __init__(self, coupling_strength: float = 0.1, enabled: bool = False):
        self.coupling_strength = coupling_strength
        self.enabled = enabled

    def apply_to_channel(self, h: np.ndarray, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray, ImpairmentMetadata]:
        if not self.enabled or self.coupling_strength == 0:
            return h, g, ImpairmentMetadata("Mutual_Coupling", False, {})

        N = len(h)
        C = np.eye(N, dtype=complex)
        for i in range(N - 1):
            C[i, i + 1] = self.coupling_strength
            C[i + 1, i] = self.coupling_strength
        C = C / np.max(np.abs(np.linalg.eigvals(C)))

        h_coupled = C @ h
        g_coupled = C @ g

        metadata = ImpairmentMetadata("Mutual_Coupling", True, {
            'coupling_strength': self.coupling_strength,
            'coupling_matrix_condition': float(np.linalg.cond(C))
        })
        return h_coupled, g_coupled, metadata


class ImpairmentPipeline:
    """Orchestrates application of all impairments."""

    def __init__(self, csi_error=None, channel_aging=None, quantization=None,
                 phase_quantization=None, amplitude_control=None, mutual_coupling=None):
        self.csi_error = csi_error or CSIEstimationError(enabled=False)
        self.channel_aging = channel_aging or ChannelAging(enabled=False)
        self.quantization = quantization or QuantizationNoise(enabled=False)
        self.phase_quantization = phase_quantization or PhaseShifterQuantization(enabled=False)
        self.amplitude_control = amplitude_control or AmplitudeControl(enabled=False)
        self.mutual_coupling = mutual_coupling or MutualCoupling(enabled=False)

    def apply_channel_impairments(self, h: np.ndarray, g: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray, Dict]:
        metadata_list = []
        h, g, meta = self.mutual_coupling.apply_to_channel(h, g)
        metadata_list.append(meta)
        h, g, meta = self.csi_error.apply(h, g, rng)
        metadata_list.append(meta)
        h, g, meta = self.channel_aging.apply(h, g, rng)
        metadata_list.append(meta)
        h, g, meta = self.quantization.apply(h, g, rng)
        metadata_list.append(meta)
        return h, g, {'impairments': [m.to_dict() for m in metadata_list]}

    def apply_hardware_impairments(self, phases: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, Dict]:
        phases, meta = self.phase_quantization.apply_to_phases(phases)
        return phases, {'impairments': [meta.to_dict()]}

    def apply_amplitude_impairments(self, reflection_coeffs: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, Dict]:
        reflection_coeffs, meta = self.amplitude_control.apply_to_reflection(reflection_coeffs, rng)
        return reflection_coeffs, {'impairments': [meta.to_dict()]}

    def get_configuration_summary(self) -> Dict:
        return {
            'channel_impairments': {
                'csi_error': {'enabled': self.csi_error.enabled, 'error_variance_db': self.csi_error.error_variance_db},
                'channel_aging': {'enabled': self.channel_aging.enabled, 'doppler_hz': self.channel_aging.doppler_hz, 'feedback_delay_ms': self.channel_aging.feedback_delay_ms},
                'quantization': {'enabled': self.quantization.enabled, 'adc_bits': self.quantization.adc_bits},
                'mutual_coupling': {'enabled': self.mutual_coupling.enabled, 'coupling_strength': self.mutual_coupling.coupling_strength}
            },
            'hardware_impairments': {
                'phase_quantization': {'enabled': self.phase_quantization.enabled, 'phase_bits': self.phase_quantization.phase_bits},
                'amplitude_control': {'enabled': self.amplitude_control.enabled, 'insertion_loss_db': self.amplitude_control.insertion_loss_db, 'amplitude_variation_db': self.amplitude_control.amplitude_variation_db}
            }
        }
