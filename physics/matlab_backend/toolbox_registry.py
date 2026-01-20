"""
MATLAB Toolbox Registry
=======================
Catalog of verified MathWorks toolboxes and their capabilities.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolboxInfo:
    """Information about a MathWorks toolbox."""
    name: str
    required_version: str
    capabilities: List[str]
    description: str
    matlab_check_command: str  # Command to verify toolbox


@dataclass
class ScenarioTemplate:
    """Pre-configured scenario using MathWorks functions."""
    name: str
    toolbox: str
    description: str
    matlab_function: str  # Main MATLAB function to call
    default_params: Dict
    reference: str  # MathWorks doc URL or paper citation


# =============================================================================
# TOOLBOX DEFINITIONS
# =============================================================================

TOOLBOX_REGISTRY = {
    'communications': ToolboxInfo(
        name='Communications Toolbox',
        required_version='R2021b',
        capabilities=['rayleigh_fading', 'rician_fading', 'awgn', 'evm'],
        description='Industry-standard channel models',
        matlab_check_command="ver('comm')"
    ),

    '5g': ToolboxInfo(
        name='5G Toolbox',
        required_version='R2021b',
        capabilities=['nrCDL', 'nrTDL', '5g_channel_estimation'],
        description='3GPP-compliant 5G channel models',
        matlab_check_command="ver('5g')"
    ),

    'phased_array': ToolboxInfo(
        name='Phased Array System Toolbox',
        required_version='R2021b',
        capabilities=['conformal_array', 'beamforming', 'array_response'],
        description='Antenna array and beamforming tools',
        matlab_check_command="ver('phased')"
    ),

    'signal_processing': ToolboxInfo(
        name='Signal Processing Toolbox',
        required_version='R2021b',
        capabilities=['filtering', 'spectral_analysis'],
        description='Core signal processing functions',
        matlab_check_command="ver('signal')"
    )
}

# =============================================================================
# SCENARIO TEMPLATES (Based on MathWorks Examples)
# =============================================================================

SCENARIO_TEMPLATES = {
    'rayleigh_basic': ScenarioTemplate(
        name='Rayleigh Fading (Basic)',
        toolbox='communications',
        description='Basic Rayleigh fading using comm.RayleighChannel',
        matlab_function='generate_rayleigh_channel',
        default_params={
            'PathDelays': [0],
            'AveragePathGains': [0],
            'MaximumDopplerShift': 0,
            'SampleRate': 1e6
        },
        reference='https://www.mathworks.com/help/comm/ref/comm.rayleighchannel.html'
    ),

    'cdl_ris': ScenarioTemplate(
        name='CDL Channel with RIS',
        toolbox='5g',
        description='3GPP CDL channel model with RIS (from MathWorks RIS example)',
        matlab_function='generate_cdl_ris_channel',
        default_params={
            'DelayProfile': 'CDL-C',
            'CarrierFrequency': 28e9,
            'MaximumDopplerShift': 5,
            'RISSize': [8, 4, 2],
            'TransmitAntennaSize': [4, 2, 2, 1, 1],
            'ReceiveAntennaSize': [1, 1, 1, 1, 1]
        },
        reference='https://www.mathworks.com/help/comm/ug/model-ris-with-cdl-channels.html'
    ),

    'rician_los': ScenarioTemplate(
        name='Rician Fading (LOS)',
        toolbox='communications',
        description='Rician fading with line-of-sight component',
        matlab_function='generate_rician_channel',
        default_params={
            'KFactor': 10,  # dB
            'PathDelays': [0],
            'AveragePathGains': [0],
            'MaximumDopplerShift': 10
        },
        reference='https://www.mathworks.com/help/comm/ref/comm.ricianchannel.html'
    ),

    'tdl_urban': ScenarioTemplate(
        name='TDL Urban (5G)',
        toolbox='5g',
        description='3GPP TDL channel for urban scenarios',
        matlab_function='generate_tdl_channel',
        default_params={
            'DelayProfile': 'TDL-C',
            'DelaySpread': 300e-9,
            'MaximumDopplerShift': 30,
            'CarrierFrequency': 4e9
        },
        reference='https://www.mathworks.com/help/5g/ref/nrtdlchannel.html'
    )
}

# =============================================================================
# CAPABILITY ROUTING
# =============================================================================

CAPABILITY_TO_SCENARIO = {
    'rayleigh_fading': 'rayleigh_basic',
    'cdl_ris_channel': 'cdl_ris',
    'rician_los': 'rician_los',
    'urban_5g': 'tdl_urban'
}


class ToolboxManager:
    """Manage available MATLAB toolboxes and scenarios."""

    def __init__(self, matlab_engine=None):
        self.engine = matlab_engine
        self._available_toolboxes = None

    def check_available_toolboxes(self) -> Dict[str, bool]:
        """
        Check which toolboxes are installed in MATLAB.

        Returns:
            Dict mapping toolbox name to availability
        """
        if self.engine is None:
            logger.warning("MATLAB engine not available, cannot check toolboxes")
            return {name: False for name in TOOLBOX_REGISTRY.keys()}

        available = {}

        for toolbox_key, toolbox_info in TOOLBOX_REGISTRY.items():
            try:
                # Run MATLAB check command
                result = self.engine.eval(toolbox_info.matlab_check_command, nargout=0)
                available[toolbox_key] = True
                logger.info(f"✓ {toolbox_info.name} available")
            except:
                available[toolbox_key] = False
                logger.warning(f"✗ {toolbox_info.name} NOT available")

        self._available_toolboxes = available
        return available

    def get_available_scenarios(self) -> List[str]:
        """Get list of scenarios that can be run with available toolboxes."""
        if self._available_toolboxes is None:
            self.check_available_toolboxes()

        available_scenarios = []

        for scenario_key, scenario in SCENARIO_TEMPLATES.items():
            if self._available_toolboxes.get(scenario.toolbox, False):
                available_scenarios.append(scenario_key)

        return available_scenarios

    def get_scenario(self, scenario_name: str) -> Optional[ScenarioTemplate]:
        """Get scenario template by name."""
        return SCENARIO_TEMPLATES.get(scenario_name)

    def get_toolbox_info(self, toolbox_name: str) -> Optional[ToolboxInfo]:
        """Get toolbox information."""
        return TOOLBOX_REGISTRY.get(toolbox_name)