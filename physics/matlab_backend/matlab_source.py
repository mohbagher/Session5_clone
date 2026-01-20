"""
MATLAB Engine Channel Source
=============================
Channel source using verified MathWorks toolbox functions.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging
from pathlib import Path

from physics.matlab_backend.session_manager import get_session_manager
from physics.matlab_backend.toolbox_registry import (
    ToolboxManager,
    SCENARIO_TEMPLATES,
    ScenarioTemplate
)
from physics.matlab_backend.script_generator import MATLABScriptGenerator

logger = logging.getLogger(__name__)


class MATLABEngineSource:
    """
    Channel source using MATLAB Engine with MathWorks verified toolboxes.

    Supports:
    - Communications Toolbox (Rayleigh, Rician)
    - 5G Toolbox (CDL, TDL)
    - Custom scenarios based on MathWorks examples
    """

    def __init__(
        self,
        scenario: str = 'rayleigh_basic',
        **scenario_params
    ):
        """
        Initialize MATLAB Engine source.

        Args:
            scenario: Scenario name from SCENARIO_TEMPLATES
            **scenario_params: Override default scenario parameters
        """
        self.scenario_name = scenario
        self.scenario_params = scenario_params

        # Get session manager
        self.session_manager = get_session_manager()

        # Start MATLAB session if not already running
        if not self.session_manager.is_session_active():
            logger.info("Starting MATLAB Engine session...")
            success = self.session_manager.start_session()
            if not success:
                raise RuntimeError(
                    "Failed to start MATLAB Engine. "
                    "Ensure MATLAB is installed and matlab.engine is configured."
                )

        # Get MATLAB engine
        self.engine = self.session_manager.get_engine()

        # Initialize toolbox manager
        self.toolbox_manager = ToolboxManager(self.engine)

        # Check available toolboxes
        self.available_toolboxes = self.toolbox_manager.check_available_toolboxes()

        # Get scenario template
        self.scenario_template = self.toolbox_manager.get_scenario(scenario)
        if self.scenario_template is None:
            raise ValueError(f"Unknown scenario: {scenario}")

        # Check if required toolbox is available
        if not self.available_toolboxes.get(self.scenario_template.toolbox, False):
            raise RuntimeError(
                f"Required toolbox '{self.scenario_template.toolbox}' not available. "
                f"Scenario '{scenario}' cannot be run."
            )

        # Generate MATLAB scripts
        self._setup_matlab_environment()

        logger.info(f"MATLAB source initialized with scenario: {scenario}")

    def _setup_matlab_environment(self):
        """Setup MATLAB environment (generate scripts, add paths)."""

        # Generate MATLAB scripts
        script_gen = MATLABScriptGenerator()
        script_gen.generate_all_scripts()

        # Add script directory to MATLAB path
        script_dir = Path('physics/matlab_backend/matlab_scripts').resolve()
        self.engine.addpath(str(script_dir), nargout=0)

        logger.info(f"Added MATLAB script path: {script_dir}")

    def generate_channels(
        self,
        N: int,
        K: int,
        num_samples: int = 1,
        sigma_h_sq: float = 1.0,
        sigma_g_sq: float = 1.0,
        seed: Optional[int] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate channel realizations using MATLAB toolbox.

        Args:
            N: Number of RIS elements
            K: Number of transmit antennas
            num_samples: Number of channel realizations
            sigma_h_sq: BS-RIS channel variance
            sigma_g_sq: RIS-UE channel variance
            seed: Random seed
            **kwargs: Additional scenario-specific parameters

        Returns:
            h_all: BS-RIS channels (N, num_samples)
            g_all: RIS-UE channels (N, num_samples)
            metadata: Generation metadata
        """

        logger.info(f"Generating {num_samples} channel realizations using MATLAB ({self.scenario_name})...")

        # Select generation method based on scenario
        if self.scenario_name == 'rayleigh_basic':
            return self._generate_rayleigh(N, num_samples, sigma_h_sq, sigma_g_sq, seed)

        elif self.scenario_name == 'cdl_ris':
            return self._generate_cdl_ris(N, num_samples, seed, **kwargs)

        elif self.scenario_name == 'rician_los':
            return self._generate_rician(N, num_samples, sigma_h_sq, sigma_g_sq, seed, **kwargs)

        else:
            raise NotImplementedError(f"Scenario '{self.scenario_name}' not yet implemented")

    def _generate_rayleigh(
        self,
        N: int,
        num_samples: int,
        sigma_h_sq: float,
        sigma_g_sq: float,
        seed: Optional[int]
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Generate Rayleigh fading channels using Communications Toolbox."""

        h_all = np.zeros((N, num_samples), dtype=complex)
        g_all = np.zeros((N, num_samples), dtype=complex)

        metadata_list = []

        for i in range(num_samples):
            # Use different seed for each sample
            sample_seed = seed + i if seed is not None else None

            # Call MATLAB function
            h_matlab, g_matlab, meta_matlab = self.engine.generate_rayleigh_channel(
                float(N),
                sigma_h_sq,
                sigma_g_sq,
                float(sample_seed) if sample_seed is not None else [],
                nargout=3
            )

            # Convert to numpy
            h_all[:, i] = np.array(h_matlab).flatten()
            g_all[:, i] = np.array(g_matlab).flatten()

            # Extract metadata
            metadata_list.append(self._matlab_struct_to_dict(meta_matlab))

        # Aggregate metadata
        metadata = {
            'backend_name': 'matlab',
            'toolbox': 'communications',
            'scenario': 'rayleigh_basic',
            'function': 'generate_rayleigh_channel',
            'method': 'rayleigh_fading',
            'num_samples': num_samples,
            'N': N,
            'sigma_h_sq': sigma_h_sq,
            'sigma_g_sq': sigma_g_sq,
            'seed': seed,
            'matlab_version': self.session_manager.get_session_info().matlab_version,
            'sample_metadata': metadata_list,
            'reference': self.scenario_template.reference
        }

        logger.info(f"Generated {num_samples} Rayleigh channel realizations")

        return h_all, g_all, metadata

    def _generate_cdl_ris(
        self,
        N: int,
        num_samples: int,
        seed: Optional[int],
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Generate CDL-RIS channels using 5G Toolbox."""

        # Build parameters struct for MATLAB
        params = {
            'N': float(N),
            'CarrierFrequency': kwargs.get('carrier_frequency', 28e9),
            'DelayProfile': kwargs.get('delay_profile', 'CDL-C'),
            'MaximumDopplerShift': kwargs.get('doppler_shift', 5),
            'seed': float(seed) if seed is not None else 0
        }

        h_all = np.zeros((N, num_samples), dtype=complex)
        g_all = np.zeros((N, num_samples), dtype=complex)

        metadata_list = []

        for i in range(num_samples):
            # Update seed for each sample
            params['seed'] = float(seed + i) if seed is not None else float(i)

            # Convert params to MATLAB struct
            params_matlab = self._dict_to_matlab_struct(params)

            # Call MATLAB function
            h_matlab, g_matlab, meta_matlab = self.engine.generate_cdl_ris_channel(
                params_matlab,
                nargout=3
            )

            # Convert to numpy
            h_all[:, i] = np.array(h_matlab).flatten()
            g_all[:, i] = np.array(g_matlab).flatten()

            # Extract metadata
            metadata_list.append(self._matlab_struct_to_dict(meta_matlab))

        # Aggregate metadata
        metadata = {
            'backend_name': 'matlab',
            'toolbox': '5g',
            'scenario': 'cdl_ris',
            'function': 'generate_cdl_ris_channel',
            'method': 'cdl_ris',
            'num_samples': num_samples,
            'N': N,
            'carrier_frequency': params['CarrierFrequency'],
            'delay_profile': params['DelayProfile'],
            'doppler_shift': params['MaximumDopplerShift'],
            'seed': seed,
            'matlab_version': self.session_manager.get_session_info().matlab_version,
            'sample_metadata': metadata_list,
            'reference': self.scenario_template.reference
        }

        logger.info(f"Generated {num_samples} CDL-RIS channel realizations")

        return h_all, g_all, metadata

    def _generate_rician(
        self,
        N: int,
        num_samples: int,
        sigma_h_sq: float,
        sigma_g_sq: float,
        seed: Optional[int],
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Generate Rician fading channels (placeholder for future expansion)."""
        raise NotImplementedError(
            "Rician channel generation not yet implemented. "
            "This is a placeholder for Phase 2 expansion."
        )

    def _dict_to_matlab_struct(self, d: Dict) -> Any:
        """Convert Python dict to MATLAB struct."""
        # MATLAB Engine handles dict conversion automatically in recent versions
        return d

    def _matlab_struct_to_dict(self, s: Any) -> Dict:
        """Convert MATLAB struct to Python dict."""
        if s is None:
            return {}

        result = {}
        try:
            # MATLAB struct fields
            for field in s._fieldnames:
                value = getattr(s, field)

                # Convert MATLAB types to Python
                if hasattr(value, '__iter__') and not isinstance(value, str):
                    value = np.array(value)

                result[field] = value
        except:
            # Fallback: if conversion fails, return empty dict
            logger.warning("Failed to convert MATLAB struct to dict")

        return result

    def get_info(self) -> Dict:
        """Get information about the MATLAB source."""
        session_info = self.session_manager.get_session_info()

        return {
            'backend': 'matlab_engine',
            'scenario': self.scenario_name,
            'scenario_description': self.scenario_template.description,
            'toolbox': self.scenario_template.toolbox,
            'matlab_version': session_info.matlab_version if session_info else 'unknown',
            'available_toolboxes': [
                name for name, available in self.available_toolboxes.items() if available
            ],
            'session_status': session_info.status if session_info else 'disconnected',
            'reference': self.scenario_template.reference
        }