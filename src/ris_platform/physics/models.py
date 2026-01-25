"""
Physics Models (Interface Compliant)
====================================
Correctly implements PhysicsModel ABC and matches Component signatures.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

# Import Interfaces
from src.ris_platform.core.interfaces import PhysicsModel

# Import Components (Safe Import)
try:
    from src.ris_platform.physics.components.unit_cell import IdealUnitCell, VaractorUnitCell
    from src.ris_platform.physics.components.coupling import NoCoupling, GeometricCoupling
    from src.ris_platform.physics.components.wavefront import PlanarWavefront
    from src.ris_platform.physics.components.aging import JakesAging
except ImportError as e:
    logging.getLogger(__name__).warning(f"Component import warning: {e}")
    IdealUnitCell = VaractorUnitCell = None
    NoCoupling = GeometricCoupling = None
    PlanarWavefront = JakesAging = None

logger = logging.getLogger(__name__)

# --- 1. REALISTIC PHYSICS MODEL ---
class RealisticPhysicsModel(PhysicsModel):
    def __init__(self, unit_cell=None, coupling=None, wavefront=None, aging=None):
        self.unit_cell = unit_cell
        self.coupling = coupling
        self.wavefront = wavefront
        self.aging = aging

    def compute_received_power(self, h, g, phases, **kwargs):
        # 1. Apply Aging
        if self.aging:
            h, g = self.aging.age_channel(h, g, time_delta=0.0) # Delta handled internally or via config

        # 2. Physics Chain
        # Fallback if unit_cell is None
        if self.unit_cell:
            gamma = self.unit_cell.compute_reflection(phases)
        else:
            gamma = np.exp(1j * phases)

        # Coupling
        if self.coupling:
            gamma = self.coupling.apply_coupling(gamma)

        # Wavefront
        cascaded = h.conjugate() * g
        signal = np.dot(gamma, cascaded)

        return np.abs(signal)**2

    def get_metadata(self) -> Dict[str, Any]:
        """Required by PhysicsModel interface."""
        return {
            'name': 'RealisticPhysicsModel',
            'unit_cell': type(self.unit_cell).__name__ if self.unit_cell else 'None',
            'coupling': type(self.coupling).__name__ if self.coupling else 'None'
        }

# --- 2. IDEAL PHYSICS MODEL ---
class IdealPhysicsModel(PhysicsModel):
    """Wrapper for the Realistic Model but forced to use Ideal components."""
    def __init__(self):
        self.unit_cell = IdealUnitCell() if IdealUnitCell else None
        self.coupling = NoCoupling() if NoCoupling else None
        self.wavefront = PlanarWavefront() if PlanarWavefront else None
        self.aging = None

    def compute_received_power(self, h, g, phases, **kwargs):
        if self.unit_cell:
            gamma = self.unit_cell.compute_reflection(phases)
        else:
            gamma = np.exp(1j * phases)

        cascaded = h.conjugate() * g
        signal = np.dot(gamma, cascaded)
        return np.abs(signal)**2

    def get_metadata(self) -> Dict[str, Any]:
        """Required by PhysicsModel interface. FIXED: Was missing."""
        return {
            'name': 'IdealPhysicsModel',
            'description': 'Baseline ideal physics'
        }

# --- 3. PHYSICS ENGINE (Adapter) ---
class PhysicsEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.profile = config.get('realism_profile', 'ideal')

        # Build the model on init
        self._model = self._build_model()
        logger.info(f"PhysicsEngine initialized. Profile: {self.profile}")

    def _build_model(self):
        # 1. IDEAL CASE
        if self.profile == 'ideal':
            return IdealPhysicsModel()

        # 2. REALISTIC CASE

        # Unit Cell (Signature: coupling_strength)
        uc_coupling = self.config.get('varactor_coupling_strength', 0.1)

        if VaractorUnitCell:
            # FIXED: Passing coupling_strength, not bits
            unit_cell = VaractorUnitCell(coupling_strength=uc_coupling)
        else:
            unit_cell = None

        # Coupling (Signature: coupling_strength)
        if GeometricCoupling:
            # FIXED: Passing coupling_strength, not strength
            coupling = GeometricCoupling(coupling_strength=uc_coupling) if uc_coupling > 0 else NoCoupling()
        else:
             coupling = None

        # Aging (Signature: doppler_hz)
        doppler = self.config.get('doppler_hz', 0.0)
        aging = None
        if doppler > 0 and JakesAging:
            aging = JakesAging(doppler_hz=doppler)

        return RealisticPhysicsModel(
            unit_cell=unit_cell,
            coupling=coupling,
            wavefront=PlanarWavefront() if PlanarWavefront else None,
            aging=aging
        )

    def compute_received_power(self, h, g, probe_phases):
        return self._model.compute_received_power(h, g, probe_phases)