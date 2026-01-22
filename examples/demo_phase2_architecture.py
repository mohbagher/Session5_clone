"""
Phase 2 Architecture Demo
=========================
Demonstrates the new composable physics architecture.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Quick demo showing key features
def main():
    print("\n" + "="*70)
    print("PHASE 2: COMPOSABLE PHYSICS ARCHITECTURE DEMO")
    print("="*70)
    
    from src.ris_platform.physics.models import IdealPhysicsModel, RealisticPhysicsModel
    from src.ris_platform.physics.components.unit_cell import VaractorUnitCell
    from src.ris_platform.physics.components.coupling import GeometricCoupling
    
    # Demo: Create ideal model
    print("\n[1] Ideal Physics Model")
    ideal = IdealPhysicsModel()
    print(f"    Type: {type(ideal).__name__}")
    print(f"    Metadata: {ideal.get_metadata()['description']}")
    
    # Demo: Create realistic model with components
    print("\n[2] Realistic Physics Model (Composable)")
    realistic = RealisticPhysicsModel(
        unit_cell=VaractorUnitCell(coupling_strength=0.3),
        coupling=GeometricCoupling(coupling_strength=0.5)
    )
    print(f"    Type: {type(realistic).__name__}")
    print(f"    Unit Cell: {realistic.unit_cell.get_parameters()['type']}")
    print(f"    Coupling: {realistic.coupling.__class__.__name__}")
    
    # Demo: Compute power
    print("\n[3] Power Computation")
    N = 16
    h = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
    g = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
    phases = np.random.rand(N) * 2 * np.pi
    
    power_ideal = ideal.compute_received_power(h, g, phases)
    power_realistic = realistic.compute_received_power(h, g, phases)
    
    print(f"    Ideal power: {power_ideal:.4f}")
    print(f"    Realistic power: {power_realistic:.4f}")
    print(f"    Difference: {abs(power_ideal - power_realistic)/power_ideal * 100:.1f}%")
    
    print("\n" + "="*70)
    print("✓ Demo completed successfully!")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
