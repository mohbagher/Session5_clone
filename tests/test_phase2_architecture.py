"""
Test Phase 2: Composable Physics Architecture
==============================================
Basic validation tests for new architecture.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_unit_cell_models():
    """Test unit cell models."""
    print("\n[TEST] Unit Cell Models")
    
    from src.ris_platform.physics.components.unit_cell import IdealUnitCell, VaractorUnitCell
    
    # Test ideal unit cell
    ideal = IdealUnitCell()
    phases = np.array([0, np.pi/4, np.pi/2, np.pi])
    gamma = ideal.compute_reflection(phases)
    
    assert np.allclose(np.abs(gamma), 1.0), "Ideal cell should have |Γ| = 1"
    print("  ✓ IdealUnitCell: |Γ| = 1 verified")
    
    # Test varactor unit cell
    varactor = VaractorUnitCell(coupling_strength=0.3)
    gamma_v = varactor.compute_reflection(phases)
    
    # Amplitude should vary with phase
    amplitudes = np.abs(gamma_v)
    assert np.std(amplitudes) > 0, "Varactor should have amplitude variation"
    print(f"  ✓ VaractorUnitCell: amplitude range [{amplitudes.min():.3f}, {amplitudes.max():.3f}]")
    
    # Test thermal drift
    gamma_cold = varactor.compute_reflection(phases, temperature=10.0)
    gamma_hot = varactor.compute_reflection(phases, temperature=40.0)
    phase_shift = np.angle(gamma_hot) - np.angle(gamma_cold)
    assert np.any(np.abs(phase_shift) > 0.1), "Should have thermal drift"
    print(f"  ✓ VaractorUnitCell: thermal drift verified (Δφ = {np.abs(phase_shift[0]):.3f} rad)")


def test_coupling_models():
    """Test coupling models."""
    print("\n[TEST] Coupling Models")
    
    from src.ris_platform.physics.components.coupling import NoCoupling, GeometricCoupling
    
    # Test no coupling
    no_coupling = NoCoupling()
    gamma = np.exp(1j * np.random.rand(10))
    gamma_coupled = no_coupling.apply_coupling(gamma)
    assert np.allclose(gamma_coupled, gamma), "No coupling should be identity"
    print("  ✓ NoCoupling: identity verified")
    
    # Test geometric coupling
    geometric = GeometricCoupling(coupling_strength=0.5)
    N = 8
    positions = np.column_stack([np.arange(N) * 0.0625, np.zeros(N)])
    geometry = {'positions': positions}
    
    C = geometric.get_coupling_matrix(geometry)
    assert C.shape == (N, N), "Coupling matrix should be N x N"
    assert np.allclose(np.diag(C), 1.0), "Diagonal should be 1"
    assert not np.allclose(C, np.eye(N)), "Should have off-diagonal coupling"
    print(f"  ✓ GeometricCoupling: matrix shape {C.shape}, max off-diag = {np.abs(C - np.eye(N)).max():.3f}")


def test_wavefront_models():
    """Test wavefront models."""
    print("\n[TEST] Wavefront Models")
    
    from src.ris_platform.physics.components.wavefront import PlanarWavefront, SphericalWavefront
    
    N = 16
    h = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
    g = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
    gamma = np.exp(1j * np.random.rand(N) * 2 * np.pi)
    
    # Test planar
    planar = PlanarWavefront()
    signal_planar = planar.compute_signal(h, g, gamma)
    assert np.isscalar(signal_planar) or signal_planar.size == 1, "Should return scalar"
    print(f"  ✓ PlanarWavefront: signal power = {np.abs(signal_planar)**2:.3f}")
    
    # Test spherical
    spherical = SphericalWavefront()
    signal_spherical = spherical.compute_signal(h, g, gamma)
    print(f"  ✓ SphericalWavefront: signal power = {np.abs(signal_spherical)**2:.3f}")


def test_channel_aging():
    """Test channel aging model."""
    print("\n[TEST] Channel Aging")
    
    from src.ris_platform.physics.components.aging import JakesAging
    
    N = 16
    h = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
    g = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
    
    jakes = JakesAging(doppler_hz=10.0)
    
    # Test aging at different time deltas
    h_aged_short, g_aged_short = jakes.age_channel(h, g, time_delta=0.001)
    h_aged_long, g_aged_long = jakes.age_channel(h, g, time_delta=0.1)
    
    # Short time: high correlation
    corr_short = np.abs(np.vdot(h, h_aged_short)) / (np.linalg.norm(h) * np.linalg.norm(h_aged_short))
    corr_long = np.abs(np.vdot(h, h_aged_long)) / (np.linalg.norm(h) * np.linalg.norm(h_aged_long))
    
    assert corr_short > corr_long, "Correlation should decrease with time"
    print(f"  ✓ JakesAging: ρ(0.001s) = {corr_short:.3f}, ρ(0.1s) = {corr_long:.3f}")
    
    coherence_time = jakes.get_coherence_time()
    print(f"  ✓ JakesAging: coherence time = {coherence_time:.3f} s")


def test_physics_models():
    """Test high-level physics models."""
    print("\n[TEST] Physics Models")
    
    from src.ris_platform.physics.models import IdealPhysicsModel, RealisticPhysicsModel
    from src.ris_platform.physics.components.unit_cell import VaractorUnitCell
    from src.ris_platform.physics.components.coupling import GeometricCoupling
    
    N = 16
    h = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
    g = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
    phases = np.random.rand(N) * 2 * np.pi
    
    # Test ideal model
    ideal = IdealPhysicsModel()
    power_ideal = ideal.compute_received_power(h, g, phases)
    assert power_ideal > 0, "Power should be positive"
    print(f"  ✓ IdealPhysicsModel: power = {power_ideal:.3f}")
    
    # Test realistic model with components
    realistic = RealisticPhysicsModel(
        unit_cell=VaractorUnitCell(coupling_strength=0.3),
        coupling=GeometricCoupling(coupling_strength=0.5)
    )
    power_realistic = realistic.compute_received_power(h, g, phases)
    assert power_realistic > 0, "Power should be positive"
    print(f"  ✓ RealisticPhysicsModel: power = {power_realistic:.3f}")
    
    # Metadata
    metadata = realistic.get_metadata()
    assert 'unit_cell' in metadata, "Should have unit_cell in metadata"
    print(f"  ✓ RealisticPhysicsModel: metadata = {list(metadata.keys())}")


def test_backends():
    """Test channel backends."""
    print("\n[TEST] Channel Backends")
    
    from src.ris_platform.backend.python_synthetic import PythonSyntheticBackend
    from src.ris_platform.backend.matlab import MATLABBackend
    
    # Test Python backend
    python_backend = PythonSyntheticBackend()
    assert python_backend.is_available(), "Python backend should always be available"
    
    h, g, metadata = python_backend.generate_channels(N=16, K=1, num_samples=10, seed=42)
    assert h.shape == (16, 10), f"Expected (16, 10), got {h.shape}"
    assert g.shape == (16, 10), f"Expected (16, 10), got {g.shape}"
    assert metadata['backend'] == 'python_synthetic'
    print(f"  ✓ PythonSyntheticBackend: generated {h.shape}")
    
    # Test MATLAB backend (may not be available)
    matlab_backend = MATLABBackend(auto_fallback=True)
    info = matlab_backend.get_backend_info()
    print(f"  ✓ MATLABBackend: available = {matlab_backend.is_available()}")


def test_probing_strategies():
    """Test probing strategies."""
    print("\n[TEST] Probing Strategies")
    
    from src.ris_platform.probing.structured import RandomProbing, SobolProbing, HadamardProbing
    
    K, M = 64, 16
    
    # Test random probing
    random = RandomProbing(seed=42)
    indices = random.select_probes(K, M)
    assert len(indices) == M, f"Should select M={M} probes"
    assert len(np.unique(indices)) == M, "Indices should be unique"
    print(f"  ✓ RandomProbing: selected {len(indices)} unique probes")
    
    # Test Sobol probing
    sobol = SobolProbing(seed=42)
    indices_sobol = sobol.select_probes(K, M)
    assert len(indices_sobol) == M
    print(f"  ✓ SobolProbing: selected {len(indices_sobol)} probes")
    
    # Test Hadamard probing
    hadamard = HadamardProbing(seed=42)
    indices_hadamard = hadamard.select_probes(K, M)
    assert len(indices_hadamard) == M
    print(f"  ✓ HadamardProbing: selected {len(indices_hadamard)} probes")


def test_factory_functions():
    """Test factory functions for backward compatibility."""
    print("\n[TEST] Factory Functions")
    
    # Import directly from file to avoid dashboard dependencies
    import sys
    import importlib.util
    
    spec = importlib.util.spec_from_file_location(
        "experiment_runner",
        "dashboard/experiment_runner.py"
    )
    experiment_runner = importlib.util.module_from_spec(spec)
    sys.modules["experiment_runner"] = experiment_runner
    spec.loader.exec_module(experiment_runner)
    
    create_physics_model = experiment_runner.create_physics_model
    create_backend = experiment_runner.create_backend
    create_probing_strategy = experiment_runner.create_probing_strategy
    
    # Test ideal config
    config_ideal = {'realism_profile': 'ideal'}
    physics = create_physics_model(config_ideal)
    assert physics is not None
    print(f"  ✓ create_physics_model (ideal): {type(physics).__name__}")
    
    # Test realistic config
    config_realistic = {
        'realism_profile': 'realistic',
        'varactor_coupling_strength': 0.3,
        'coupling_strength': 0.5,
        'enable_near_field': True,
        'enable_aging': True,
        'doppler_hz': 10.0
    }
    physics_realistic = create_physics_model(config_realistic)
    assert physics_realistic is not None
    print(f"  ✓ create_physics_model (realistic): {type(physics_realistic).__name__}")
    
    # Test backend creation
    backend = create_backend({'physics_backend': 'python'})
    assert backend is not None
    print(f"  ✓ create_backend: {type(backend).__name__}")
    
    # Test probing strategy
    strategy = create_probing_strategy({'probing_strategy': 'sobol', 'seed': 42})
    assert strategy is not None
    print(f"  ✓ create_probing_strategy: {type(strategy).__name__}")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("PHASE 2 ARCHITECTURE VALIDATION TESTS")
    print("=" * 70)
    
    try:
        test_unit_cell_models()
        test_coupling_models()
        test_wavefront_models()
        test_channel_aging()
        test_physics_models()
        test_backends()
        test_probing_strategies()
        test_factory_functions()
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
