"""
Verify Phase 2 Imports
======================
Check that all Phase 2 modules can be imported correctly.
"""


def verify_imports():
    """Test all Phase 2 imports."""

    print("=" * 70)
    print("VERIFYING PHASE 2 IMPORTS")
    print("=" * 70)
    print()

    # Test 1: Physics module
    print("[1/6] Testing physics module...")
    try:
        from physics import (
            create_source_from_name,
            list_available_sources,
            get_source_info
        )
        print("  ✓ physics imports OK")

        # Test list_available_sources
        sources = list_available_sources()
        print(f"  ✓ Available sources: {sources}")

    except ImportError as e:
        print(f"  ✗ physics import failed: {e}")
        return False

    # Test 2: Create Python source
    print("\n[2/6] Testing Python source creation...")
    try:
        source = create_source_from_name('python_synthetic')
        print(f"  ✓ Created: {type(source).__name__}")
    except Exception as e:
        print(f"  ✗ Python source creation failed: {e}")
        return False

    # Test 3: MATLAB backend module (may fail if MATLAB not installed)
    print("\n[3/6] Testing MATLAB backend module...")
    try:
        from physics.matlab_backend import (
            get_session_manager,
            ToolboxManager,
            MATLABScriptGenerator
        )
        print("  ✓ MATLAB backend imports OK")
    except ImportError as e:
        print(f"  ⚠ MATLAB backend import failed (expected if MATLAB not installed): {e}")

    # Test 4: Config
    print("\n[4/6] Testing config module...")
    try:
        from config.system_config import SystemConfig, get_config

        config = get_config()
        print(f"  ✓ Config created")
        print(f"    - physics_backend: {config.system.physics_backend}")
        print(f"    - matlab_scenario: {config.system.matlab_scenario}")
    except Exception as e:
        print(f"  ✗ Config failed: {e}")
        return False

    # Test 5: Dashboard
    print("\n[5/6] Testing dashboard module...")
    try:
        from dashboard.main import create_complete_interface, get_widget_values
        print("  ✓ Dashboard imports OK")
    except ImportError as e:
        print(f"  ✗ Dashboard import failed: {e}")
        return False

    # Test 6: Experiment runner
    print("\n[6/6] Testing experiment runner...")
    try:
        from dashboard.experiment_runner import (
            run_single_experiment,
            generate_channels_python,
            generate_channels_matlab
        )
        print("  ✓ Experiment runner imports OK")
    except ImportError as e:
        print(f"  ✗ Experiment runner import failed: {e}")
        return False

    print()
    print("=" * 70)
    print("ALL IMPORTS VERIFIED ✓")
    print("=" * 70)
    print()
    print("Phase 2 is ready to use!")
    print()

    return True


if __name__ == "__main__":
    success = verify_imports()
    exit(0 if success else 1)