"""
Quick Phase 1 Test Script
Run this in a notebook cell to verify installation.
"""

def test_phase1():
    """Test Phase 1 installation."""
    print("="*70)
    print("PHASE 1 INSTALLATION TEST")
    print("="*70)

    # Test 1: Import physics module
    print("\n[TEST 1] Importing physics module...")
    try:
        from physics import (
            list_available_sources,
            list_profiles,
            create_pipeline_from_profile
        )
        print("[OK] Physics module imported successfully")
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return False

    # Test 2: Check channel sources
    print("\n[TEST 2] Checking channel sources...")
    try:
        sources = list_available_sources()
        print(f"[OK] Found {len(sources)} channel sources:")
        for name, info in sources.items():
            print(f"     - {name}: {info.get('description', 'N/A')}")
    except Exception as e:
        print(f"[FAIL] Error listing sources: {e}")
        return False

    # Test 3: Check profiles
    print("\n[TEST 3] Checking realism profiles...")
    try:
        profiles = list_profiles()
        print(f"[OK] Found {len(profiles)} realism profiles:")
        for name, info in profiles.items():
            print(f"     - {info['name']}")
    except Exception as e:
        print(f"[FAIL] Error listing profiles: {e}")
        return False

    # Test 4: Create pipeline
    print("\n[TEST 4] Creating impairment pipeline...")
    try:
        pipeline = create_pipeline_from_profile('moderate_impairments')
        config_summary = pipeline.get_configuration_summary()
        print("[OK] Pipeline created successfully")
        print(f"     Impairments configured: {len(config_summary['channel_impairments'])}")
    except Exception as e:
        print(f"[FAIL] Error creating pipeline: {e}")
        return False

    # Test 5: Generate test channel
    print("\n[TEST 5] Generating test channel with impairments...")
    try:
        import numpy as np
        from physics import generate_channel_realization_with_physics

        rng = np.random.RandomState(42)
        h, g, metadata = generate_channel_realization_with_physics(
            N=32,
            channel_source_type='python_synthetic',
            realism_profile='moderate_impairments',
            rng=rng
        )
        print(f"[OK] Channel generated: h.shape={h.shape}, g.shape={g.shape}")
        print(f"     Source: {metadata['channel_source']['source_type']}")
        print(f"     Profile: {metadata['realism_profile']}")
    except Exception as e:
        print(f"[FAIL] Error generating channel: {e}")
        return False

    print("\n" + "="*70)
    print("[SUCCESS] All Phase 1 tests passed!")
    print("="*70)
    print("\nPhase 1 is ready to use. See PHASE1_README.md for integration.")
    return True

# Run test if executed directly
if __name__ == "__main__":
    test_phase1()
