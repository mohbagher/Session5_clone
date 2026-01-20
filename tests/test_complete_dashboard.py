"""
Comprehensive Test Script for New Dashboard Architecture
=========================================================
Tests all components to ensure everything works correctly.
"""

def test_complete_dashboard():
    """Run complete test suite for new dashboard."""

    print("="*70)
    print("COMPREHENSIVE DASHBOARD TEST SUITE")
    print("="*70)
    print()

    # Test 1: Import all modules
    print("[TEST 1] Importing all modules...")
    try:
        from dashboard.tabs import (
            create_system_tab,
            create_physics_tab,
            create_model_tab,
            create_training_tab,
            create_evaluation_tab,
            create_visualization_tab
        )
        from dashboard.components import (
            create_stack_manager,
            create_action_buttons,
            create_status_display,
            create_results_display
        )
        from dashboard.main import (
            create_unified_dashboard,
            create_complete_interface,
            get_widget_values
        )
        print("[OK] All modules imported successfully")
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return False

    # Test 2: Create individual tabs
    print("\n[TEST 2] Creating individual tabs...")
    try:
        tab_system = create_system_tab()
        print("  [OK] System tab created")

        tab_physics = create_physics_tab()
        print("  [OK] Physics tab created")

        tab_model = create_model_tab()
        print("  [OK] Model tab created")

        tab_training = create_training_tab()
        print("  [OK] Training tab created")

        tab_evaluation = create_evaluation_tab()
        print("  [OK] Evaluation tab created")

        tab_visualization = create_visualization_tab()
        print("  [OK] Visualization tab created")
    except Exception as e:
        print(f"  [FAIL] Tab creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Create components
    print("\n[TEST 3] Creating components...")
    try:
        stack_manager = create_stack_manager()
        print("  [OK] Stack manager created")

        action_buttons = create_action_buttons()
        print("  [OK] Action buttons created")

        status_display = create_status_display()
        print("  [OK] Status display created")

        results_display = create_results_display()
        print("  [OK] Results display created")
    except Exception as e:
        print(f"  [FAIL] Component creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Create unified dashboard
    print("\n[TEST 4] Creating unified dashboard...")
    try:
        dashboard, widget_dict = create_unified_dashboard()
        print(f"  [OK] Dashboard created with {len(widget_dict)} widgets")
    except Exception as e:
        print(f"  [FAIL] Dashboard creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 5: Create complete interface
    print("\n[TEST 5] Creating complete interface...")
    try:
        ui, widgets = create_complete_interface()
        print(f"  [OK] Complete interface created")
    except Exception as e:
        print(f"  [FAIL] Interface creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 6: Extract configuration
    print("\n[TEST 6] Extracting configuration...")
    try:
        config = get_widget_values(widgets)
        print(f"  [OK] Configuration extracted")
        print(f"      N={config.get('N')}, K={config.get('K')}, M={config.get('M')}")
        print(f"      Channel source: {config.get('channel_source')}")
        print(f"      Realism profile: {config.get('realism_profile')}")
    except Exception as e:
        print(f"  [FAIL] Configuration extraction error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 7: Widget access
    print("\n[TEST 7] Testing widget access...")
    try:
        # Test system widgets
        assert 'N' in widgets, "N widget not found"
        assert 'K' in widgets, "K widget not found"
        assert 'M' in widgets, "M widget not found"

        # Test physics widgets
        assert 'channel_source' in widgets, "channel_source widget not found"
        assert 'realism_profile' in widgets, "realism_profile widget not found"

        # Test buttons
        assert 'button_run_experiment' in widgets, "Run button not found"
        assert 'button_add_to_stack' in widgets, "Add to stack button not found"

        print("  [OK] All key widgets accessible")
    except AssertionError as e:
        print(f"  [FAIL] Widget access error: {e}")
        return False

    # Test 8: Physics integration
    print("\n[TEST 8] Testing Phase 1 physics integration...")
    try:
        from physics import (
            list_available_sources,
            list_profiles,
            create_pipeline_from_profile
        )

        sources = list_available_sources()
        print(f"  [OK] {len(sources)} channel sources available")

        profiles = list_profiles()
        print(f"  [OK] {len(profiles)} realism profiles available")

        pipeline = create_pipeline_from_profile('moderate_impairments')
        print(f"  [OK] Impairment pipeline created")
    except Exception as e:
        print(f"  [FAIL] Physics integration error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("="*70)
    print("ALL TESTS PASSED!")
    print("="*70)
    print()
    print("The new dashboard architecture is ready to use.")
    print()
    print("To display the dashboard:")
    print("  from dashboard import create_complete_interface")
    print("  ui, widgets = create_complete_interface()")
    print("  display(ui)")

    return True

# Run test
if __name__ == "__main__":
    test_complete_dashboard()
