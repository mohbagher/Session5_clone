"""
Main Dashboard Orchestrator
============================
Brings together all tabs, components, and creates the unified interface.
"""

import ipywidgets as widgets
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


def create_unified_dashboard():
    """
    Create the complete unified dashboard.

    Returns:
        tuple: (dashboard_widget, widget_dict)
            - dashboard_widget: The complete UI
            - widget_dict: Dictionary of all widgets for callbacks
    """

    # ========================================================================
    # Create Tabs
    # ========================================================================

    tab_system = create_system_tab()
    tab_physics = create_physics_tab()
    tab_model = create_model_tab()
    tab_training = create_training_tab()
    tab_evaluation = create_evaluation_tab()
    tab_visualization = create_visualization_tab()

    # Create tab container
    tabs = widgets.Tab(children=[
        tab_system,
        tab_physics,
        tab_model,
        tab_training,
        tab_evaluation,
        tab_visualization
    ])

    tabs.set_title(0, 'System')
    tabs.set_title(1, 'Physics & Realism')
    tabs.set_title(2, 'Model')
    tabs.set_title(3, 'Training')
    tabs.set_title(4, 'Evaluation')
    tabs.set_title(5, 'Visualization')

    # ========================================================================
    # Create Components (Below Tabs)
    # ========================================================================

    stack_manager = create_stack_manager()
    action_buttons = create_action_buttons()
    status_display = create_status_display()

    # ========================================================================
    # Assemble Dashboard
    # ========================================================================

    dashboard = widgets.VBox([
        tabs,                    # Parameter tabs at top
        stack_manager,           # Stack manager below tabs
        action_buttons,          # Main action buttons
        status_display           # Progress and logs
    ])

    # ========================================================================
    # Collect All Widgets into Dictionary
    # ========================================================================

    widget_dict = {}

    # Add tab widgets
    widget_dict.update(tab_system._widgets)
    widget_dict.update(tab_physics._widgets)
    widget_dict.update(tab_model._widgets)
    widget_dict.update(tab_training._widgets)
    widget_dict.update(tab_evaluation._widgets)
    widget_dict.update(tab_visualization._widgets)

    # Add component widgets
    widget_dict.update(stack_manager._widgets)
    widget_dict.update(action_buttons._widgets)
    widget_dict.update(status_display._widgets)

    return dashboard, widget_dict


def create_complete_interface():
    """
    Create complete interface with dashboard and results area.

    Returns:
        tuple: (complete_ui, widget_dict)
    """

    # Create dashboard
    dashboard, widget_dict = create_unified_dashboard()

    # Create results display
    results_display = create_results_display()

    # Add results widgets to dictionary
    widget_dict.update(results_display._widgets)

    # Combine dashboard and results
    complete_ui = widgets.VBox([
        widgets.HTML("<h1 style='text-align: center; color: #1976D2;'>"
                    "RIS Probe-Based Control - PhD Research Dashboard"
                    "</h1>"),
        widgets.HTML("<hr style='margin: 10px 0;'>"),
        dashboard,
        widgets.HTML("<hr style='margin: 20px 0;'>"),
        results_display
    ])

    return complete_ui, widget_dict


def get_widget_values(widget_dict):
    """
    Extract current values from all widgets.

    Args:
        widget_dict: Dictionary of widgets

    Returns:
        dict: Configuration dictionary with all current values
    """

    config = {}

    # System parameters
    config['N'] = widget_dict['N'].value
    config['K'] = widget_dict['K'].value
    config['M'] = widget_dict['M'].value
    config['P_tx'] = widget_dict['P_tx'].value
    config['sigma_h_sq'] = widget_dict['sigma_h_sq'].value
    config['sigma_g_sq'] = widget_dict['sigma_g_sq'].value
    config['phase_mode'] = widget_dict['phase_mode'].value
    config['phase_bits'] = widget_dict['phase_bits'].value
    config['probe_category'] = widget_dict['probe_category'].value
    config['probe_type'] = widget_dict['probe_type'].value

    # Physics & Realism (Phase 1)
    config['channel_source'] = widget_dict['channel_source'].value
    config['realism_profile'] = widget_dict['realism_profile'].value
    config['use_custom_impairments'] = widget_dict['use_custom_impairments'].value

    if config['use_custom_impairments']:
        config['custom_impairments_config'] = {
            'csi_error': {
                'enabled': True,
                'error_variance_db': widget_dict['csi_error_db'].value
            },
            'channel_aging': {
                'enabled': True,
                'doppler_hz': widget_dict['doppler_hz'].value,
                'feedback_delay_ms': 20  # Fixed for now
            },
            'phase_quantization': {
                'enabled': True,
                'phase_bits': widget_dict['phase_bits_hw'].value
            },
            'quantization': {
                'enabled': True,
                'adc_bits': widget_dict['adc_bits'].value
            },
            'amplitude_control': {'enabled': False},
            'mutual_coupling': {'enabled': False}
        }
    else:
        config['custom_impairments_config'] = None

    # Model parameters (placeholder - will be populated when tab is complete)
    # config['model_preset'] = widget_dict.get('model_preset', widgets.Widget()).value
    # config['num_layers'] = widget_dict.get('num_layers', widgets.Widget()).value
    # ... etc

    # Training parameters (placeholder)
    # config['n_train'] = widget_dict.get('n_train', widgets.Widget()).value
    # ... etc

    # Evaluation parameters (placeholder)
    # config['top_m_values'] = list(widget_dict.get('top_m_values', widgets.Widget()).value)
    # ... etc

    # Visualization parameters (placeholder)
    # config['selected_plots'] = list(widget_dict.get('selected_plots', widgets.Widget()).value)
    # ... etc

    return config


def print_dashboard_info():
    """Print information about the dashboard structure."""

    print("="*70)
    print("RIS DASHBOARD - CLEAN ARCHITECTURE")
    print("="*70)
    print()
    print("Tab Structure:")
    print("  1. System - Core parameters (N, K, M, probes)")
    print("  2. Physics & Realism - Channel sources and impairments")
    print("  3. Model - Architecture configuration")
    print("  4. Training - Training hyperparameters")
    print("  5. Evaluation - Metrics and comparison")
    print("  6. Visualization - Plot selection and settings")
    print()
    print("Components:")
    print("  - Stack Manager (below tabs)")
    print("  - Action Buttons (run, save, load)")
    print("  - Status Display (progress, metrics, logs)")
    print("  - Results Display (summary, plots, export)")
    print()
    print("="*70)
