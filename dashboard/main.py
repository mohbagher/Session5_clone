"""
Main Dashboard Orchestrator - Professional Visual Edition
========================================================
Refined layout with compact panels, gray-style buttons, and full logic integration.
"""

import ipywidgets as widgets
from IPython.display import display, HTML

# Import Tabs
from dashboard.tabs import (
    create_system_tab,
    create_physics_tab,
    create_model_tab,
    create_training_tab,
    create_evaluation_tab,
    create_visualization_tab,
    tab_ablation
)

# Import Components
from dashboard.components import (
    create_stack_manager,
    create_action_buttons,
    create_results_buttons,
    create_export_buttons,
    create_status_display,
    create_results_display
)

# Import Callbacks
from dashboard.callbacks import setup_all_callbacks, setup_experiment_handlers

def create_panel(title, content, color="#2C3E50"):
    """Wraps a widget in a stylish, distinctive box."""
    header = widgets.HTML(
        value=f"<div style='background-color: {color}; color: white; padding: 8px 15px; border-radius: 4px 4px 0 0; font-weight: bold; font-family: sans-serif;'>{title}</div>"
    )

    body = widgets.VBox([content], layout=widgets.Layout(
        padding='15px',
        border=f'1px solid {color}',
        border_top='none',
        border_radius='0 0 4px 4px',
        width='100%',
        background_color='#ffffff'
    ))

    return widgets.VBox([header, body], layout=widgets.Layout(margin='10px 0', width='100%'))

def create_unified_dashboard():
    # --- Create Tabs ---
    tab_system = create_system_tab()
    tab_physics = create_physics_tab()
    tab_model = create_model_tab()
    tab_training = create_training_tab()
    tab_evaluation = create_evaluation_tab()
    tab_visualization = create_visualization_tab()

    try:
        tab_ablation_widget = tab_ablation.render({})
    except:
        tab_ablation_widget = widgets.HTML("Ablation tab unavailable")

    tabs = widgets.Tab(children=[
        tab_system, tab_physics, tab_model, tab_training,
        tab_evaluation, tab_visualization, tab_ablation_widget
    ])

    titles = ['System', 'Physics', 'Model', 'Training', 'Metrics', 'Visualization', 'Ablation']
    for i, title in enumerate(titles): tabs.set_title(i, title)

    # --- Create Components ---
    stack_manager = create_stack_manager()
    action_buttons = create_action_buttons()
    results_buttons = create_results_buttons()
    export_buttons = create_export_buttons()
    status_display = create_status_display()

    # --- Collect Widgets ---
    widget_dict = {}
    for t in [tab_system, tab_physics, tab_model, tab_training, tab_evaluation, tab_visualization]:
        if hasattr(t, '_widgets'): widget_dict.update(t._widgets)
    for c in [stack_manager, action_buttons, results_buttons, export_buttons, status_display]:
        if hasattr(c, '_widgets'): widget_dict.update(c._widgets)

    # --- WIRE UP CALLBACKS ---
    try:
        setup_all_callbacks(widget_dict)
        setup_experiment_handlers(widget_dict)
    except Exception as e:
        print(f"Warning: Could not setup callbacks: {e}")

    # --- Assemble Layout ---
    config_panel = create_panel("1. CONFIGURATION", tabs, color="#1565C0")
    stack_panel = create_panel("2. EXPERIMENT STACK", stack_manager, color="#6A1B9A")

    cmd_content = widgets.VBox([
        widgets.HTML("<b>Actions:</b>"),
        action_buttons,
        widgets.HTML("<hr style='margin: 8px 0'><b>Results:</b>"),
        widgets.HBox([results_buttons, export_buttons]),
        widgets.HTML("<hr style='margin: 8px 0'>"),
        status_display
    ])
    cmd_panel = create_panel("3. CONTROL PANEL", cmd_content, color="#2E7D32")

    dashboard_ui = widgets.VBox([config_panel, stack_panel, cmd_panel])

    return dashboard_ui, widget_dict

def create_complete_interface():
    dashboard_ui, widget_dict = create_unified_dashboard()
    results_display = create_results_display()

    if hasattr(results_display, '_widgets'):
        widget_dict.update(results_display._widgets)

    results_panel = create_panel("4. ANALYSIS & PLOTS", results_display, color="#C62828")

    # --- CSS STYLING (Gray Buttons Included) ---
    style_html = widgets.HTML("""
        <style>
            .widget-tab-contents { padding: 10px; border: 1px solid #ddd; }
            .widget-tab > .p-TabBar .p-TabBar-tab { min-width: 100px; font-weight: bold; background: #eee; }
            .widget-tab > .p-TabBar .p-TabBar-tab.p-mod-current { background: #1565C0; color: white; }
            
            /* Stylish Gray Toggle Buttons */
            .widget-toggle-buttons .widget-toggle-button {
                background-color: #f5f5f5; 
                color: #424242; 
                font-weight: normal;
                border: 1px solid #bdbdbd;
                opacity: 1.0;
            }
            .widget-toggle-buttons .widget-toggle-button.p-mod-active {
                background-color: #546E7A !important; 
                color: white !important;
                font-weight: bold;
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
            }
        </style>
    """)

    title_html = widgets.HTML("<h2 style='text-align: center; color: #333;'>RIS Platform - Phase 2</h2>")

    full_app = widgets.VBox(
        [style_html, title_html, dashboard_ui, widgets.HTML("<br>"), results_panel],
        layout=widgets.Layout(width='98%', padding='10px')
    )

    return full_app, widget_dict

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_widget_values(widget_dict):
    """Safe extraction of values for config."""
    config = {}
    # Key parameters only
    keys = [
        'N', 'K', 'M', 'realism_profile', 'physics_backend',
        'P_tx', 'sigma_h_sq', 'sigma_g_sq',
        'probe_type', 'probe_category', 'model_preset'
    ]
    for k in keys:
        if k in widget_dict: config[k] = widget_dict[k].value
    return config

def print_dashboard_info():
    print("RIS DASHBOARD - PHASE 2 PROFESSIONAL EDITION")

def render_dashboard(config=None):
    """Renders the dashboard and returns the UI object."""
    ui, _ = create_complete_interface()
    if config:
        print(f"⚙️ Config Applied: {config.get('realism_profile', 'default')}")
    return ui