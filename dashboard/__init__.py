"""
RIS PhD Ultimate Dashboard - Minimalist Research Platform.

A fully customizable Jupyter notebook dashboard for RIS probe-based ML research.
Optimized for unified workflow, stacking, and tabbed analysis.
"""

__version__ = "1.2.0"
__author__ = "RIS Research Team"

# Import components from widgets.py
from dashboard.widgets import (
    get_all_widgets,
    create_tab_layout,
    create_unified_dashboard,
    create_results_area,

    # Action Buttons
    button_run_experiment,
    button_save_config,
    button_load_config,
    button_reset_defaults,

    # Stack Buttons
    button_add_to_stack,
    button_clear_stack,
    button_run_stack,

    # Export Buttons
    button_export_csv,
    button_export_json,
    button_export_latex,
    button_save_model,

    # Status Widgets
    widget_status_output,
    widget_progress_bar,
    widget_live_metrics,

    # Results Widgets (UPDATED for Tabbed Layout)
    widget_results_summary,
    widget_results_plots_training, # <--- New Name
    widget_results_plots_analysis, # <--- New Name
)

# Import components from callbacks.py
from dashboard.callbacks import (
    setup_all_callbacks,
    setup_experiment_handlers,
    reset_to_defaults,
    update_param_count_preview,
)

# Core Logic Imports
from dashboard.validators import (
    get_validation_errors,
)

from dashboard.config_manager import (
    config_to_dict,
    save_config,
    load_config,
)

from dashboard.experiment_runner import (
    run_single_experiment,
    run_multi_model_comparison,
    run_multi_seed_experiment,
    aggregate_results,
    ExperimentResults,
)

from dashboard.plots import EXTENDED_PLOT_REGISTRY

__all__ = [
    # Version info
    '__version__',
    '__author__',

    # Widgets & Layouts
    'get_all_widgets',
    'create_tab_layout',
    'create_unified_dashboard',
    'create_results_area',

    # Buttons
    'button_run_experiment',
    'button_save_config',
    'button_load_config',
    'button_reset_defaults',
    'button_add_to_stack',
    'button_clear_stack',
    'button_run_stack',
    'button_export_csv',
    'button_export_json',
    'button_export_latex',
    'button_save_model',

    # Status & Results
    'widget_status_output',
    'widget_progress_bar',
    'widget_live_metrics',
    'widget_results_summary',
    'widget_results_plots_training',
    'widget_results_plots_analysis',

    # Callbacks & Handlers
    'setup_all_callbacks',
    'setup_experiment_handlers',
    'reset_to_defaults',
    'update_param_count_preview',

    # Core Logic
    'get_validation_errors',
    'config_to_dict',
    'save_config',
    'load_config',
    'run_single_experiment',
    'run_multi_model_comparison',
    'run_multi_seed_experiment',
    'aggregate_results',
    'ExperimentResults',
    'EXTENDED_PLOT_REGISTRY',
]

def print_welcome():
    """Print welcome message with system info."""
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║           RIS PhD Ultimate Research Dashboard (Stacking Ed.)         ║")
    print("║           Version: {}                                            ║".format(__version__))
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print("New Features:")
    print("  ✓ Experiment Stacking & Queueing")
    print("  ✓ Transfer Learning (Model Chaining)")
    print("  ✓ Categorized Probes (Physics vs. Math)")
    print("  ✓ Tabbed Results & Decoupled Plotting")
    print()