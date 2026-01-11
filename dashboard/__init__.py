"""
RIS PhD Ultimate Dashboard - Minimalist Research Platform.

A fully customizable Jupyter notebook dashboard for RIS probe-based ML research.
Optimized for unified workflow and streamlined exports.
"""

__version__ = "1.1.0"
__author__ = "RIS Research Team"

# Import components from widgets.py (including new unified layout)
from dashboard.widgets import (
    get_all_widgets,
    create_tab_layout,
    create_unified_dashboard,
    create_results_area,
    button_run_experiment,
    button_save_config,
    button_load_config,
    button_reset_defaults,
    widget_status_output,
    widget_progress_bar,
    widget_live_metrics,
    widget_results_summary,
    widget_results_plots,
    button_export_csv,
    button_export_json,
    button_export_latex,
    button_save_model,
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
    'button_run_experiment',
    'button_save_config',
    'button_load_config',
    'button_reset_defaults',
    'widget_status_output',
    'widget_progress_bar',
    'widget_live_metrics',
    'widget_results_summary',
    'widget_results_plots',
    'button_export_csv',
    'button_export_json',
    'button_export_latex',
    'button_save_model',
    
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
    print("║           RIS PhD Ultimate Research Dashboard (Minimalist)           ║")
    print("║           Version: {}                                            ║".format(__version__))
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print("Available Features:")
    print("  ✓ Unified Dashboard (Tabs + Buttons + Progress in one view)")
    print("  ✓ Integrated Results Dashboard with Export Buttons")
    print("  ✓ Export Formats: CSV, JSON, LaTeX, and PyTorch Model (.pt)")
    print("  ✓ Multi-Model & Multi-Seed Analysis")
    print()