"""
RIS PhD Ultimate Dashboard - Comprehensive Research Platform.

A fully customizable Jupyter notebook dashboard for RIS probe-based ML research.

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "RIS Research Team"

# Import main components
from dashboard.widgets import (
    get_all_widgets,
    create_tab_layout,
    button_run_experiment,
    button_save_config,
    button_load_config,
    button_reset_defaults,
    widget_status_output,
    widget_progress_bar,
    widget_live_metrics,
    widget_results_summary,
    widget_results_plots,
)

from dashboard.callbacks import (
    setup_all_callbacks,
    reset_to_defaults,
    update_param_count_preview,
)

from dashboard.validators import (
    get_validation_errors,
    validate_system_params,
    validate_model_config,
    validate_training_config,
    validate_eval_config,
    validate_visualization_config,
)

from dashboard.config_manager import (
    config_to_dict,
    dict_to_widgets,
    save_config,
    load_config,
    save_config_json,
    save_config_yaml,
    load_config_json,
    load_config_yaml,
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
    
    # Widgets
    'get_all_widgets',
    'create_tab_layout',
    'button_run_experiment',
    'button_save_config',
    'button_load_config',
    'button_reset_defaults',
    'widget_status_output',
    'widget_progress_bar',
    'widget_live_metrics',
    'widget_results_summary',
    'widget_results_plots',
    
    # Callbacks
    'setup_all_callbacks',
    'reset_to_defaults',
    'update_param_count_preview',
    
    # Validators
    'get_validation_errors',
    'validate_system_params',
    'validate_model_config',
    'validate_training_config',
    'validate_eval_config',
    'validate_visualization_config',
    
    # Config Manager
    'config_to_dict',
    'dict_to_widgets',
    'save_config',
    'load_config',
    'save_config_json',
    'save_config_yaml',
    'load_config_json',
    'load_config_yaml',
    
    # Experiment Runner
    'run_single_experiment',
    'run_multi_model_comparison',
    'run_multi_seed_experiment',
    'aggregate_results',
    'ExperimentResults',
    
    # Plots
    'EXTENDED_PLOT_REGISTRY',
]


def print_welcome():
    """Print welcome message with system info."""
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║           RIS PhD Ultimate Research Dashboard                        ║")
    print("║           Version: {}                                            ║".format(__version__))
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print("Available Features:")
    print("  ✓ 5 Customization Tabs (System, Model, Training, Eval, Viz)")
    print("  ✓ 19 Model Architectures")
    print("  ✓ 6 Probe Types")
    print("  ✓ 25+ Plot Types")
    print("  ✓ Multi-Model Comparison")
    print("  ✓ Multi-Seed Statistical Analysis")
    print("  ✓ Config Save/Load (JSON/YAML)")
    print()
