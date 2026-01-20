"""
Action Buttons Component
========================
Main action buttons for running experiments, saving/loading configs, etc.
"""

import ipywidgets as widgets


def create_action_buttons():
    """Create main action buttons."""

    button_run_experiment = widgets.Button(
        description='RUN SINGLE',
        button_style='success',
        layout=widgets.Layout(width='150px'),
        icon='play',
        tooltip='Run single experiment with current configuration'
    )

    button_save_config = widgets.Button(
        description='SAVE CONFIG',
        button_style='info',
        layout=widgets.Layout(width='150px'),
        icon='save',
        tooltip='Save current configuration to file'
    )

    button_load_config = widgets.Button(
        description='LOAD CONFIG',
        button_style='warning',
        layout=widgets.Layout(width='150px'),
        icon='folder-open',
        tooltip='Load configuration from file'
    )

    button_reset_defaults = widgets.Button(
        description='RESET',
        button_style='danger',
        layout=widgets.Layout(width='150px'),
        icon='refresh',
        tooltip='Reset all parameters to defaults'
    )

    # Layout
    buttons_layout = widgets.HBox([
        button_run_experiment,
        button_save_config,
        button_load_config,
        button_reset_defaults
    ], layout=widgets.Layout(
        justify_content='space-around',
        padding='10px'
    ))

    # Store references
    buttons_layout._widgets = {
        'button_run_experiment': button_run_experiment,
        'button_save_config': button_save_config,
        'button_load_config': button_load_config,
        'button_reset_defaults': button_reset_defaults
    }

    return buttons_layout


def create_results_buttons():
    """Create results management buttons."""

    button_plot_only = widgets.Button(
        description='PLOT ONLY',
        button_style='info',
        layout=widgets.Layout(width='150px'),
        icon='chart-bar',
        tooltip='Generate plots without training'
    )

    button_load_results = widgets.Button(
        description='LOAD RESULTS',
        button_style='warning',
        layout=widgets.Layout(width='150px'),
        icon='folder-open',
        tooltip='Load previously saved results'
    )

    button_save_results = widgets.Button(
        description='SAVE RESULTS',
        button_style='success',
        layout=widgets.Layout(width='150px'),
        icon='save',
        tooltip='Save current results to file'
    )

    # Layout
    buttons_layout = widgets.HBox([
        button_plot_only,
        button_load_results,
        button_save_results
    ], layout=widgets.Layout(
        justify_content='center',
        padding='10px'
    ))

    # Store references
    buttons_layout._widgets = {
        'button_plot_only': button_plot_only,
        'button_load_results': button_load_results,
        'button_save_results': button_save_results
    }

    return buttons_layout


def create_export_buttons():
    """Create export buttons for results."""

    button_export_csv = widgets.Button(
        description='Export CSV',
        button_style='primary',
        layout=widgets.Layout(width='150px'),
        icon='table',
        tooltip='Export results as CSV'
    )

    button_export_json = widgets.Button(
        description='Export JSON',
        button_style='primary',
        layout=widgets.Layout(width='150px'),
        icon='code',
        tooltip='Export results as JSON'
    )

    button_export_latex = widgets.Button(
        description='Export LaTeX',
        button_style='primary',
        layout=widgets.Layout(width='150px'),
        icon='file-text',
        tooltip='Export results as LaTeX table'
    )

    button_save_model = widgets.Button(
        description='Save Model (.pt)',
        button_style='warning',
        layout=widgets.Layout(width='150px'),
        icon='save',
        tooltip='Save trained model weights'
    )

    # Layout
    buttons_layout = widgets.HBox([
        button_export_csv,
        button_export_json,
        button_export_latex,
        button_save_model
    ], layout=widgets.Layout(
        justify_content='center',
        padding='10px'
    ))

    # Store references
    buttons_layout._widgets = {
        'button_export_csv': button_export_csv,
        'button_export_json': button_export_json,
        'button_export_latex': button_export_latex,
        'button_save_model': button_save_model
    }

    return buttons_layout
