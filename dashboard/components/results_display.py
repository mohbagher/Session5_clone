"""
Results Display Component
=========================
Results summary, plots, and export options.
"""

import ipywidgets as widgets
from dashboard.components.buttons import create_export_buttons


def create_results_display():
    """Create results display area with tabs."""

    # Results summary HTML
    widget_results_summary = widgets.HTML(
        value="<div style='padding: 10px;'><i>No results yet. Run an experiment to see results.</i></div>",
        layout=widgets.Layout(width='100%', min_height='200px')
    )

    # Training plots output
    widget_results_plots_training = widgets.Output(
        layout=widgets.Layout(width='100%', padding='10px')
    )

    # Analysis plots output
    widget_results_plots_analysis = widgets.Output(
        layout=widgets.Layout(width='100%', padding='10px')
    )

    # Export buttons
    export_buttons = create_export_buttons()

    # Create tabs
    tab_summary = widgets.VBox([
        widget_results_summary,
        export_buttons
    ])

    tab_training = widgets.VBox([
        widget_results_plots_training
    ])

    tab_analysis = widgets.VBox([
        widget_results_plots_analysis
    ])

    results_tabs = widgets.Tab(children=[tab_summary, tab_training, tab_analysis])
    results_tabs.set_title(0, 'Summary & Export')
    results_tabs.set_title(1, 'Training Curves')
    results_tabs.set_title(2, 'Deep Analysis')

    # Main layout
    results_layout = widgets.VBox([
        widgets.HTML("<h2 style='text-align: center;'>Results & Analysis Dashboard</h2>"),
        results_tabs
    ], layout=widgets.Layout(
        padding='20px',
        border='2px solid #eee'
    ))

    # Store references
    results_layout._widgets = {
        'results_summary': widget_results_summary,
        'results_plots_training': widget_results_plots_training,
        'results_plots_analysis': widget_results_plots_analysis,
        'export_buttons': export_buttons._widgets
    }

    return results_layout
