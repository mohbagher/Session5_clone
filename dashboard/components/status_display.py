"""
Status Display Component
========================
Progress bars, live metrics, and log output.
"""

import ipywidgets as widgets


def create_status_display():
    """Create status display component with progress and logs."""

    # Progress bar
    widget_progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='Progress:',
        bar_style='info',
        layout=widgets.Layout(width='100%')
    )

    # Live metrics display
    widget_live_metrics = widgets.HTML(
        value="<div style='font-family: monospace; padding: 10px;'><b>Waiting to start...</b></div>",
        layout=widgets.Layout(
            width='100%',
            height='150px',
            border='1px solid #ccc',
            padding='10px',
            background_color='#fafafa'
        )
    )

    # Status output (log)
    widget_status_output = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            height='200px',
            border='1px solid #ccc',
            padding='10px',
            overflow='auto',
            background_color='#ffffff'
        )
    )

    # Layout
    status_layout = widgets.VBox([
        widgets.Label("Training Progress:"),
        widget_progress_bar,
        widgets.Label("Live Metrics:"),
        widget_live_metrics,
        widgets.HTML("<hr style='margin: 10px 0;'>"),
        widgets.Label("Execution Log:"),
        widget_status_output
    ], layout=widgets.Layout(
        border='1px solid #ddd',
        padding='15px',
        background_color='#f9f9f9',
        margin='10px 0'
    ))

    # Store references
    status_layout._widgets = {
        'progress_bar': widget_progress_bar,
        'live_metrics': widget_live_metrics,
        'status_output': widget_status_output
    }

    return status_layout
