"""
Tab 6: Visualization Control
=============================
Plot selection, output format, styling options.
"""

import ipywidgets as widgets

def create_visualization_tab():
    """Create Visualization Control tab."""

    # Plot type selection
    ALL_PLOT_TYPES = [
        'training_curves', 'eta_distribution', 'cdf',
        'top_m_comparison', 'top_m_efficiency', 'baseline_comparison',
        'violin', 'box', 'scatter',
        'heatmap',
        'eta_vs_M', 'eta_vs_K', 'eta_vs_N'
    ]

    widget_selected_plots = widgets.SelectMultiple(
        options=ALL_PLOT_TYPES,
        value=['training_curves', 'eta_distribution', 'top_m_comparison', 'top_m_efficiency'],
        description='Select plots:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px', height='280px')
    )

    # Output settings
    widget_figure_format = widgets.Dropdown(
        options=['png', 'pdf', 'svg'],
        value='png',
        description='Figure format:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px')
    )

    widget_dpi = widgets.IntSlider(
        value=150, min=72, max=300, step=6,
        description='DPI:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px')
    )

    # Styling
    widget_color_palette = widgets.Dropdown(
        options=['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'seaborn'],
        value='viridis',
        description='Color palette:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px')
    )

    # Save options
    widget_save_plots = widgets.Checkbox(
        value=True,
        description='Save plots to disk',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px')
    )

    widget_output_dir = widgets.Text(
        value='results/',
        description='Output directory:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px')
    )

    # Layout - CLEANED: No buttons here anymore
    tab_layout = widgets.VBox([
        widgets.HTML("<h3>Visualization Configuration</h3>"),
        widgets.HTML("<p>Configure which plots to generate when you click 'Plot Only' or 'Run Stack' below.</p>"),

        widgets.HTML("<h4>Plot Selection</h4>"),
        widget_selected_plots,

        widgets.HTML("<h4 style='margin-top: 20px;'>Output Settings</h4>"),
        widget_figure_format,
        widget_dpi,

        widgets.HTML("<h4 style='margin-top: 20px;'>Styling</h4>"),
        widget_color_palette,

        widgets.HTML("<h4 style='margin-top: 20px;'>Save Options</h4>"),
        widget_save_plots,
        widget_output_dir
    ], layout=widgets.Layout(padding='20px'))

    # Store widget references
    tab_layout._widgets = {
        'selected_plots': widget_selected_plots,
        'figure_format': widget_figure_format,
        'dpi': widget_dpi,
        'color_palette': widget_color_palette,
        'save_plots': widget_save_plots,
        'output_dir': widget_output_dir
    }

    return tab_layout