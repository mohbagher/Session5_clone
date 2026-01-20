"""
Tab 1: Core System Parameters
==============================
N, K, M, transmit power, channel variances, phase configuration, probe types.
"""

import ipywidgets as widgets


def create_system_tab():
    """Create Core System tab with essential parameters."""

    # RIS Configuration
    widget_N = widgets.IntSlider(
        value=32, min=4, max=256, step=1,
        description='N (RIS elements):',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px'),
        tooltip='Number of reconfigurable elements on the RIS'
    )

    widget_K = widgets.IntSlider(
        value=64, min=4, max=512, step=1,
        description='K (Codebook size):',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px'),
        tooltip='Total number of probe configurations in the bank'
    )

    widget_M = widgets.IntSlider(
        value=8, min=1, max=64, step=1,
        description='M (Sensing budget):',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px'),
        tooltip='Number of probes measured per channel realization'
    )

    # M/K ratio indicator
    ratio_display = widgets.HTML(
        value="<div style='margin-left: 190px; color: #666;'><i>M/K ratio: 12.5% (sparse sensing)</i></div>",
        layout=widgets.Layout(width='500px')
    )

    def update_ratio(change=None):
        M_val = widget_M.value
        K_val = widget_K.value
        ratio = M_val / K_val * 100

        if ratio < 10:
            color, label = "#d32f2f", "very sparse"
        elif ratio < 25:
            color, label = "#f57c00", "sparse"
        elif ratio < 50:
            color, label = "#fbc02d", "moderate"
        else:
            color, label = "#388e3c", "dense"

        ratio_display.value = (
            f"<div style='margin-left: 190px; color: {color};'>"
            f"<b>M/K ratio: {ratio:.1f}%</b> ({label} sensing)"
            f"</div>"
        )

    widget_M.observe(update_ratio, 'value')
    widget_K.observe(update_ratio, 'value')

    # Channel Physics
    widget_P_tx = widgets.FloatSlider(
        value=1.0, min=0.1, max=10.0, step=0.1,
        description='P_tx (Transmit power):',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px')
    )

    widget_sigma_h_sq = widgets.FloatSlider(
        value=1.0, min=0.1, max=10.0, step=0.1,
        description='σ_h² (BS-RIS variance):',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px')
    )

    widget_sigma_g_sq = widgets.FloatSlider(
        value=1.0, min=0.1, max=10.0, step=0.1,
        description='σ_g² (RIS-UE variance):',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px')
    )

    # Phase Configuration
    widget_phase_mode = widgets.Dropdown(
        options=['continuous', 'discrete'],
        value='continuous',
        description='Phase mode:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px')
    )

    widget_phase_bits = widgets.IntSlider(
        value=3, min=1, max=8, step=1,
        description='Phase bits:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px'),
        disabled=True
    )

    def toggle_phase_bits(change):
        widget_phase_bits.disabled = (change['new'] == 'continuous')

    widget_phase_mode.observe(toggle_phase_bits, 'value')

    # Probe Configuration
    widget_probe_category = widgets.Dropdown(
        options=['Physics-Based', 'Mathematical Sequence'],
        value='Physics-Based',
        description='Probe Category:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px')
    )

    widget_probe_type = widgets.Dropdown(
        options=['continuous', 'binary', '2bit'],
        value='continuous',
        description='Probe Type:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px')
    )

    def update_probe_options(change):
        if change['new'] == 'Physics-Based':
            widget_probe_type.options = ['continuous', 'binary', '2bit']
        else:
            widget_probe_type.options = ['hadamard', 'sobol', 'halton']
        widget_probe_type.value = widget_probe_type.options[0]

    widget_probe_category.observe(update_probe_options, 'value')

    # Layout
    tab_layout = widgets.VBox([
        widgets.HTML("<h3>Core System Parameters</h3>"),
        widgets.HTML("<h4>RIS Configuration</h4>"),
        widget_N,
        widget_K,
        widget_M,
        ratio_display,
        widgets.HTML("<h4 style='margin-top: 20px;'>Channel Physics</h4>"),
        widget_P_tx,
        widget_sigma_h_sq,
        widget_sigma_g_sq,
        widgets.HTML("<h4 style='margin-top: 20px;'>Phase Configuration</h4>"),
        widget_phase_mode,
        widget_phase_bits,
        widgets.HTML("<h4 style='margin-top: 20px;'>Probe Configuration</h4>"),
        widget_probe_category,
        widget_probe_type
    ], layout=widgets.Layout(padding='20px'))

    # Store widget references for external access
    tab_layout._widgets = {
        'N': widget_N,
        'K': widget_K,
        'M': widget_M,
        'P_tx': widget_P_tx,
        'sigma_h_sq': widget_sigma_h_sq,
        'sigma_g_sq': widget_sigma_g_sq,
        'phase_mode': widget_phase_mode,
        'phase_bits': widget_phase_bits,
        'probe_category': widget_probe_category,
        'probe_type': widget_probe_type
    }

    return tab_layout
