"""
Tab 1: System Configuration (Rich Logic Edition)
===============================================
Restores M/K Ratio calculation and Probing Logic.
"""
import ipywidgets as widgets

def create_group_box(title, content_list, color="#546E7A"):
    header = widgets.HTML(f"<div style='background-color: {color}; color: white; padding: 4px 10px; font-size: 12px; font-weight: bold; border-radius: 4px 4px 0 0;'>{title}</div>")

    # Use GridBox if simple items, VBox if complex
    if len(content_list) > 1 and not isinstance(content_list[-1], widgets.HTML):
        body = widgets.GridBox(
            content_list,
            layout=widgets.Layout(grid_template_columns='repeat(2, 50%)', padding='10px', grid_gap='10px', border=f'1px solid {color}', border_radius='0 0 4px 4px', margin_bottom='10px')
        )
    else:
        body = widgets.VBox(content_list, layout=widgets.Layout(padding='10px', border=f'1px solid {color}', border_radius='0 0 4px 4px', margin_bottom='10px'))

    return widgets.VBox([header, body])

def create_system_tab():
    # --- 1. Geometry Section ---
    widget_N = widgets.IntSlider(value=32, min=4, max=256, description='N (Elements):', style={'description_width': '100px'}, layout=widgets.Layout(width='98%'))
    widget_K = widgets.IntSlider(value=64, min=4, max=128, description='K (Probes):', style={'description_width': '100px'}, layout=widgets.Layout(width='98%'))
    widget_M = widgets.IntSlider(value=8, min=1, max=64, description='M (Observed):', style={'description_width': '100px'}, layout=widgets.Layout(width='98%'))

    # RESTORED: M/K Ratio Indicator
    ratio_display = widgets.HTML(value="<div style='text-align: center; color: #666; font-size: 12px; margin-top: 5px;'><i>M/K Ratio: 12.5% (Sparse Sensing)</i></div>")

    box_geometry = create_group_box("📐 RIS GEOMETRY", [widget_N, widget_K, widget_M, ratio_display])

    # --- 2. Signal & Hardware Section ---
    widget_P_tx = widgets.FloatSlider(value=1.0, min=0.0, max=50.0, description='Tx Power:', style={'description_width': '100px'}, layout=widgets.Layout(width='98%'))
    widget_sigma_h = widgets.FloatSlider(value=1.0, min=0.1, max=10.0, description='σ_h²:', style={'description_width': '100px'}, layout=widgets.Layout(width='98%'))
    widget_sigma_g = widgets.FloatSlider(value=1.0, min=0.1, max=10.0, description='σ_g²:', style={'description_width': '100px'}, layout=widgets.Layout(width='98%'))

    box_signal = create_group_box("📡 SIGNAL & HARDWARE", [widget_P_tx, widget_sigma_h, widget_sigma_g])

    # --- 3. Probing Strategy Section ---
    widget_probe_cat = widgets.ToggleButtons(
        options=['Physics-Based', 'Mathematical'],
        value='Physics-Based',
        description='Category:',
        button_style='',
        style={'button_width': '100px'}
    )
    widget_probe_type = widgets.Dropdown(options=['continuous', 'binary', '2bit'], value='continuous', description='Pattern:', style={'description_width': '80px'}, layout=widgets.Layout(width='98%'))

    widget_phase_mode = widgets.ToggleButtons(
        options=['Continuous', 'Discrete'],
        value='Continuous',
        description='Phase:',
        button_style='',
        style={'button_width': '100px'}
    )
    widget_phase_bits = widgets.IntSlider(value=3, min=1, max=8, description='Bits:', style={'description_width': '80px'}, layout=widgets.Layout(width='98%'))

    box_probing = create_group_box("🎯 PROBING STRATEGY", [widget_probe_cat, widget_probe_type, widget_phase_mode, widget_phase_bits])

    # --- LOGIC: Updates & Validation ---

    def update_ratio(change=None):
        """Calculates M/K ratio and updates the display color."""
        M_val = widget_M.value
        K_val = widget_K.value

        # Validation: M cannot exceed K
        if M_val > K_val:
            M_val = K_val
            widget_M.value = K_val

        ratio = (M_val / K_val) * 100

        # Determine color based on sparsity
        if ratio < 10:
            color, label = "#d32f2f", "Very Sparse" # Red
        elif ratio < 25:
            color, label = "#f57c00", "Sparse"      # Orange
        elif ratio < 50:
            color, label = "#fbc02d", "Moderate"    # Yellow
        else:
            color, label = "#388e3c", "Dense"       # Green

        ratio_display.value = f"<div style='text-align: center; color: {color}; font-size: 12px; margin-top: 5px;'><b>M/K Ratio: {ratio:.1f}% ({label})</b></div>"

    widget_M.observe(update_ratio, names='value')
    widget_K.observe(update_ratio, names='value')

    def on_probe_cat_change(change):
        """Swaps dropdown options based on category."""
        if change['new'] == 'Physics-Based':
            widget_probe_type.options = ['continuous', 'binary', '2bit']
        else:
            widget_probe_type.options = ['sobol', 'hadamard', 'dft']
        widget_probe_type.value = widget_probe_type.options[0]

    widget_probe_cat.observe(on_probe_cat_change, names='value')

    def on_phase_mode_change(change):
        widget_phase_bits.disabled = (change['new'] == 'Continuous')

    widget_phase_mode.observe(on_phase_mode_change, names='value')

    # Init
    on_phase_mode_change({'new': widget_phase_mode.value})
    update_ratio()

    # --- ASSEMBLY ---
    tab_layout = widgets.VBox([box_geometry, box_signal, box_probing], layout=widgets.Layout(padding='10px'))

    # Store references
    tab_layout._widgets = {
        'N': widget_N, 'K': widget_K, 'M': widget_M,
        'P_tx': widget_P_tx, 'sigma_h_sq': widget_sigma_h, 'sigma_g_sq': widget_sigma_g,
        'probe_category': widget_probe_cat, 'probe_type': widget_probe_type,
        'phase_mode': widget_phase_mode, 'phase_bits': widget_phase_bits
    }

    return tab_layout