"""
Tab 2: Physics & Realism Configuration
=======================================
Channel sources, realism profiles, advanced impairment settings.
"""

import ipywidgets as widgets


def create_physics_tab():
    """Create Physics & Realism tab."""

    # Channel Source
    widget_channel_source = widgets.Dropdown(
        options=[
            ('Python Synthetic (Built-in)', 'python_synthetic'),
            ('MATLAB Engine (Phase 2)', 'matlab_engine'),
            ('MATLAB Verified Data (Phase 2)', 'matlab_verified')
        ],
        value='python_synthetic',
        description='Channel Source:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='600px'),
        tooltip='Select physics simulation backend'
    )

    # Source info display
    widget_source_info = widgets.HTML(
        value=(
            "<div style='margin-left: 190px; padding: 10px; background: #e3f2fd; border-left: 3px solid #2196f3;'>"
            "<b>Python Synthetic</b><br>"
            "Built-in numpy-based Rayleigh fading<br>"
            "[OK] Analytically verified<br>"
            "[OK] Fast and reliable"
            "</div>"
        ),
        layout=widgets.Layout(width='600px')
    )

    def update_source_info(change):
        source = change['new']

        if source == 'python_synthetic':
            info = (
                "<div style='margin-left: 190px; padding: 10px; background: #e3f2fd; border-left: 3px solid #2196f3;'>"
                "<b>Python Synthetic</b><br>"
                "Built-in numpy-based Rayleigh fading<br>"
                "[OK] Analytically verified<br>"
                "[OK] Fast and reliable"
                "</div>"
            )
        elif source == 'matlab_engine':
            info = (
                "<div style='margin-left: 190px; padding: 10px; background: #fff3e0; border-left: 3px solid #ff9800;'>"
                "<b>MATLAB Engine (Not Implemented)</b><br>"
                "Live MATLAB channel generation<br>"
                "[WARN] Requires MATLAB R2021b+<br>"
                "[WARN] Coming in Phase 2"
                "</div>"
            )
        else:
            info = (
                "<div style='margin-left: 190px; padding: 10px; background: #fff3e0; border-left: 3px solid #ff9800;'>"
                "<b>MATLAB Verified Data (Not Implemented)</b><br>"
                "Load pre-verified .mat files<br>"
                "[WARN] Requires pre-generated data<br>"
                "[WARN] Coming in Phase 2"
                "</div>"
            )

        widget_source_info.value = info

    widget_channel_source.observe(update_source_info, 'value')

    # Realism Profile
    widget_realism_profile = widgets.Dropdown(
        options=[
            ('Ideal (No Impairments)', 'ideal'),
            ('Mild Impairments (Lab)', 'mild_impairments'),
            ('Moderate Impairments (Indoor)', 'moderate_impairments'),
            ('Severe Impairments (Outdoor)', 'severe_impairments'),
            ('Worst Case (Stress Test)', 'worst_case')
        ],
        value='ideal',
        description='Realism Profile:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='600px'),
        tooltip='Select pre-configured impairment bundle'
    )

    # Profile descriptions
    profile_descriptions = {
        'ideal': (
            "<b>Ideal Conditions</b><br>"
            "• Perfect CSI<br>"
            "• Infinite precision hardware<br>"
            "• No environmental effects<br>"
            "<i>Use for: Theoretical upper bounds</i>"
        ),
        'mild_impairments': (
            "<b>Mild Impairments</b><br>"
            "• -30 dB CSI error (0.1%)<br>"
            "• 5 Hz Doppler, 10ms delay<br>"
            "• 6-bit phase shifters<br>"
            "<i>Use for: High-quality lab equipment</i>"
        ),
        'moderate_impairments': (
            "<b>Moderate Impairments</b><br>"
            "• -20 dB CSI error (1%)<br>"
            "• 10 Hz Doppler, 20ms delay<br>"
            "• 4-bit phase shifters<br>"
            "<i>Use for: Typical indoor deployment</i>"
        ),
        'severe_impairments': (
            "<b>Severe Impairments</b><br>"
            "• -15 dB CSI error (3%)<br>"
            "• 50 Hz Doppler, 50ms delay<br>"
            "• 3-bit phase shifters<br>"
            "<i>Use for: Outdoor/vehicular scenarios</i>"
        ),
        'worst_case': (
            "<b>Worst Case</b><br>"
            "• -10 dB CSI error (10%)<br>"
            "• 100 Hz Doppler, 100ms delay<br>"
            "• 2-bit phase shifters<br>"
            "<i>Use for: Robustness testing</i>"
        )
    }

    widget_profile_info = widgets.HTML(
        value=(
            "<div style='margin-left: 190px; padding: 10px; background: #e8f5e9; border-left: 3px solid #4caf50;'>"
            + profile_descriptions['ideal'] +
            "</div>"
        ),
        layout=widgets.Layout(width='600px')
    )

    def update_profile_info(change):
        profile = change['new']

        color_map = {
            'ideal': ('#e8f5e9', '#4caf50'),
            'mild_impairments': ('#fff9c4', '#fbc02d'),
            'moderate_impairments': ('#ffe0b2', '#f57c00'),
            'severe_impairments': ('#ffccbc', '#d84315'),
            'worst_case': ('#f3e5f5', '#7b1fa2')
        }

        bg_color, border_color = color_map.get(profile, ('#fff', '#999'))

        widget_profile_info.value = (
            f"<div style='margin-left: 190px; padding: 10px; background: {bg_color}; border-left: 3px solid {border_color};'>"
            + profile_descriptions.get(profile, "Unknown profile") +
            "</div>"
        )

    widget_realism_profile.observe(update_profile_info, 'value')

    # Advanced Impairments
    widget_use_custom_impairments = widgets.Checkbox(
        value=False,
        description='Advanced: Use Custom Impairments',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='600px', margin='20px 0 10px 0')
    )

    widget_csi_error_db = widgets.FloatSlider(
        value=-20, min=-40, max=-5, step=1,
        description='CSI Error (dB):',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px'),
        disabled=True
    )

    widget_doppler_hz = widgets.FloatSlider(
        value=10, min=0, max=200, step=5,
        description='Doppler (Hz):',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px'),
        disabled=True
    )

    widget_phase_bits_hw = widgets.IntSlider(
        value=4, min=1, max=8, step=1,
        description='Phase Shifter Bits:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px'),
        disabled=True
    )

    widget_adc_bits = widgets.IntSlider(
        value=10, min=6, max=16, step=1,
        description='ADC Bits:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px'),
        disabled=True
    )

    advanced_warning = widgets.HTML(
        value=(
            "<div style='margin: 10px 0 10px 190px; padding: 10px; background: #ffebee; border-left: 3px solid #f44336;'>"
            "[WARNING] Modifying these settings will override the selected realism profile. "
            "Custom configurations are logged separately."
            "</div>"
        ),
        layout=widgets.Layout(width='600px')
    )

    advanced_container = widgets.VBox([
        advanced_warning,
        widget_csi_error_db,
        widget_doppler_hz,
        widget_phase_bits_hw,
        widget_adc_bits
    ])
    advanced_container.layout.display = 'none'

    def toggle_advanced(change):
        is_custom = change['new']
        advanced_container.layout.display = 'block' if is_custom else 'none'
        widget_csi_error_db.disabled = not is_custom
        widget_doppler_hz.disabled = not is_custom
        widget_phase_bits_hw.disabled = not is_custom
        widget_adc_bits.disabled = not is_custom

    widget_use_custom_impairments.observe(toggle_advanced, 'value')

    # Layout
    tab_layout = widgets.VBox([
        widgets.HTML("<h3>Physics & Realism Configuration</h3>"),
        widgets.HTML("<h4>Channel Source</h4>"),
        widget_channel_source,
        widget_source_info,
        widgets.HTML("<h4 style='margin-top: 20px;'>Realism Profile</h4>"),
        widget_realism_profile,
        widget_profile_info,
        widgets.HTML("<hr style='margin: 20px 0;'>"),
        widget_use_custom_impairments,
        advanced_container
    ], layout=widgets.Layout(padding='20px'))

    # Store widget references
    tab_layout._widgets = {
        'channel_source': widget_channel_source,
        'source_info': widget_source_info,
        'realism_profile': widget_realism_profile,
        'profile_info': widget_profile_info,
        'use_custom_impairments': widget_use_custom_impairments,
        'csi_error_db': widget_csi_error_db,
        'doppler_hz': widget_doppler_hz,
        'phase_bits_hw': widget_phase_bits_hw,
        'adc_bits': widget_adc_bits
    }

    return tab_layout
