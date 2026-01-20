"""
Tab 2: Physics & Realism Configuration
=======================================
Channel sources, realism profiles, advanced impairment settings.

Phase 2 Extensions:
- MATLAB backend selection
- Toolbox scenario selector
- Advanced MATLAB parameters (CDL-RIS)
"""

import ipywidgets as widgets


def create_physics_tab():
    """Create Physics & Realism tab with Phase 2 MATLAB integration."""

    # ========================================================================
    # PHASE 1: Channel Source (kept for backward compatibility)
    # ========================================================================

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
                "<div style='margin-left: 190px; padding: 10px; background: #c8e6c9; border-left: 3px solid #4caf50;'>"
                "<b>MATLAB Engine (Phase 2 Active)</b><br>"
                "Live MATLAB channel generation<br>"
                "[OK] MathWorks verified toolboxes<br>"
                "[OK] Configure below in Backend Selection"
                "</div>"
            )
        else:
            info = (
                "<div style='margin-left: 190px; padding: 10px; background: #fff3e0; border-left: 3px solid #ff9800;'>"
                "<b>MATLAB Verified Data (Not Implemented)</b><br>"
                "Load pre-verified .mat files<br>"
                "[WARN] Requires pre-generated data<br>"
                "[WARN] Coming in Phase 2.2"
                "</div>"
            )

        widget_source_info.value = info

    widget_channel_source.observe(update_source_info, 'value')

    # ========================================================================
    # PHASE 2: BACKEND SELECTION (NEW)
    # ========================================================================

    widget_physics_backend = widgets.Dropdown(
        options=[
            ('Python (Default)', 'python'),
            ('MATLAB Engine (Verified Toolboxes)', 'matlab')
        ],
        value='python',
        description='Physics Backend:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='600px'),
        tooltip='Select computational backend for channel generation'
    )

    # MATLAB scenario selector
    widget_matlab_scenario = widgets.Dropdown(
        options=[
            ('Rayleigh Fading (Communications Toolbox)', 'rayleigh_basic'),
            ('CDL-RIS Channel (5G Toolbox)', 'cdl_ris'),
            ('Rician LOS (Communications Toolbox)', 'rician_los'),
            ('TDL Urban (5G Toolbox)', 'tdl_urban')
        ],
        value='rayleigh_basic',
        description='MATLAB Scenario:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='600px'),
        disabled=True,
        tooltip='Select verified MathWorks scenario'
    )

    # MATLAB status indicator
    widget_matlab_status = widgets.HTML(
        value="<div style='padding: 10px; background: #e0e0e0;'>"
              "<b>MATLAB Status:</b> Not selected (using Python)"
              "</div>",
        layout=widgets.Layout(width='600px', margin='10px 0')
    )

    # MATLAB toolbox info box
    widget_matlab_toolbox_info = widgets.HTML(
        value="",
        layout=widgets.Layout(width='600px', min_height='80px', margin='10px 0')
    )

    # ========================================================================
    # PHASE 2: Advanced MATLAB Parameters (CDL-RIS specific)
    # ========================================================================

    widget_carrier_frequency = widgets.FloatText(
        value=28e9,
        description='Carrier Freq (Hz):',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='600px'),
        disabled=True,
        tooltip='Carrier frequency for CDL channel'
    )

    widget_delay_profile = widgets.Dropdown(
        options=['CDL-A', 'CDL-B', 'CDL-C', 'CDL-D', 'CDL-E'],
        value='CDL-C',
        description='CDL Profile:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='600px'),
        disabled=True,
        tooltip='3GPP CDL delay profile'
    )

    widget_doppler_shift = widgets.FloatText(
        value=5,
        description='Doppler (Hz):',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='600px'),
        disabled=True,
        tooltip='Maximum Doppler shift'
    )

    # Advanced parameters container
    matlab_advanced_params = widgets.VBox([
        widgets.HTML("<b>CDL-RIS Parameters:</b> (only for cdl_ris scenario)"),
        widget_carrier_frequency,
        widget_delay_profile,
        widget_doppler_shift
    ], layout=widgets.Layout(
        border='1px solid #ddd',
        padding='10px',
        margin='10px 0',
        display='none'  # Hidden by default
    ))

    # ========================================================================
    # PHASE 2: Dynamic UI Updates
    # ========================================================================

    def update_backend_ui(change):
        """Update UI when backend selection changes."""
        backend = change['new']

        if backend == 'matlab':
            # Enable MATLAB controls
            widget_matlab_scenario.disabled = False

            # Check MATLAB status
            try:
                from physics.matlab_backend.session_manager import get_session_manager
                session_mgr = get_session_manager()

                if session_mgr.start_session():
                    # Success
                    widget_matlab_status.value = (
                        "<div style='padding: 10px; background: #c8e6c9;'>"
                        "<b>✓ MATLAB Status:</b> Connected and ready"
                        "</div>"
                    )

                    # Get toolbox info
                    from physics.matlab_backend.toolbox_registry import ToolboxManager
                    toolbox_mgr = ToolboxManager(session_mgr.get_engine())
                    available = toolbox_mgr.check_available_toolboxes()

                    toolbox_html = "<div style='padding: 10px; background: #e3f2fd;'>"
                    toolbox_html += "<b>Available Toolboxes:</b><br>"
                    for name, avail in available.items():
                        status = "✓" if avail else "✗"
                        color = "green" if avail else "red"
                        toolbox_html += f"<span style='color:{color};'>{status} {name.replace('_', ' ').title()}</span><br>"
                    toolbox_html += "</div>"

                    widget_matlab_toolbox_info.value = toolbox_html
                else:
                    # Failed
                    widget_matlab_status.value = (
                        "<div style='padding: 10px; background: #ffcdd2;'>"
                        "<b>✗ MATLAB Status:</b> Connection failed"
                        "</div>"
                    )
                    widget_matlab_toolbox_info.value = (
                        "<div style='padding: 10px; background: #fff3e0;'>"
                        "<b>⚠ Error:</b> Could not start MATLAB Engine. "
                        "Ensure MATLAB is installed and matlab.engine is configured."
                        "</div>"
                    )
            except Exception as e:
                widget_matlab_status.value = (
                    "<div style='padding: 10px; background: #ffcdd2;'>"
                    f"<b>✗ MATLAB Status:</b> Error - {str(e)}"
                    "</div>"
                )
                widget_matlab_toolbox_info.value = (
                    "<div style='padding: 10px; background: #fff3e0;'>"
                    "<b>⚠ Note:</b> MATLAB Engine not available. Install with:<br>"
                    "<code>cd /path/to/MATLAB/extern/engines/python && python setup.py install</code>"
                    "</div>"
                )
        else:
            # Python backend - disable MATLAB controls
            widget_matlab_scenario.disabled = True
            widget_matlab_status.value = (
                "<div style='padding: 10px; background: #e0e0e0;'>"
                "<b>MATLAB Status:</b> Not selected (using Python)"
                "</div>"
            )
            widget_matlab_toolbox_info.value = ""
            matlab_advanced_params.layout.display = 'none'

    def update_scenario_ui(change):
        """Update UI when MATLAB scenario changes."""
        scenario = change['new']

        # Show/hide advanced params based on scenario
        if scenario == 'cdl_ris':
            matlab_advanced_params.layout.display = 'block'
            widget_carrier_frequency.disabled = False
            widget_delay_profile.disabled = False
            widget_doppler_shift.disabled = False
        else:
            matlab_advanced_params.layout.display = 'none'
            widget_carrier_frequency.disabled = True
            widget_delay_profile.disabled = True
            widget_doppler_shift.disabled = True

        # Update info box with scenario details
        try:
            from physics.matlab_backend.toolbox_registry import SCENARIO_TEMPLATES
            template = SCENARIO_TEMPLATES.get(scenario)

            if template:
                info_html = f"<div style='padding: 10px; background: #f3e5f5;'>"
                info_html += f"<b>Scenario:</b> {template.name}<br>"
                info_html += f"<b>Toolbox:</b> {template.toolbox.replace('_', ' ').title()}<br>"
                info_html += f"<b>Description:</b> {template.description}<br>"
                info_html += f"<b>Reference:</b> <a href='{template.reference}' target='_blank'>MathWorks Docs</a>"
                info_html += "</div>"

                widget_matlab_toolbox_info.value = info_html
        except:
            pass

    # Attach observers
    widget_physics_backend.observe(update_backend_ui, 'value')
    widget_matlab_scenario.observe(update_scenario_ui, 'value')

    # ========================================================================
    # PHASE 1: Realism Profile (unchanged)
    # ========================================================================

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

    # ========================================================================
    # PHASE 1: Advanced Impairments (unchanged)
    # ========================================================================

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

    # ========================================================================
    # FINAL LAYOUT
    # ========================================================================

    tab_layout = widgets.VBox([
        widgets.HTML("<h3>Physics & Realism Configuration</h3>"),

        widgets.HTML("<h4>Channel Source (Phase 1)</h4>"),
        widget_channel_source,
        widget_source_info,

        widgets.HTML("<hr style='margin: 20px 0;'>"),
        widgets.HTML("<h4>Backend Selection (Phase 2)</h4>"),
        widget_physics_backend,
        widget_matlab_status,
        widget_matlab_scenario,
        widget_matlab_toolbox_info,
        matlab_advanced_params,

        widgets.HTML("<hr style='margin: 20px 0;'>"),
        widgets.HTML("<h4>Realism Profile</h4>"),
        widget_realism_profile,
        widget_profile_info,

        widgets.HTML("<hr style='margin: 20px 0;'>"),
        widget_use_custom_impairments,
        advanced_container
    ], layout=widgets.Layout(padding='20px'))

    # ========================================================================
    # Store widget references
    # ========================================================================

    tab_layout._widgets = {
        # Phase 1 widgets
        'channel_source': widget_channel_source,
        'source_info': widget_source_info,
        'realism_profile': widget_realism_profile,
        'profile_info': widget_profile_info,
        'use_custom_impairments': widget_use_custom_impairments,
        'csi_error_db': widget_csi_error_db,
        'doppler_hz': widget_doppler_hz,
        'phase_bits_hw': widget_phase_bits_hw,
        'adc_bits': widget_adc_bits,
        # Phase 2 widgets (NEW)
        'physics_backend': widget_physics_backend,
        'matlab_scenario': widget_matlab_scenario,
        'matlab_status': widget_matlab_status,
        'matlab_toolbox_info': widget_matlab_toolbox_info,
        'carrier_frequency': widget_carrier_frequency,
        'delay_profile': widget_delay_profile,
        'doppler_shift_matlab': widget_doppler_shift,
    }

    return tab_layout