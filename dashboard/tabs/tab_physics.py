"""
Tab 2: Physics & Realism (Real MATLAB Logic)
============================================
Features:
- Real MATLAB Engine Detection
- Dynamic Status Indicators (Green/Red)
- Carrier Frequency always visible
- Rich Profile Descriptions
"""

import ipywidgets as widgets
import time
import sys

def create_group_box(title, content_list, color="#00897B"):
    header = widgets.HTML(f"<div style='background-color: {color}; color: white; padding: 4px 10px; font-size: 12px; font-weight: bold; border-radius: 4px 4px 0 0;'>{title}</div>")
    body = widgets.VBox(content_list, layout=widgets.Layout(padding='10px', border=f'1px solid {color}', border_radius='0 0 4px 4px', margin_bottom='10px', width='100%'))
    return widgets.VBox([header, body])

def check_matlab_installed():
    """Checks if the Python-MATLAB bridge is actually installed."""
    try:
        import matlab.engine
        return True, "MATLAB Engine Detected"
    except ImportError:
        return False, "❌ ERROR: 'matlab.engine' not installed. Run: pip install matlabengine"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def create_physics_tab():
    # ========================================================================
    # 1. SIMULATION ENGINE
    # ========================================================================

    widget_backend = widgets.ToggleButtons(
        options=['python', 'matlab'],
        value='python',
        description='Engine:',
        button_style='',
        style={'button_width': '100px'},
        layout=widgets.Layout(width='100%')
    )

    # --- MATLAB CONFIGURATION ---
    widget_matlab_mode = widgets.ToggleButtons(
        options=['Live Engine', 'Data File'],
        value='Live Engine',
        description='Source:',
        button_style='',
        style={'button_width': '100px'},
    )

    # DYNAMIC STATUS WIDGET
    widget_matlab_status = widgets.HTML(
        value="<div style='padding: 8px; background: #eee; border-left: 4px solid #999; color: #666;'><i>Select MATLAB backend to connect...</i></div>",
        layout=widgets.Layout(width='98%', margin='5px 0')
    )

    widget_scenario = widgets.Dropdown(
        options=[('Rayleigh Fading (Comm Toolbox)', 'rayleigh_basic'), ('CDL-RIS Channel (5G Toolbox)', 'cdl_ris')],
        value='rayleigh_basic',
        description='Scenario:',
        layout=widgets.Layout(width='98%')
    )

    # GLOBAL MATLAB PARAMS (Moved out of CDL box so they are always visible)
    widget_carrier_freq = widgets.FloatText(value=28e9, description='Freq (Hz):', layout=widgets.Layout(width='98%'))

    # CDL Specifics
    widget_delay_profile = widgets.Dropdown(options=['CDL-A', 'CDL-B', 'CDL-C'], value='CDL-C', description='Profile:', layout=widgets.Layout(width='98%'))
    widget_doppler_matlab = widgets.FloatText(value=5.0, description='Doppler (Hz):', layout=widgets.Layout(width='98%'))

    # Scenario Info Box
    widget_scenario_info = widgets.HTML(value="", layout=widgets.Layout(width='98%', margin='5px 0'))

    # Containers
    container_cdl_params = widgets.VBox([
        widgets.HTML("<b>CDL-RIS Specifics:</b>"),
        widget_delay_profile,
        widget_doppler_matlab
    ], layout=widgets.Layout(border='1px solid #ddd', padding='10px', margin='10px 0'))

    widget_file_path = widgets.Text(placeholder='C:/Data/channel.mat', description='Path:', layout=widgets.Layout(width='98%'))

    container_matlab_live = widgets.VBox([
        widget_matlab_status,
        widget_scenario,
        widget_carrier_freq, # Moved here
        widget_scenario_info,
        container_cdl_params
    ])

    container_matlab_file = widgets.VBox([widgets.HTML("<i>Load .mat file:</i>"), widget_file_path])

    container_matlab = widgets.VBox([
        widgets.HTML("<hr style='margin: 5px 0'>"),
        widget_matlab_mode,
        container_matlab_live,
        container_matlab_file
    ], layout=widgets.Layout(display='none', padding='5px 0 0 10px'))

    box_engine = create_group_box("⚙️ SIMULATION ENGINE", [widget_backend, container_matlab])

    # ========================================================================
    # 2. REALISM & IMPAIRMENTS
    # ========================================================================

    widget_realism = widgets.Dropdown(
        options=[('Ideal (No Impairments)', 'ideal'), ('Mild (Lab Conditions)', 'mild'), ('Moderate (Typical Indoor)', 'moderate'), ('Severe (Outdoor/Mobile)', 'severe'), ('Worst Case (Stress Test)', 'worst')],
        value='ideal', description='Profile:', style={'description_width': '80px'}, layout=widgets.Layout(width='98%')
    )

    widget_profile_info = widgets.HTML(value="")
    widget_use_custom = widgets.Checkbox(value=False, description='Unlock & Override Settings')

    widget_coupling = widgets.FloatSlider(value=0.0, min=0.0, max=1.0, step=0.05, description='Coupling:', disabled=True, layout=widgets.Layout(width='98%'))
    widget_doppler = widgets.FloatSlider(value=0.0, min=0.0, max=200.0, description='Doppler (Hz):', disabled=True, layout=widgets.Layout(width='98%'))
    widget_csi_error = widgets.FloatSlider(value=-100.0, min=-50.0, max=0.0, description='CSI Err (dB):', disabled=True, layout=widgets.Layout(width='98%'))
    widget_phase_bits = widgets.IntSlider(value=8, min=1, max=8, description='HW Bits:', disabled=True, layout=widgets.Layout(width='98%'))
    widget_adc_bits = widgets.IntSlider(value=12, min=1, max=16, description='ADC Bits:', disabled=True, layout=widgets.Layout(width='98%'))

    grid_impairments = widgets.GridBox([widget_coupling, widget_doppler, widget_csi_error, widget_phase_bits, widget_adc_bits], layout=widgets.Layout(grid_template_columns='repeat(2, 50%)', grid_gap='5px'))

    box_impairments = create_group_box("⚠️ HARDWARE IMPAIRMENTS", [widget_realism, widget_profile_info, widgets.HTML("<hr style='margin: 5px 0'>"), widget_use_custom, grid_impairments])

    # ========================================================================
    # LOGIC
    # ========================================================================

    def on_backend_change(change):
        if change['new'] == 'matlab':
            container_matlab.layout.display = 'block'
            # RUN REAL CHECK
            widget_matlab_status.value = "<div style='padding: 8px; background: #fff3e0; border-left: 4px solid #ff9800; color: #ef6c00;'>⏳ Checking MATLAB Engine...</div>"

            # Simple check
            is_installed, msg = check_matlab_installed()
            if is_installed:
                widget_matlab_status.value = f"<div style='padding: 8px; background: #e8f5e9; border-left: 4px solid #4caf50; color: #2e7d32;'><b>✓ {msg}</b><br><span style='font-size:10px'>Ready to query toolboxes on run.</span></div>"
            else:
                widget_matlab_status.value = f"<div style='padding: 8px; background: #ffebee; border-left: 4px solid #f44336; color: #c62828;'><b>{msg}</b></div>"
        else:
            container_matlab.layout.display = 'none'

    widget_backend.observe(on_backend_change, names='value')

    def on_matlab_mode_change(change):
        is_live = (change['new'] == 'Live Engine')
        container_matlab_live.layout.display = 'block' if is_live else 'none'
        container_matlab_file.layout.display = 'none' if is_live else 'block'
    widget_matlab_mode.observe(on_matlab_mode_change, names='value')

    def on_scenario_change(change):
        scen = change['new']
        if scen == 'cdl_ris':
            container_cdl_params.layout.display = 'block'
            info_html = "<div style='padding: 10px; background: #f3e5f5; border-left: 4px solid #ab47bc;'><b>Scenario:</b> CDL-RIS<br><b>Required:</b> 5G Toolbox<br><b>Desc:</b> 3GPP Clustered Delay Line model.</div>"
        else:
            container_cdl_params.layout.display = 'none'
            info_html = "<div style='padding: 10px; background: #e3f2fd; border-left: 4px solid #2196f3;'><b>Scenario:</b> Rayleigh Fading<br><b>Required:</b> Comm Toolbox<br><b>Desc:</b> Standard NLOS statistical fading.</div>"
        widget_scenario_info.value = info_html

    widget_scenario.observe(on_scenario_change, names='value')

    # Profile Logic
    PROFILES = {
        'ideal':    {'c': 0.0, 'd': 0.0, 'e': -100, 'b': 8, 'a': 16, 'desc': '<b>Ideal:</b> Perfect CSI, no errors.', 'color': '#e8f5e9', 'border': '#4caf50'},
        'mild':     {'c': 0.1, 'd': 5.0, 'e': -30,  'b': 6, 'a': 12, 'desc': '<b>Mild:</b> Lab equipment quality.', 'color': '#fff9c4', 'border': '#fbc02d'},
        'moderate': {'c': 0.3, 'd': 20.0, 'e': -20, 'b': 4, 'a': 10, 'desc': '<b>Moderate:</b> Typical indoor usage.', 'color': '#ffe0b2', 'border': '#f57c00'},
        'severe':   {'c': 0.5, 'd': 100.0,'e': -15, 'b': 2, 'a': 8,  'desc': '<b>Severe:</b> Outdoor/Vehicular.', 'color': '#ffccbc', 'border': '#d84315'},
        'worst':    {'c': 0.8, 'd': 200.0,'e': -5,  'b': 1, 'a': 6,  'desc': '<b>Worst:</b> Stress test limits.', 'color': '#f3e5f5', 'border': '#7b1fa2'},
    }

    def update_profile(change=None):
        p = PROFILES[widget_realism.value]
        widget_profile_info.value = f"<div style='margin: 5px 0; padding: 8px; background: {p['color']}; border-left: 4px solid {p['border']}; font-size: 11px;'>{p['desc']}</div>"
        if not widget_use_custom.value:
            widget_coupling.value = p['c']; widget_doppler.value = p['d']; widget_csi_error.value = p['e']; widget_phase_bits.value = p['b']; widget_adc_bits.value = p['a']

    def on_override_change(change):
        is_custom = change['new']
        for w in [widget_coupling, widget_doppler, widget_csi_error, widget_phase_bits, widget_adc_bits]: w.disabled = not is_custom
        if not is_custom: update_profile()

    widget_realism.observe(update_profile, names='value')
    widget_use_custom.observe(on_override_change, names='value')

    # Init
    on_backend_change({'new': widget_backend.value})
    on_matlab_mode_change({'new': widget_matlab_mode.value})
    on_scenario_change({'new': widget_scenario.value})
    update_profile()

    tab_layout = widgets.VBox([box_engine, box_impairments])

    # Store references
    tab_layout._widgets = {
        'physics_backend': widget_backend,
        'matlab_mode': widget_matlab_mode,
        'matlab_scenario': widget_scenario,
        'carrier_frequency': widget_carrier_freq,
        'delay_profile': widget_delay_profile,
        'matlab_file_path': widget_file_path,
        'realism_profile': widget_realism,
        'use_custom_impairments': widget_use_custom,
        'varactor_coupling_strength': widget_coupling,
        'doppler_hz': widget_doppler,
        'csi_error_db': widget_csi_error,
        'phase_bits_hw': widget_phase_bits,
        'adc_bits': widget_adc_bits,
        'doppler_shift_matlab': widget_doppler_matlab,
        'coupling_strength': widget_coupling,
        'phase_bits': widget_phase_bits
    }
    return tab_layout