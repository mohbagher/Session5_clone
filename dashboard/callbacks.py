"""
Dashboard Callbacks for RIS PhD Ultimate Dashboard.

Event handlers for widget interactions and dynamic UI updates.
"""

import ipywidgets as widgets
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_registry import get_model_architecture


def on_phase_mode_change(change, widgets_dict):
    """Enable/disable phase_bits based on phase_mode selection."""
    phase_bits_widget = widgets_dict['phase_bits']
    if change['new'] == 'discrete':
        phase_bits_widget.disabled = False
    else:
        phase_bits_widget.disabled = True


def on_K_change(change, widgets_dict):
    """Update M widget's max value when K changes."""
    M_widget = widgets_dict['M']
    new_K = change['new']
    M_widget.max = new_K
    # Ensure M doesn't exceed K
    if M_widget.value > new_K:
        M_widget.value = new_K


def on_M_change(change, widgets_dict):
    """Validate that M <= K."""
    M_value = change['new']
    K_value = widgets_dict['K'].value
    status_output = widgets_dict.get('status_output')
    
    if M_value > K_value:
        if status_output:
            with status_output:
                print(f"⚠️ Warning: M ({M_value}) cannot exceed K ({K_value}). Adjusting M to {K_value}.")
        widgets_dict['M'].value = K_value


def on_model_preset_change(change, widgets_dict):
    """Show/hide custom layer configuration when preset changes."""
    num_layers_widget = widgets_dict['num_layers']
    layer_sizes_container = widgets_dict['layer_sizes_container']
    
    if change['new'] == 'Custom':
        num_layers_widget.disabled = False
        # Create initial layer size inputs
        update_layer_size_widgets(num_layers_widget.value, widgets_dict)
    else:
        num_layers_widget.disabled = True
        layer_sizes_container.children = []
    
    # Update parameter count
    update_param_count_preview(widgets_dict)


def on_num_layers_change(change, widgets_dict):
    """Update layer size input widgets when number of layers changes."""
    num_layers = change['new']
    update_layer_size_widgets(num_layers, widgets_dict)
    update_param_count_preview(widgets_dict)


def update_layer_size_widgets(num_layers, widgets_dict):
    """Create/update layer size input widgets."""
    layer_sizes_container = widgets_dict['layer_sizes_container']
    
    # Create widgets for each layer
    layer_widgets = []
    for i in range(num_layers):
        default_size = max(32, 512 // (2 ** i))  # Decreasing sizes: 512, 256, 128, ...
        widget = widgets.IntText(
            value=default_size,
            description=f'Layer {i+1} size:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='300px')
        )
        # Attach callback to update parameter count
        widget.observe(lambda change: update_param_count_preview(widgets_dict), names='value')
        layer_widgets.append(widget)
    
    layer_sizes_container.children = layer_widgets


def get_current_hidden_sizes(widgets_dict):
    """Extract current hidden layer sizes from widgets."""
    preset = widgets_dict['model_preset'].value
    
    if preset == 'Custom':
        layer_sizes_container = widgets_dict['layer_sizes_container']
        hidden_sizes = []
        for widget in layer_sizes_container.children:
            if isinstance(widget, widgets.IntText):
                hidden_sizes.append(widget.value)
        return hidden_sizes
    else:
        try:
            return get_model_architecture(preset)
        except:
            return [256, 128]  # Fallback


def update_param_count_preview(widgets_dict):
    """Calculate and display estimated number of parameters."""
    try:
        K = widgets_dict['K'].value
        input_size = 2 * K
        output_size = K
        
        hidden_sizes = get_current_hidden_sizes(widgets_dict)
        
        # Calculate parameters
        total_params = 0
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            # Weights + biases
            total_params += prev_size * hidden_size + hidden_size
            prev_size = hidden_size
        
        # Output layer
        total_params += prev_size * output_size + output_size
        
        # Display
        param_display = widgets_dict['param_count_display']
        param_display.value = f"<b>Estimated parameters:</b> {total_params:,} ({total_params/1e6:.2f}M)"
        
    except Exception as e:
        param_display = widgets_dict['param_count_display']
        param_display.value = f"<b>Estimated parameters:</b> Error - {str(e)}"


def on_compare_models_change(change, widgets_dict):
    """Show/hide model selection when compare_multiple_models changes."""
    models_to_compare_widget = widgets_dict['models_to_compare']
    
    if change['new']:
        models_to_compare_widget.disabled = False
    else:
        models_to_compare_widget.disabled = True
        models_to_compare_widget.value = []


def on_multi_seed_change(change, widgets_dict):
    """Show/hide num_seeds when multi_seed_runs changes."""
    num_seeds_widget = widgets_dict['num_seeds']
    
    if change['new']:
        num_seeds_widget.disabled = False
    else:
        num_seeds_widget.disabled = True


def setup_all_callbacks(widgets_dict):
    """
    Attach all callbacks to widgets.
    
    Args:
        widgets_dict: Dictionary of all widgets from get_all_widgets()
    """
    # Tab 1: System callbacks
    widgets_dict['phase_mode'].observe(
        lambda change: on_phase_mode_change(change, widgets_dict),
        names='value'
    )
    
    widgets_dict['K'].observe(
        lambda change: on_K_change(change, widgets_dict),
        names='value'
    )
    
    widgets_dict['M'].observe(
        lambda change: on_M_change(change, widgets_dict),
        names='value'
    )
    
    # Tab 2: Model callbacks
    widgets_dict['model_preset'].observe(
        lambda change: on_model_preset_change(change, widgets_dict),
        names='value'
    )
    
    widgets_dict['num_layers'].observe(
        lambda change: on_num_layers_change(change, widgets_dict),
        names='value'
    )
    
    # Update parameter count on K change as well
    widgets_dict['K'].observe(
        lambda change: update_param_count_preview(widgets_dict),
        names='value'
    )
    
    # Tab 4: Evaluation callbacks
    widgets_dict['compare_multiple_models'].observe(
        lambda change: on_compare_models_change(change, widgets_dict),
        names='value'
    )
    
    widgets_dict['multi_seed_runs'].observe(
        lambda change: on_multi_seed_change(change, widgets_dict),
        names='value'
    )
    
    # Initial parameter count calculation
    update_param_count_preview(widgets_dict)


def reset_to_defaults(widgets_dict):
    """Reset all widgets to their default values."""
    # Tab 1: System & Physics
    widgets_dict['N'].value = 32
    widgets_dict['K'].value = 64
    widgets_dict['M'].value = 8
    widgets_dict['P_tx'].value = 1.0
    widgets_dict['sigma_h_sq'].value = 1.0
    widgets_dict['sigma_g_sq'].value = 1.0
    widgets_dict['phase_mode'].value = 'continuous'
    widgets_dict['phase_bits'].value = 3
    widgets_dict['probe_type'].value = 'continuous'
    
    # Tab 2: Model Architecture
    widgets_dict['model_preset'].value = 'Baseline_MLP'
    widgets_dict['num_layers'].value = 3
    widgets_dict['dropout_prob'].value = 0.1
    widgets_dict['use_batch_norm'].value = True
    widgets_dict['activation_function'].value = 'ReLU'
    widgets_dict['weight_init'].value = 'xavier_uniform'
    
    # Tab 3: Training Configuration
    widgets_dict['n_train'].value = 50000
    widgets_dict['n_val'].value = 5000
    widgets_dict['n_test'].value = 5000
    widgets_dict['seed'].value = 42
    widgets_dict['normalize_input'].value = True
    widgets_dict['normalization_type'].value = 'mean'
    widgets_dict['batch_size'].value = 128
    widgets_dict['learning_rate'].value = 1e-3
    widgets_dict['weight_decay'].value = 1e-4
    widgets_dict['n_epochs'].value = 50
    widgets_dict['early_stop_patience'].value = 10
    widgets_dict['optimizer'].value = 'Adam'
    widgets_dict['scheduler'].value = 'ReduceLROnPlateau'
    
    # Tab 4: Evaluation & Comparison
    widgets_dict['top_m_values'].value = [1, 2, 4, 8]
    widgets_dict['compare_multiple_models'].value = False
    widgets_dict['models_to_compare'].value = []
    widgets_dict['multi_seed_runs'].value = False
    widgets_dict['num_seeds'].value = 3
    widgets_dict['compute_confidence_intervals'].value = False
    
    # Tab 5: Visualization
    widgets_dict['selected_plots'].value = ['training_curves', 'eta_distribution', 'top_m_comparison']
    widgets_dict['figure_format'].value = 'png'
    widgets_dict['dpi'].value = 150
    widgets_dict['color_palette'].value = 'viridis'
    widgets_dict['save_plots'].value = True
    widgets_dict['output_dir'].value = 'results/'
    
    # Clear status
    status_output = widgets_dict.get('status_output')
    if status_output:
        status_output.clear_output()
        with status_output:
            print("✅ All settings reset to defaults.")
