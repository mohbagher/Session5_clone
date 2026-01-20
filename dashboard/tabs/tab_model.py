"""
Tab 3: Model Architecture Configuration
========================================
Model presets, layer configuration, transfer learning.
"""

import ipywidgets as widgets
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from models.model_registry import list_models
except ImportError:
    def list_models():
        return ['Baseline_MLP', 'Deep_MLP', 'Custom']


def create_model_tab():
    """Create Model Architecture tab."""

    # Model preset dropdown
    model_options = [
        '━━ Standard MLPs ━━',
        'Baseline_MLP', 'Deep_MLP', 'Tiny_MLP', 'Wide_Deep',
        '━━ Learnable M Selection ━━',
        'LearnedTopK_MLP', 'Attention_MLP', 'Gumbel_MLP', 'RL_MLP',
        '━━ Research Architectures ━━',
        'ResNet_Style', 'Pyramid', 'Hourglass', 'PhD_Custom_1', 'PhD_Custom_2',
        '━━ Custom ━━',
        'Custom'
    ]

    widget_model_preset = widgets.Dropdown(
        options=model_options,
        value='Baseline_MLP',
        description='Model preset:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px')
    )

    # Transfer learning
    widget_transfer_source = widgets.Dropdown(
        options=['None'],
        value='None',
        description='Transfer from:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px'),
        tooltip='Initialize weights from pre-trained model',
        disabled=True
    )

    # Custom architecture controls
    widget_num_layers = widgets.IntSlider(
        value=3, min=1, max=10, step=1,
        description='Number of layers:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px'),
        disabled=True
    )

    # Layer sizes container (dynamically generated)
    widget_layer_sizes_container = widgets.VBox([])

    def update_layer_size_inputs(change=None):
        """Update layer size input widgets based on num_layers."""
        num = widget_num_layers.value
        children = []
        for i in range(num):
            size_widget = widgets.IntText(
                value=512 // (2**i),  # Default: 512, 256, 128, ...
                description=f'Layer {i+1} size:',
                style={'description_width': '180px'},
                layout=widgets.Layout(width='500px')
            )
            children.append(size_widget)
        widget_layer_sizes_container.children = children

    widget_num_layers.observe(update_layer_size_inputs, 'value')
    update_layer_size_inputs()  # Initialize

    # Regularization
    widget_dropout_prob = widgets.FloatSlider(
        value=0.1, min=0.0, max=0.8, step=0.05,
        description='Dropout probability:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px')
    )

    widget_use_batch_norm = widgets.Checkbox(
        value=True,
        description='Use Batch Normalization',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px')
    )

    # Advanced settings
    widget_activation_function = widgets.Dropdown(
        options=['ReLU', 'LeakyReLU', 'GELU', 'ELU', 'Tanh'],
        value='ReLU',
        description='Activation function:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px')
    )

    widget_weight_init = widgets.Dropdown(
        options=['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal'],
        value='xavier_uniform',
        description='Weight initialization:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px')
    )

    # Parameter count display
    widget_param_count_display = widgets.HTML(
        value="<b>Estimated parameters:</b> Calculating...",
        layout=widgets.Layout(width='500px')
    )

    # Enable/disable custom controls based on preset
    def toggle_custom_controls(change):
        is_custom = (change['new'] == 'Custom')
        widget_num_layers.disabled = not is_custom
        for child in widget_layer_sizes_container.children:
            child.disabled = not is_custom

    widget_model_preset.observe(toggle_custom_controls, 'value')

    # Layout
    tab_layout = widgets.VBox([
        widgets.HTML("<h3>Model Architecture & Transfer Learning</h3>"),
        widgets.HTML("<h4>Model Selection</h4>"),
        widget_model_preset,
        widget_transfer_source,
        widgets.HTML("<hr style='margin: 15px 0;'>"),
        widgets.HTML("<h4>Custom Architecture (only if 'Custom' selected above)</h4>"),
        widget_num_layers,
        widget_layer_sizes_container,
        widgets.HTML("<hr style='margin: 15px 0;'>"),
        widgets.HTML("<h4>Regularization</h4>"),
        widget_dropout_prob,
        widget_use_batch_norm,
        widgets.HTML("<h4 style='margin-top: 20px;'>Advanced Settings</h4>"),
        widget_activation_function,
        widget_weight_init,
        widgets.HTML("<hr style='margin: 15px 0;'>"),
        widget_param_count_display
    ], layout=widgets.Layout(padding='20px'))

    # Store widget references
    tab_layout._widgets = {
        'model_preset': widget_model_preset,
        'transfer_source': widget_transfer_source,
        'num_layers': widget_num_layers,
        'layer_sizes_container': widget_layer_sizes_container,
        'dropout_prob': widget_dropout_prob,
        'use_batch_norm': widget_use_batch_norm,
        'activation_function': widget_activation_function,
        'weight_init': widget_weight_init,
        'param_count_display': widget_param_count_display
    }

    return tab_layout
