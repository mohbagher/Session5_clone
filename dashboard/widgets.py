"""
Dashboard Widgets for RIS PhD Ultimate Dashboard.

All widget definitions organized by tab/category.
"""

import ipywidgets as widgets
from typing import Dict, List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_registry import list_models


# ============================================================================
# TAB 1: SYSTEM & PHYSICS PARAMETERS
# ============================================================================

widget_N = widgets.IntSlider(
    value=32,
    min=4,
    max=256,
    step=1,
    description='N (RIS elements):',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)

widget_K = widgets.IntSlider(
    value=64,
    min=4,
    max=512,
    step=1,
    description='K (Codebook size):',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)

widget_M = widgets.IntSlider(
    value=8,
    min=1,
    max=64,  # Will be dynamically updated based on K
    step=1,
    description='M (Sensing budget):',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)

widget_P_tx = widgets.FloatSlider(
    value=1.0,
    min=0.1,
    max=10.0,
    step=0.1,
    description='P_tx (Transmit power):',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)

widget_sigma_h_sq = widgets.FloatSlider(
    value=1.0,
    min=0.1,
    max=10.0,
    step=0.1,
    description='σ_h² (BS-RIS variance):',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)

widget_sigma_g_sq = widgets.FloatSlider(
    value=1.0,
    min=0.1,
    max=10.0,
    step=0.1,
    description='σ_g² (RIS-UE variance):',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)

widget_phase_mode = widgets.Dropdown(
    options=['continuous', 'discrete'],
    value='continuous',
    description='Phase mode:',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)

widget_phase_bits = widgets.IntSlider(
    value=3,
    min=1,
    max=8,
    step=1,
    description='Phase bits:',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px'),
    disabled=True  # Enabled only when phase_mode='discrete'
)

widget_probe_type = widgets.Dropdown(
    options=['continuous', 'binary', '2bit', 'hadamard', 'sobol', 'halton'],
    value='continuous',
    description='Probe type:',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)


# ============================================================================
# TAB 2: MODEL ARCHITECTURE
# ============================================================================

widget_model_preset = widgets.Dropdown(
    options=list_models() + ['Custom'],
    value='Baseline_MLP',
    description='Model preset:',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)

widget_num_layers = widgets.IntSlider(
    value=3,
    min=1,
    max=10,
    step=1,
    description='Number of layers:',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px'),
    disabled=True  # Enabled only when preset='Custom'
)

# Dynamic layer size widgets (created in callbacks when needed)
widget_layer_sizes_container = widgets.VBox([])

widget_dropout_prob = widgets.FloatSlider(
    value=0.1,
    min=0.0,
    max=0.8,
    step=0.05,
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

widget_param_count_display = widgets.HTML(
    value="<b>Estimated parameters:</b> Calculating...",
    layout=widgets.Layout(width='500px')
)


# ============================================================================
# TAB 3: TRAINING CONFIGURATION
# ============================================================================

widget_n_train = widgets.IntText(
    value=50000,
    description='Training samples:',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)

widget_n_val = widgets.IntText(
    value=5000,
    description='Validation samples:',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)

widget_n_test = widgets.IntText(
    value=5000,
    description='Test samples:',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)

widget_seed = widgets.IntText(
    value=42,
    description='Random seed:',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)

widget_normalize_input = widgets.Checkbox(
    value=True,
    description='Normalize input',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)

widget_normalization_type = widgets.Dropdown(
    options=['mean', 'std', 'log'],
    value='mean',
    description='Normalization type:',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)

widget_batch_size = widgets.Dropdown(
    options=[32, 64, 128, 256, 512],
    value=128,
    description='Batch size:',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)

widget_learning_rate = widgets.FloatLogSlider(
    value=1e-3,
    min=-5,  # 1e-5
    max=-1,  # 1e-1
    step=0.1,
    description='Learning rate:',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px'),
    readout_format='.1e'
)

widget_weight_decay = widgets.FloatLogSlider(
    value=1e-4,
    min=-6,  # 1e-6
    max=-2,  # 1e-2
    step=0.1,
    description='Weight decay:',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px'),
    readout_format='.1e'
)

widget_n_epochs = widgets.IntSlider(
    value=50,
    min=1,
    max=500,
    step=1,
    description='Max epochs:',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)

widget_early_stop_patience = widgets.IntSlider(
    value=10,
    min=1,
    max=50,
    step=1,
    description='Early stop patience:',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)

widget_optimizer = widgets.Dropdown(
    options=['Adam', 'AdamW', 'SGD', 'RMSprop'],
    value='Adam',
    description='Optimizer:',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)

widget_scheduler = widgets.Dropdown(
    options=['ReduceLROnPlateau', 'CosineAnnealing', 'StepLR', 'None'],
    value='ReduceLROnPlateau',
    description='LR scheduler:',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)


# ============================================================================
# TAB 4: EVALUATION & COMPARISON
# ============================================================================

widget_top_m_values = widgets.SelectMultiple(
    options=[1, 2, 4, 8, 16, 32],
    value=[1, 2, 4, 8],
    description='Top-m values:',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px', height='120px')
)

widget_compare_multiple_models = widgets.Checkbox(
    value=False,
    description='Compare multiple models',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)

widget_models_to_compare = widgets.SelectMultiple(
    options=list_models(),
    value=[],
    description='Models to compare:',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px', height='150px'),
    disabled=True  # Enabled when compare_multiple_models=True
)

widget_multi_seed_runs = widgets.Checkbox(
    value=False,
    description='Multi-seed runs',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)

widget_num_seeds = widgets.IntSlider(
    value=3,
    min=1,
    max=10,
    step=1,
    description='Number of seeds:',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px'),
    disabled=True  # Enabled when multi_seed_runs=True
)

widget_compute_confidence_intervals = widgets.Checkbox(
    value=False,
    description='Compute confidence intervals',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)


# ============================================================================
# TAB 5: VISUALIZATION
# ============================================================================

# All available plot types
ALL_PLOT_TYPES = [
    'training_curves', 'eta_distribution', 'cdf', 'top_m_comparison',
    'baseline_comparison', 'violin', 'box', 'scatter', 'heatmap',
    'correlation_matrix', 'confusion_matrix', 'learning_curve',
    'eta_vs_M', 'eta_vs_K', 'eta_vs_N', 'probe_type_comparison',
    'phase_bits_comparison', 'convergence_analysis', 'model_size_vs_performance',
    'radar_chart', 'error_analysis', 'power_distribution', 'channel_statistics',
    '3d_surface', 'pareto_front'
]

widget_selected_plots = widgets.SelectMultiple(
    options=ALL_PLOT_TYPES,
    value=['training_curves', 'eta_distribution', 'top_m_comparison'],
    description='Select plots:',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px', height='200px')
)

widget_figure_format = widgets.Dropdown(
    options=['png', 'pdf', 'svg'],
    value='png',
    description='Figure format:',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)

widget_dpi = widgets.IntSlider(
    value=150,
    min=72,
    max=300,
    step=6,
    description='DPI:',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)

widget_color_palette = widgets.Dropdown(
    options=['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'seaborn'],
    value='viridis',
    description='Color palette:',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)

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


# ============================================================================
# CELL 8: ACTION BUTTONS & STATUS
# ============================================================================

button_run_experiment = widgets.Button(
    description='🚀 RUN EXPERIMENT',
    button_style='success',
    layout=widgets.Layout(width='200px', height='50px'),
    style={'font_weight': 'bold'}
)

button_save_config = widgets.Button(
    description='💾 SAVE CONFIG',
    button_style='info',
    layout=widgets.Layout(width='200px', height='50px'),
    style={'font_weight': 'bold'}
)

button_load_config = widgets.Button(
    description='📂 LOAD CONFIG',
    button_style='warning',
    layout=widgets.Layout(width='200px', height='50px'),
    style={'font_weight': 'bold'}
)

button_reset_defaults = widgets.Button(
    description='🔄 RESET DEFAULTS',
    button_style='danger',
    layout=widgets.Layout(width='200px', height='50px'),
    style={'font_weight': 'bold'}
)

widget_status_output = widgets.Output(
    layout=widgets.Layout(
        width='100%',
        height='200px',
        border='1px solid #ccc',
        padding='10px',
        overflow='auto'
    )
)


# ============================================================================
# CELL 9: REAL-TIME PROGRESS
# ============================================================================

widget_progress_bar = widgets.IntProgress(
    value=0,
    min=0,
    max=100,
    description='Progress:',
    bar_style='info',
    style={'bar_color': '#4CAF50', 'description_width': '100px'},
    layout=widgets.Layout(width='100%')
)

widget_live_metrics = widgets.HTML(
    value="<div style='font-family: monospace; padding: 10px;'><b>Waiting to start...</b></div>",
    layout=widgets.Layout(width='100%', height='150px', border='1px solid #ccc', padding='10px')
)


# ============================================================================
# CELL 10: RESULTS DASHBOARD
# ============================================================================

widget_results_summary = widgets.HTML(
    value="<div style='padding: 10px;'><i>No results yet. Run an experiment to see results.</i></div>",
    layout=widgets.Layout(width='100%', min_height='200px', border='1px solid #ccc', padding='10px')
)

widget_results_plots = widgets.Output(
    layout=widgets.Layout(width='100%', border='1px solid #ccc', padding='10px')
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_widgets() -> Dict:
    """Return dictionary of all widgets for easy access."""
    return {
        # Tab 1: System & Physics
        'N': widget_N,
        'K': widget_K,
        'M': widget_M,
        'P_tx': widget_P_tx,
        'sigma_h_sq': widget_sigma_h_sq,
        'sigma_g_sq': widget_sigma_g_sq,
        'phase_mode': widget_phase_mode,
        'phase_bits': widget_phase_bits,
        'probe_type': widget_probe_type,
        
        # Tab 2: Model Architecture
        'model_preset': widget_model_preset,
        'num_layers': widget_num_layers,
        'layer_sizes_container': widget_layer_sizes_container,
        'dropout_prob': widget_dropout_prob,
        'use_batch_norm': widget_use_batch_norm,
        'activation_function': widget_activation_function,
        'weight_init': widget_weight_init,
        'param_count_display': widget_param_count_display,
        
        # Tab 3: Training Configuration
        'n_train': widget_n_train,
        'n_val': widget_n_val,
        'n_test': widget_n_test,
        'seed': widget_seed,
        'normalize_input': widget_normalize_input,
        'normalization_type': widget_normalization_type,
        'batch_size': widget_batch_size,
        'learning_rate': widget_learning_rate,
        'weight_decay': widget_weight_decay,
        'n_epochs': widget_n_epochs,
        'early_stop_patience': widget_early_stop_patience,
        'optimizer': widget_optimizer,
        'scheduler': widget_scheduler,
        
        # Tab 4: Evaluation & Comparison
        'top_m_values': widget_top_m_values,
        'compare_multiple_models': widget_compare_multiple_models,
        'models_to_compare': widget_models_to_compare,
        'multi_seed_runs': widget_multi_seed_runs,
        'num_seeds': widget_num_seeds,
        'compute_confidence_intervals': widget_compute_confidence_intervals,
        
        # Tab 5: Visualization
        'selected_plots': widget_selected_plots,
        'figure_format': widget_figure_format,
        'dpi': widget_dpi,
        'color_palette': widget_color_palette,
        'save_plots': widget_save_plots,
        'output_dir': widget_output_dir,
        
        # Action buttons
        'button_run_experiment': button_run_experiment,
        'button_save_config': button_save_config,
        'button_load_config': button_load_config,
        'button_reset_defaults': button_reset_defaults,
        'status_output': widget_status_output,
        
        # Progress widgets
        'progress_bar': widget_progress_bar,
        'live_metrics': widget_live_metrics,
        
        # Results widgets
        'results_summary': widget_results_summary,
        'results_plots': widget_results_plots,
    }


def create_tab_layout():
    """Create the tabbed interface layout."""
    
    # Tab 1: System & Physics
    tab1_content = widgets.VBox([
        widgets.HTML("<h3>System & Physics Parameters</h3>"),
        widget_N,
        widget_K,
        widget_M,
        widget_P_tx,
        widget_sigma_h_sq,
        widget_sigma_g_sq,
        widget_phase_mode,
        widget_phase_bits,
        widget_probe_type,
    ], layout=widgets.Layout(padding='20px'))
    
    # Tab 2: Model Architecture
    tab2_content = widgets.VBox([
        widgets.HTML("<h3>Model Architecture</h3>"),
        widget_model_preset,
        widget_num_layers,
        widget_layer_sizes_container,
        widget_dropout_prob,
        widget_use_batch_norm,
        widget_activation_function,
        widget_weight_init,
        widget_param_count_display,
    ], layout=widgets.Layout(padding='20px'))
    
    # Tab 3: Training Configuration
    tab3_content = widgets.VBox([
        widgets.HTML("<h3>Training Configuration</h3>"),
        widget_n_train,
        widget_n_val,
        widget_n_test,
        widget_seed,
        widget_normalize_input,
        widget_normalization_type,
        widget_batch_size,
        widget_learning_rate,
        widget_weight_decay,
        widget_n_epochs,
        widget_early_stop_patience,
        widget_optimizer,
        widget_scheduler,
    ], layout=widgets.Layout(padding='20px'))
    
    # Tab 4: Evaluation & Comparison
    tab4_content = widgets.VBox([
        widgets.HTML("<h3>Evaluation & Comparison</h3>"),
        widget_top_m_values,
        widget_compare_multiple_models,
        widget_models_to_compare,
        widget_multi_seed_runs,
        widget_num_seeds,
        widget_compute_confidence_intervals,
    ], layout=widgets.Layout(padding='20px'))
    
    # Tab 5: Visualization
    tab5_content = widgets.VBox([
        widgets.HTML("<h3>Visualization Options</h3>"),
        widget_selected_plots,
        widget_figure_format,
        widget_dpi,
        widget_color_palette,
        widget_save_plots,
        widget_output_dir,
    ], layout=widgets.Layout(padding='20px'))
    
    # Create tabs
    tabs = widgets.Tab(children=[
        tab1_content,
        tab2_content,
        tab3_content,
        tab4_content,
        tab5_content
    ])
    
    tabs.set_title(0, '⚙️ System & Physics')
    tabs.set_title(1, '🧠 Model Architecture')
    tabs.set_title(2, '📊 Training Config')
    tabs.set_title(3, '📈 Evaluation')
    tabs.set_title(4, '🎨 Visualization')
    
    return tabs
