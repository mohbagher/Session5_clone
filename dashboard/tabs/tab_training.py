"""
Tab 4: Training Configuration
==============================
Dataset sizes, batch size, optimizer, learning rate, scheduler.
"""

import ipywidgets as widgets


def create_training_tab():
    """Create Training Configuration tab."""

    # Dataset parameters
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

    # Data preprocessing
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

    # Training hyperparameters
    widget_batch_size = widgets.Dropdown(
        options=[32, 64, 128, 256, 512],
        value=128,
        description='Batch size:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px')
    )

    widget_learning_rate = widgets.FloatLogSlider(
        value=1e-3, min=-5, max=-1, step=0.1,
        description='Learning rate:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px'),
        readout_format='.1e'
    )

    widget_weight_decay = widgets.FloatLogSlider(
        value=1e-4, min=-6, max=-2, step=0.1,
        description='Weight decay:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px'),
        readout_format='.1e'
    )

    widget_n_epochs = widgets.IntSlider(
        value=50, min=1, max=500, step=1,
        description='Max epochs:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px')
    )

    widget_early_stop_patience = widgets.IntSlider(
        value=10, min=1, max=50, step=1,
        description='Early stop patience:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px')
    )

    # Optimizer and scheduler
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

    # Layout
    tab_layout = widgets.VBox([
        widgets.HTML("<h3>Training Configuration</h3>"),
        widgets.HTML("<h4>Dataset Sizes</h4>"),
        widget_n_train,
        widget_n_val,
        widget_n_test,
        widget_seed,
        widgets.HTML("<h4 style='margin-top: 20px;'>Data Preprocessing</h4>"),
        widget_normalize_input,
        widget_normalization_type,
        widgets.HTML("<h4 style='margin-top: 20px;'>Training Hyperparameters</h4>"),
        widget_batch_size,
        widget_learning_rate,
        widget_weight_decay,
        widget_n_epochs,
        widget_early_stop_patience,
        widgets.HTML("<h4 style='margin-top: 20px;'>Optimizer & Scheduler</h4>"),
        widget_optimizer,
        widget_scheduler
    ], layout=widgets.Layout(padding='20px'))

    # Store widget references
    tab_layout._widgets = {
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
        'scheduler': widget_scheduler
    }

    return tab_layout
