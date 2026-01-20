"""
Tab 5: Evaluation & Comparison
===============================
Top-m evaluation, model comparison, multi-seed runs.
"""

import ipywidgets as widgets


def create_evaluation_tab():
    """Create Evaluation & Comparison tab."""

    # Top-m evaluation
    widget_top_m_values = widgets.SelectMultiple(
        options=[1, 2, 4, 8, 16, 32],
        value=[1, 2, 4, 8],
        description='Top-m values:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px', height='120px'),
        tooltip='Select which top-m values to evaluate'
    )

    # Model comparison
    widget_compare_multiple_models = widgets.Checkbox(
        value=False,
        description='Compare multiple models',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px')
    )

    widget_models_to_compare = widgets.SelectMultiple(
        options=['Baseline_MLP', 'Deep_MLP', 'Tiny_MLP'],
        value=[],
        description='Models to compare:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px', height='150px'),
        disabled=True,
        tooltip='Select 2+ models for comparison'
    )

    def toggle_model_comparison(change):
        widget_models_to_compare.disabled = not change['new']

    widget_compare_multiple_models.observe(toggle_model_comparison, 'value')

    # Multi-seed runs
    widget_multi_seed_runs = widgets.Checkbox(
        value=False,
        description='Multi-seed runs',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px')
    )

    widget_num_seeds = widgets.IntSlider(
        value=3, min=1, max=10, step=1,
        description='Number of seeds:',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px'),
        disabled=True
    )

    def toggle_multi_seed(change):
        widget_num_seeds.disabled = not change['new']

    widget_multi_seed_runs.observe(toggle_multi_seed, 'value')

    # Statistical analysis
    widget_compute_confidence_intervals = widgets.Checkbox(
        value=False,
        description='Compute confidence intervals',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='500px')
    )

    # Layout
    tab_layout = widgets.VBox([
        widgets.HTML("<h3>Evaluation & Comparison</h3>"),
        widgets.HTML("<h4>Top-m Evaluation</h4>"),
        widget_top_m_values,
        widgets.HTML("<h4 style='margin-top: 20px;'>Model Comparison</h4>"),
        widget_compare_multiple_models,
        widget_models_to_compare,
        widgets.HTML("<h4 style='margin-top: 20px;'>Statistical Validation</h4>"),
        widget_multi_seed_runs,
        widget_num_seeds,
        widget_compute_confidence_intervals
    ], layout=widgets.Layout(padding='20px'))

    # Store widget references
    tab_layout._widgets = {
        'top_m_values': widget_top_m_values,
        'compare_multiple_models': widget_compare_multiple_models,
        'models_to_compare': widget_models_to_compare,
        'multi_seed_runs': widget_multi_seed_runs,
        'num_seeds': widget_num_seeds,
        'compute_confidence_intervals': widget_compute_confidence_intervals
    }

    return tab_layout
