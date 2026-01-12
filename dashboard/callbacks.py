"""
Dashboard Callbacks for RIS PhD Ultimate Dashboard.
Optimized for robust routing and user notifications.
"""
import ipywidgets as widgets
import sys, os, time, json, torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Ensure root path is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_registry import get_model_architecture
from dashboard.config_manager import config_to_dict, save_config
from dashboard.validators import get_validation_errors
from dashboard.experiment_runner import (
    run_single_experiment,
    run_multi_model_comparison,
    run_multi_seed_experiment,
    aggregate_results
)
from dashboard.plots import EXTENDED_PLOT_REGISTRY, set_plot_style

# Global state to hold results across cells
CURRENT_RESULTS = None

# ============================================================================
# UI LOGIC & DYNAMIC STATE MANAGEMENT
# ============================================================================

def on_phase_mode_change(change, wd):
    wd['phase_bits'].disabled = (change['new'] != 'discrete')

def on_K_change(change, wd):
    wd['M'].max = change['new']
    if wd['M'].value > change['new']:
        wd['M'].value = change['new']
    update_param_count_preview(wd)

def on_M_change(change, wd):
    if change['new'] > wd['K'].value:
        wd['M'].value = wd['K'].value

def on_model_preset_change(change, wd):
    is_c = (change['new'] == 'Custom')
    wd['num_layers'].disabled = not is_c
    if is_c:
        update_layer_size_widgets(wd['num_layers'].value, wd)
    else:
        wd['layer_sizes_container'].children = []
    update_param_count_preview(wd)

def on_num_layers_change(change, wd):
    update_layer_size_widgets(change['new'], wd)
    update_param_count_preview(wd)

def on_compare_models_change(change, wd):
    """Dynamically enables/disables the comparison list based on checkbox."""
    wd['models_to_compare'].disabled = not change['new']
    if change['new']:
        with wd['status_output']:
            print("💡 Comparison Mode Active: Select multiple models from the list below.")

def on_multi_seed_change(change, wd):
    """Dynamically enables/disables the seed count slider."""
    wd['num_seeds'].disabled = not change['new']

def update_layer_size_widgets(n, wd):
    layer_widgets = [widgets.IntText(value=max(32, 512 // (2**i)),
                                    description=f'Layer {i+1}:',
                                    layout=widgets.Layout(width='200px')) for i in range(n)]
    for w in layer_widgets:
        w.observe(lambda c: update_param_count_preview(wd), names='value')
    wd['layer_sizes_container'].children = layer_widgets

def get_current_hidden_sizes(wd):
    if wd['model_preset'].value == 'Custom':
        return [w.value for w in wd['layer_sizes_container'].children]
    return get_model_architecture(wd['model_preset'].value)

def update_param_count_preview(wd):
    try:
        K = wd['K'].value; total = 0; prev = 2*K
        for s in get_current_hidden_sizes(wd):
            total += prev*s + s
            prev = s
        total += prev*K + K
        wd['param_count_display'].value = f"<b>Estimated Parameters:</b> {total:,}"
    except:
        pass

# ============================================================================
# EXPORT & SAVING LOGIC
# ============================================================================

def export_results_data(results, widgets_dict, fmt):
    """Handles data formatting and disk export for CSV, JSON, and LaTeX."""
    if results is None:
        with widgets_dict['status_output']: print("⚠️ No results found. Run an experiment first.")
        return

    out_dir = widgets_dict['output_dir'].value
    os.makedirs(out_dir, exist_ok=True)

    if isinstance(results, dict):
        rows = [{'Model': k, 'eta': v.evaluation.eta_top1, 'acc': v.evaluation.accuracy_top1} for k, v in results.items()]
    else:
        rows = [{'Model': widgets_dict['model_preset'].value, 'eta': results.evaluation.eta_top1, 'acc': results.evaluation.accuracy_top1}]

    df = pd.DataFrame(rows)
    ts = time.strftime('%Y%m%d_%H%M%S')

    try:
        if fmt == 'csv':
            df.to_csv(os.path.join(out_dir, f"results_{ts}.csv"), index=False)
        elif fmt == 'latex':
            df.to_latex(os.path.join(out_dir, f"results_{ts}.tex"), index=False)
        elif fmt == 'json':
            with open(os.path.join(out_dir, f"results_{ts}.json"), 'w') as f:
                json.dump(rows, f, indent=4)
        with widgets_dict['status_output']: print(f"✅ Successfully exported {fmt.upper()} to {out_dir}")
    except Exception as e:
        with widgets_dict['status_output']: print(f"❌ Export failed: {e}")

def save_ml_model(results, widgets_dict):
    """Saves the trained PyTorch model state."""
    if results is None or isinstance(results, dict):
        with widgets_dict['status_output']: print("⚠️ Cannot save model in comparison mode. Run a single experiment.")
        return

    out_dir = widgets_dict['output_dir'].value
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"model_{time.strftime('%Y%m%d_%H%M%S')}.pt")
    torch.save(results.model_state, path)
    with widgets_dict['status_output']: print(f"✅ Model saved to {path}")

# ============================================================================
# EXPERIMENT HANDLERS & ROUTING
# ============================================================================

def setup_experiment_handlers(wd):
    """Links buttons to core logic and progress callbacks."""
    def on_run(b):
        global CURRENT_RESULTS
        wd['status_output'].clear_output()
        wd['results_summary'].value = "<i>Running...</i>"
        wd['results_plots'].clear_output()

        with wd['status_output']:
            try:
                config = config_to_dict(wd)
                valid, errs = get_validation_errors(config)
                if not valid:
                    print("❌ Validation Errors:", *errs)
                    return

                def cb(e, t, m):
                    wd['progress_bar'].value = int((e/t)*100)
                    wd['live_metrics'].value = f"<b>Epoch {e}/{t}</b> | Loss: {m.get('val_loss',0):.4f} | η: {m.get('val_eta',0):.4f}"

                if config.get('compare_multiple_models'):
                    CURRENT_RESULTS = run_multi_model_comparison(config, list(config.get('models_to_compare')), cb)
                else:
                    CURRENT_RESULTS = run_single_experiment(config, cb)

                update_results_display(wd, CURRENT_RESULTS)
                print("✅ Experiment Completed Successfully!")
            except Exception as e:
                print(f"❌ Runtime Error: {e}")

    wd['button_run_experiment'].on_click(on_run)
    wd['button_save_config'].on_click(lambda b: save_config(config_to_dict(wd), f"configs/config_{time.strftime('%H%M%S')}.json"))
    wd['button_export_csv'].on_click(lambda b: export_results_data(CURRENT_RESULTS, wd, 'csv'))
    wd['button_export_json'].on_click(lambda b: export_results_data(CURRENT_RESULTS, wd, 'json'))
    wd['button_export_latex'].on_click(lambda b: export_results_data(CURRENT_RESULTS, wd, 'latex'))
    wd['button_save_model'].on_click(lambda b: save_ml_model(CURRENT_RESULTS, wd))

def update_results_display(wd, res):
    """Routes data to plotting registry with user notifications for mismatches."""
    if isinstance(res, dict):
        summary = "<h3>Comparison</h3><table border='1'><tr><th>Model</th><th>η</th><th>Accuracy</th></tr>"
        for k, v in res.items():
            summary += f"<tr><td>{k}</td><td>{v.evaluation.eta_top1:.4f}</td><td>{v.evaluation.accuracy_top1:.3f}</td></tr>"
        wd['results_summary'].value = summary + "</table>"
    else:
        wd['results_summary'].value = f"<h3>Results</h3><p><b>η_top1:</b> {res.evaluation.eta_top1:.4f}<br><b>Accuracy:</b> {res.evaluation.accuracy_top1:.3f}</p>"

    with wd['results_plots']:
        wd['results_plots'].clear_output()
        set_plot_style(wd['color_palette'].value)

        for p in wd['selected_plots'].value:
            try:
                # ROUTE: Comparison Results
                if isinstance(res, dict):
                    if p in ['violin', 'box', 'radar_chart', 'top_m_comparison', 'baseline_comparison', 'model_size_vs_performance']:
                        EXTENDED_PLOT_REGISTRY[p](res)
                    elif p == 'scatter':
                        EXTENDED_PLOT_REGISTRY[p](list(res.values()), list(res.keys()))
                    else:
                        print(f"ℹ️ Skipping '{p}': Requires a Single Model run for detailed metrics.")
                        continue

                # ROUTE: Single Experiment Results
                else:
                    if p in ['training_curves', 'learning_curve']:
                        EXTENDED_PLOT_REGISTRY[p](res.training_history)
                    elif p in ['heatmap', 'correlation_matrix']:
                        if hasattr(res, 'probe_bank'): EXTENDED_PLOT_REGISTRY[p](res.probe_bank)
                        else: print(f"⚠️ Skipping '{p}': Physical probe data not available.")
                    elif p in ['eta_distribution', 'cdf', 'baseline_comparison', 'error_analysis', 'top_m_comparison']:
                        EXTENDED_PLOT_REGISTRY[p](res.evaluation)
                    elif p == 'power_distribution':
                        if hasattr(res, 'powers_full'): EXTENDED_PLOT_REGISTRY[p](res.powers_full)
                    else:
                        print(f"ℹ️ Skipping '{p}': This plot is designed for 'Comparison' runs.")
                        continue

                plt.show()
            except Exception as e:
                print(f"❌ Plotting Error ({p}): {e}")

def setup_all_callbacks(wd):
    """Main setup for widget observation."""
    wd['phase_mode'].observe(lambda c: on_phase_mode_change(c, wd), names='value')
    wd['K'].observe(lambda c: on_K_change(c, wd), names='value')
    wd['M'].observe(lambda c: on_M_change(c, wd), names='value')
    wd['model_preset'].observe(lambda c: on_model_preset_change(c, wd), names='value')
    wd['num_layers'].observe(lambda c: on_num_layers_change(c, wd), names='value')

    # Enable/Disable observers for Tab 4
    wd['compare_multiple_models'].observe(lambda c: on_compare_models_change(c, wd), names='value')
    wd['multi_seed_runs'].observe(lambda c: on_multi_seed_change(c, wd), names='value')

    update_param_count_preview(wd)

def reset_to_defaults(wd):
    """Resets UI to standard research baseline."""
    wd['N'].value = 32; wd['K'].value = 64; wd['M'].value = 8
    wd['phase_mode'].value = 'continuous'; wd['probe_type'].value = 'continuous'
    wd['model_preset'].value = 'Baseline_MLP'; wd['dropout_prob'].value = 0.1
    wd['n_train'].value = 50000; wd['learning_rate'].value = 1e-3; wd['n_epochs'].value = 50
    wd['selected_plots'].value = ['training_curves', 'eta_distribution', 'top_m_comparison']
    with wd['status_output']:
        wd['status_output'].clear_output()
        print("✅ Settings reset to default.")