"""
Dashboard Callbacks for RIS PhD Ultimate Dashboard.
Optimized for robust routing and user notifications.
"""
import ipywidgets as widgets
import sys, os, time, json, torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_registry import get_model_architecture
from dashboard.config_manager import config_to_dict, save_config
from dashboard.validators import get_validation_errors
from dashboard.experiment_runner import run_single_experiment, run_multi_model_comparison, run_multi_seed_experiment, aggregate_results
from dashboard.plots import EXTENDED_PLOT_REGISTRY, set_plot_style

CURRENT_RESULTS = None

# UI LOGIC
def on_phase_mode_change(change, wd): wd['phase_bits'].disabled = (change['new'] != 'discrete')
def on_K_change(change, wd): 
    wd['M'].max = change['new']
    if wd['M'].value > change['new']: wd['M'].value = change['new']
    update_param_count_preview(wd)
def on_M_change(change, wd): 
    if change['new'] > wd['K'].value: wd['M'].value = wd['K'].value
def on_model_preset_change(change, wd):
    is_c = (change['new'] == 'Custom')
    wd['num_layers'].disabled = not is_c
    if is_c: update_layer_size_widgets(wd['num_layers'].value, wd)
    else: wd['layer_sizes_container'].children = []
    update_param_count_preview(wd)
def on_num_layers_change(change, wd): update_layer_size_widgets(change['new'], wd); update_param_count_preview(wd)

# ADDED LOGIC FOR TAB 4
def on_compare_models_change(change, wd): 
    wd['models_to_compare'].disabled = not change['new']
def on_multi_seed_change(change, wd): 
    wd['num_seeds'].disabled = not change['new']

def update_layer_size_widgets(n, wd):
    layer_widgets = [widgets.IntText(value=max(32, 512 // (2**i)), description=f'L{i+1}:', layout=widgets.Layout(width='200px')) for i in range(n)]
    for w in layer_widgets: w.observe(lambda c: update_param_count_preview(wd), names='value')
    wd['layer_sizes_container'].children = layer_widgets

def get_current_hidden_sizes(wd):
    if wd['model_preset'].value == 'Custom': return [w.value for w in wd['layer_sizes_container'].children]
    return get_model_architecture(wd['model_preset'].value)

def update_param_count_preview(wd):
    try:
        K = wd['K'].value; total = 0; prev = 2*K
        for s in get_current_hidden_sizes(wd): total += prev*s + s; prev = s
        total += prev*K + K
        wd['param_count_display'].value = f"<b>Params:</b> {total:,}"
    except: pass

# EXPORT LOGIC
def export_results_data(results, widgets_dict, fmt):
    if results is None: return
    out_dir = widgets_dict['output_dir'].value
    os.makedirs(out_dir, exist_ok=True)
    if isinstance(results, dict): rows = [{'Model': k, 'eta': v.evaluation.eta_top1, 'acc': v.evaluation.accuracy_top1} for k, v in results.items()]
    else: rows = [{'Model': widgets_dict['model_preset'].value, 'eta': results.evaluation.eta_top1, 'acc': results.evaluation.accuracy_top1}]
    df = pd.DataFrame(rows); ts = time.strftime('%Y%m%d_%H%M%S')
    if fmt == 'csv': df.to_csv(os.path.join(out_dir, f"res_{ts}.csv"), index=False)
    elif fmt == 'latex': df.to_latex(os.path.join(out_dir, f"res_{ts}.tex"), index=False)
    elif fmt == 'json':
        with open(os.path.join(out_dir, f"res_{ts}.json"), 'w') as f: json.dump(rows, f, indent=4)
    with widgets_dict['status_output']: print(f"✅ Exported {fmt.upper()}")

def save_ml_model(results, widgets_dict):
    if results is None or isinstance(results, dict): return
    out_dir = widgets_dict['output_dir'].value
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"model_{time.strftime('%Y%m%d_%H%M%S')}.pt")
    torch.save(results.model_state, path)
    with widgets_dict['status_output']: print(f"✅ Saved .pt")

# HANDLERS
def setup_experiment_handlers(wd):
    def on_run(b):
        global CURRENT_RESULTS
        wd['status_output'].clear_output(); wd['results_summary'].value = "<i>Running...</i>"; wd['results_plots'].clear_output()
        with wd['status_output']:
            try:
                config = config_to_dict(wd); valid, errs = get_validation_errors(config)
                if not valid: print("❌ Errors:", *errs); return
                def cb(e, t, m):
                    wd['progress_bar'].value = int((e/t)*100)
                    wd['live_metrics'].value = f"<b>Epoch {e}/{t}</b> | Loss: {m.get('val_loss',0):.4f} | η: {m.get('val_eta',0):.4f}"
                if config.get('compare_multiple_models'): CURRENT_RESULTS = run_multi_model_comparison(config, list(config.get('models_to_compare')), cb)
                else: CURRENT_RESULTS = run_single_experiment(config, cb)
                update_results_display(wd, CURRENT_RESULTS)
                print("✅ Done!")
            except Exception as e: print(f"❌ Error: {e}")
    wd['button_run_experiment'].on_click(on_run)
    wd['button_save_config'].on_click(lambda b: save_config(config_to_dict(wd), f"configs/c_{time.strftime('%H%M%S')}.json"))
    wd['button_export_csv'].on_click(lambda b: export_results_data(CURRENT_RESULTS, wd, 'csv'))
    wd['button_export_json'].on_click(lambda b: export_results_data(CURRENT_RESULTS, wd, 'json'))
    wd['button_export_latex'].on_click(lambda b: export_results_data(CURRENT_RESULTS, wd, 'latex'))
    wd['button_save_model'].on_click(lambda b: save_ml_model(CURRENT_RESULTS, wd))

def update_results_display(wd, res):
    if isinstance(res, dict):
        summary = "<h3>Comparison</h3><table border='1'><tr><th>Model</th><th>η</th></tr>"
        for k, v in res.items(): summary += f"<tr><td>{k}</td><td>{v.evaluation.eta_top1:.4f}</td></tr>"
        wd['results_summary'].value = summary + "</table>"
    else: wd['results_summary'].value = f"<h3>Results</h3><p>η: {res.evaluation.eta_top1:.4f}</p>"
    with wd['results_plots']:
        wd['results_plots'].clear_output(); set_plot_style(wd['color_palette'].value)
        for p in wd['selected_plots'].value:
            try:
                if isinstance(res, dict):
                    if p in ['top_m_comparison', 'baseline_comparison']: EXTENDED_PLOT_REGISTRY[p](res)
                    continue
                data = res.training_history if p == 'training_curves' else res.evaluation
                EXTENDED_PLOT_REGISTRY[p](data); plt.show()
            except Exception as e: print(f"Plot Error ({p}): {e}")

def setup_all_callbacks(wd):
    wd['phase_mode'].observe(lambda c: on_phase_mode_change(c, wd), names='value')
    wd['K'].observe(lambda c: on_K_change(c, wd), names='value')
    wd['M'].observe(lambda c: on_M_change(c, wd), names='value')
    wd['model_preset'].observe(lambda c: on_model_preset_change(c, wd), names='value')
    wd['num_layers'].observe(lambda c: on_num_layers_change(c, wd), names='value')
    # ATTACH TAB 4 OBSERVERS
    wd['compare_multiple_models'].observe(lambda c: on_compare_models_change(c, wd), names='value')
    wd['multi_seed_runs'].observe(lambda c: on_multi_seed_change(c, wd), names='value')
    update_param_count_preview(wd)

def reset_to_defaults(wd):
    wd['N'].value = 32; wd['K'].value = 64; wd['M'].value = 8; wd['phase_mode'].value = 'continuous'; wd['probe_type'].value = 'continuous'
    wd['model_preset'].value = 'Baseline_MLP'; wd['dropout_prob'].value = 0.1; wd['use_batch_norm'].value = True
    wd['n_train'].value = 50000; wd['learning_rate'].value = 1e-3; wd['n_epochs'].value = 50
    wd['selected_plots'].value = ['training_curves', 'eta_distribution', 'top_m_comparison']
    with wd['status_output']: wd['status_output'].clear_output(); print("✅ Reset")
