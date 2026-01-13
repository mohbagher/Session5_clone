"""
Dashboard Callbacks - Fixed for 'progress_callback' error and Sweep Support.
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
from dashboard.experiment_runner import (
    run_single_experiment,
    run_multi_model_comparison,
    run_multi_seed_experiment
)
from dashboard.plots import EXTENDED_PLOT_REGISTRY, set_plot_style

# --- GLOBAL STATE ---
CURRENT_RESULTS = None
EXPERIMENT_STACK = []
STACK_RESULTS = []

# ... (Keep existing UI Logic: on_phase_mode_change, on_K_change, etc.) ...
def on_phase_mode_change(change, wd): wd['phase_bits'].disabled = (change['new'] != 'discrete')
def on_K_change(change, wd):
    wd['M'].max = change['new'];
    if wd['M'].value > change['new']: wd['M'].value = change['new']
    update_param_count_preview(wd)
def on_M_change(change, wd):
    if change['new'] > wd['K'].value: wd['M'].value = wd['K'].value
def on_probe_category_change(change, wd):
    if change['new'] == 'Physics-Based':
        wd['probe_type'].options = ['continuous', 'binary', '2bit']
        wd['probe_type'].value = 'continuous'
    else:
        wd['probe_type'].options = ['hadamard', 'sobol', 'halton']
        wd['probe_type'].value = 'hadamard'
def on_model_preset_change(change, wd):
    is_c = (change['new'] == 'Custom')
    wd['num_layers'].disabled = not is_c
    if is_c: update_layer_size_widgets(wd['num_layers'].value, wd)
    else: wd['layer_sizes_container'].children = []
    update_param_count_preview(wd)
def on_num_layers_change(change, wd): update_layer_size_widgets(change['new'], wd); update_param_count_preview(wd)
def on_compare_models_change(change, wd): wd['models_to_compare'].disabled = not change['new']
def on_multi_seed_change(change, wd): wd['num_seeds'].disabled = not change['new']

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

# ... (Keep Stack Logic: on_add_to_stack, on_clear_stack) ...
def on_add_to_stack(b, wd):
    config = config_to_dict(wd)
    idx = len(EXPERIMENT_STACK) + 1
    cat = wd['probe_category'].value.split(' ')[0]
    name = f"#{idx} [{config['model_preset']}] {cat}:{config['probe_type']} (M={config['M']})" # Added M for sweep visibility
    config['experiment_name'] = name
    config['stack_index'] = idx - 1
    EXPERIMENT_STACK.append(config)
    wd['stack_display'].options = [c['experiment_name'] for c in EXPERIMENT_STACK]
    wd['transfer_source'].options = ['None'] + [f"Exp #{i+1}" for i in range(len(EXPERIMENT_STACK))]
    wd['transfer_source'].disabled = False
    with wd['status_output']: print(f"✅ Added: {name}")

def on_clear_stack(b, wd):
    global EXPERIMENT_STACK, STACK_RESULTS
    EXPERIMENT_STACK = []
    STACK_RESULTS = []
    wd['stack_display'].options = []
    wd['transfer_source'].options = ['None']; wd['transfer_source'].value = 'None'; wd['transfer_source'].disabled = True
    with wd['status_output']: print("🗑️ Stack cleared.")

def on_run_stack(b, wd):
    global STACK_RESULTS, CURRENT_RESULTS
    if not EXPERIMENT_STACK:
        with wd['status_output']: print("⚠️ Stack is empty."); return

    wd['status_output'].clear_output(); STACK_RESULTS = []
    with wd['status_output']:
        print(f"🚀 Running {len(EXPERIMENT_STACK)} Experiments...")
        for i, config in enumerate(EXPERIMENT_STACK):
            print(f"\n▶️ Running {config['experiment_name']}...")
            initial_weights = None
            if config.get('transfer_source', 'None') != 'None':
                try:
                    src_idx = int(config['transfer_source'].split('#')[1]) - 1
                    if src_idx < len(STACK_RESULTS):
                        print(f"   ↪ Init weights from Exp #{src_idx+1}...")
                        initial_weights = STACK_RESULTS[src_idx].model_state
                except: pass

            try:
                def cb(e, t, m):
                    wd['progress_bar'].value = int((e/t)*100)
                    wd['live_metrics'].value = f"<b>{config['experiment_name']}</b><br>Epoch {e}/{t} | Loss: {m.get('val_loss',0):.4f}"

                # FIX: Keyword must be 'progress_callback' NOT 'callback'
                res = run_single_experiment(config, progress_callback=cb, initial_weights=initial_weights)

                STACK_RESULTS.append(res); print(f"   ✅ Finished.")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                import traceback; traceback.print_exc() # Show full error for debugging

        CURRENT_RESULTS = {f"#{i+1}": res for i, res in enumerate(STACK_RESULTS)}
        update_results_display(wd, CURRENT_RESULTS)
        print("\n🏁 Stack Complete!")

# ... (Runner & Routing) ...
def setup_experiment_handlers(wd):
    wd['button_add_to_stack'].on_click(lambda b: on_add_to_stack(b, wd))
    wd['button_clear_stack'].on_click(lambda b: on_clear_stack(b, wd))
    wd['button_run_stack'].on_click(lambda b: on_run_stack(b, wd))

    def on_run_single(b):
        global CURRENT_RESULTS
        wd['status_output'].clear_output(); wd['results_summary'].value = "<i>Running...</i>"
        with wd['status_output']:
            try:
                config = config_to_dict(wd)
                def cb(e, t, m): wd['progress_bar'].value = int((e/t)*100); wd['live_metrics'].value = f"Epoch {e}/{t} | Loss: {m.get('val_loss',0):.4f}"
                if config.get('compare_multiple_models'): CURRENT_RESULTS = run_multi_model_comparison(config, list(config.get('models_to_compare')), cb)
                else: CURRENT_RESULTS = run_single_experiment(config, progress_callback=cb) # FIX HERE TOO
                update_results_display(wd, CURRENT_RESULTS)
                print("✅ Done!")
            except Exception as e: print(f"❌ Error: {e}")

    wd['button_run_experiment'].on_click(on_run_single)
    wd['selected_plots'].observe(lambda c: update_results_display(wd, CURRENT_RESULTS), names='value')
    wd['button_export_csv'].on_click(lambda b: export_results_data(CURRENT_RESULTS, wd, 'csv'))
    wd['button_export_json'].on_click(lambda b: export_results_data(CURRENT_RESULTS, wd, 'json'))
    wd['button_export_latex'].on_click(lambda b: export_results_data(CURRENT_RESULTS, wd, 'latex'))
    wd['button_save_model'].on_click(lambda b: save_ml_model(CURRENT_RESULTS, wd))

def update_results_display(wd, res):
    if res is None: return

    # Summary Table
    if isinstance(res, dict):
        summary = "<h3>Stack Results</h3><table border='1'><tr><th>Exp</th><th>M</th><th>Probe</th><th>η</th><th>Acc</th></tr>"
        for k, v in res.items():
            summary += f"<tr><td>{k}</td><td>{v.config['M']}</td><td>{v.config['probe_type']}</td><td>{v.evaluation.eta_top1:.4f}</td><td>{v.evaluation.accuracy_top1:.3f}</td></tr>"
        wd['results_summary'].value = summary + "</table>"
    else:
        wd['results_summary'].value = f"<h3>Results</h3><p><b>η:</b> {res.evaluation.eta_top1:.4f} | <b>Acc:</b> {res.evaluation.accuracy_top1:.3f}</p>"

    wd['results_plots_training'].clear_output()
    wd['results_plots_analysis'].clear_output()
    TRAINING_PLOTS = ['training_curves', 'learning_curve', 'convergence_analysis']
    set_plot_style(wd['color_palette'].value)

    for p in wd['selected_plots'].value:
        target_out = wd['results_plots_training'] if p in TRAINING_PLOTS else wd['results_plots_analysis']
        with target_out:
            try:
                if isinstance(res, dict):
                    # --- SWEEP PLOT LOGIC ---
                    if p in ['eta_vs_M', 'eta_vs_K', 'eta_vs_N']:
                        # Extract sweep data from Stack Results
                        EXTENDED_PLOT_REGISTRY[p](res) # Now passing the full dict to updated plots.py
                    elif p in ['violin', 'box', 'radar_chart', 'top_m_comparison', 'baseline_comparison']:
                        EXTENDED_PLOT_REGISTRY[p](res)
                    elif p == 'scatter':
                        EXTENDED_PLOT_REGISTRY[p](list(res.values()), list(res.keys()))
                    else: print(f"ℹ️ {p}: Single-model only.")
                else:
                    if p in ['training_curves', 'learning_curve']: EXTENDED_PLOT_REGISTRY[p](res.training_history)
                    elif p in ['heatmap', 'correlation_matrix']:
                        if hasattr(res, 'probe_bank'): EXTENDED_PLOT_REGISTRY[p](res.probe_bank)
                    elif p in ['eta_distribution', 'cdf', 'baseline_comparison', 'top_m_comparison']: EXTENDED_PLOT_REGISTRY[p](res.evaluation)
                    else: print(f"ℹ️ {p}: Comparison only.")
                plt.show()
            except Exception as e: print(f"❌ {p}: {e}")

def setup_all_callbacks(wd):
    wd['phase_mode'].observe(lambda c: on_phase_mode_change(c, wd), names='value')
    wd['K'].observe(lambda c: on_K_change(c, wd), names='value')
    wd['M'].observe(lambda c: on_M_change(c, wd), names='value')
    wd['probe_category'].observe(lambda c: on_probe_category_change(c, wd), names='value')
    wd['model_preset'].observe(lambda c: on_model_preset_change(c, wd), names='value')
    wd['num_layers'].observe(lambda c: on_num_layers_change(c, wd), names='value')
    wd['compare_multiple_models'].observe(lambda c: on_compare_models_change(c, wd), names='value')
    wd['multi_seed_runs'].observe(lambda c: on_multi_seed_change(c, wd), names='value')
    update_param_count_preview(wd)
    on_probe_category_change({'new': 'Physics-Based'}, wd)

# Export functions identical to before...
def export_results_data(results, widgets_dict, fmt):
    if results is None: return
    out_dir = widgets_dict['output_dir'].value; os.makedirs(out_dir, exist_ok=True)
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
    path = os.path.join(widgets_dict['output_dir'].value, f"model_{time.strftime('%Y%m%d_%H%M%S')}.pt")
    torch.save(results.model_state, path)
    with widgets_dict['status_output']: print(f"✅ Saved model to {path}")

def reset_to_defaults(wd):
    wd['N'].value = 32; wd['K'].value = 64; wd['M'].value = 8; wd['phase_mode'].value = 'continuous'; wd['probe_category'].value = 'Physics-Based'
    wd['model_preset'].value = 'Baseline_MLP'
    with wd['status_output']: print("✅ Reset")