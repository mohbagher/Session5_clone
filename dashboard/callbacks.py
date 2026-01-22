"""
RIS Dashboard Callbacks V2.0 - Complete Professional Edition
============================================================
ALL 7 FEATURES INTEGRATED:
✅ Smart caching (skip re-training)
✅ Named sessions
✅ Full state restore
✅ Auto-save everything
✅ Custom names everywhere
✅ Interactive plots
✅ Inline file browsers
"""

import ipywidgets as widgets
import sys, os, time, json, torch, matplotlib.pyplot as plt, pandas as pd, numpy as np
from datetime import datetime
from pathlib import Path
import pickle, hashlib
from IPython.display import display, clear_output
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model_registry import get_model_architecture
from dashboard.config_manager import config_to_dict, save_config, load_config, dict_to_widgets
from dashboard.experiment_runner import run_single_experiment, run_multi_model_comparison, run_multi_seed_experiment
from dashboard.plots import EXTENDED_PLOT_REGISTRY, set_plot_style
# === GLOBALS ===
CURRENT_RESULTS, EXPERIMENT_STACK, STACK_RESULTS = None, [], []
TRAINED_CACHE, SESSION_PATHS, SESSION_NAME = {}, None, None
PAUSE_REQUESTED, PLOT_UPDATE_ENABLED = False, True

# === CACHING ===
def compute_hash(cfg):
    """
    Compute hash for experiment caching - FULLY DYNAMIC.

    Automatically includes ALL parameters from config, excluding only:
    - UI/display parameters that don't affect results
    - Output/export parameters
    - Visualization parameters

    This makes the system automatically expandable - add any new parameter
    to the config and it will be included in the hash without code changes.
    """

    # Parameters to EXCLUDE from hash (UI/display only, don't affect training)
    EXCLUDE_KEYS = {
        # Visualization/output (don't affect experiment results)
        'selected_plots',
        'figure_format',
        'dpi',
        'color_palette',
        'save_plots',
        'output_dir',

        # Evaluation display options (don't affect training)
        'top_m_values',  # These affect WHICH metrics are computed, but not the model itself
        'compare_multiple_models',
        'models_to_compare',
        'multi_seed_runs',
        'num_seeds',
        'compute_confidence_intervals',

        # Stack/session management (metadata, not parameters)
        'experiment_name',
        'stack_index',
        'custom_exp_name',

        # UI state (not experiment parameters)
        'status_output',
        'results_summary',
        'progress_bar',
        'live_metrics',
    }

    # Build hash dictionary by INCLUDING everything except excluded keys
    rel = {}

    for key, value in cfg.items():
        # Skip excluded keys
        if key in EXCLUDE_KEYS:
            continue

        # Skip None values (missing parameters)
        if value is None:
            continue

        # Convert unhashable types to hashable equivalents
        if isinstance(value, list):
            value = tuple(value)  # Lists → tuples
        elif isinstance(value, dict):
            value = tuple(sorted(value.items()))  # Dicts → sorted tuples
        elif isinstance(value, set):
            value = tuple(sorted(value))  # Sets → sorted tuples

        # Skip widget objects or other non-serializable types
        try:
            json.dumps(value)  # Test if serializable
            rel[key] = value
        except (TypeError, ValueError):
            # Skip non-serializable objects (likely widgets)
            continue

    # Create deterministic hash
    try:
        hash_str = json.dumps(rel, sort_keys=True)
        return hashlib.md5(hash_str.encode()).hexdigest()
    except Exception as e:
        # Fallback: if something goes wrong, generate random hash (forces re-run)
        import random
        return hashlib.md5(str(random.random()).encode()).hexdigest()

def is_cacheable(cfg): return cfg.get('transfer_source', 'None') in ['None', None]
def get_cached(cfg): return TRAINED_CACHE.get(compute_hash(cfg)) if is_cacheable(cfg) else None
def cache(cfg, res):
    if is_cacheable(cfg): TRAINED_CACHE[compute_hash(cfg)] = res

# === SESSION MANAGEMENT ===
def prompt_session_name(wd):
    global SESSION_NAME
    if SESSION_NAME: return SESSION_NAME
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    inp = widgets.Text(value='', placeholder='Session name', description='Name:', layout=widgets.Layout(width='300px'))
    btn = widgets.Button(description='OK', button_style='success')
    res = {'n': None}
    def ok(b):
        n = inp.value.strip()
        res['n'] = f"{''.join(c if c.isalnum() or c in '_-' else '_' for c in n)}_{ts}" if n else f"session_{ts}"
        inp.disabled = btn.disabled = True
    btn.on_click(ok)
    with wd['status_output']:
        print("📝 Session name (optional):"); display(widgets.HBox([inp, btn]))
    for _ in range(50):
        if res['n']: break
        time.sleep(0.1)
    SESSION_NAME = res['n'] or f"session_{ts}"
    return SESSION_NAME

def get_paths(base='results', name=None):
    name = name or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    d = Path(base) / name
    p = {k: str(d/k) for k in ['configs','stacks','models','plots','exports','checkpoints','saved_results']}
    p.update({'base': str(d), 'state': str(d/'session_state.pkl')})
    for k, v in p.items():
        if k != 'state': Path(v).mkdir(parents=True, exist_ok=True)
    return p

def ensure_paths(wd):
    global SESSION_PATHS, SESSION_NAME
    if not SESSION_PATHS:
        SESSION_NAME = SESSION_NAME or prompt_session_name(wd)
        SESSION_PATHS = get_paths(wd['output_dir'].value.rstrip('/'), SESSION_NAME)
        with wd['status_output']:
            clear_output(wait=True); print(f"📁 {SESSION_NAME}")
    return SESSION_PATHS

# === STATE PERSISTENCE ===
def save_state(paths):
    try:
        state = {'session_name': SESSION_NAME, 'stack': EXPERIMENT_STACK, 'results': STACK_RESULTS,
                'cache': TRAINED_CACHE, 'timestamp': datetime.now().isoformat()}
        with open(paths['state'], 'wb') as f: pickle.dump(state, f)
        # Auto-save stack JSON
        with open(Path(paths['stacks'])/f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump({'stack': EXPERIMENT_STACK, 'count': len(EXPERIMENT_STACK)}, f, indent=2)
        return True
    except Exception as e: print(f"⚠️ Save failed: {e}"); return False


# ============================================================================
# SESSION STATE MANAGEMENT - LOAD FUNCTIONS
# ============================================================================

def load_state(filepath, wd):
    """
    Load complete session state and restore everything.

    This function restores:
    - Experiment stack
    - All completed results
    - Trained model cache
    - Session metadata

    Args:
        filepath: Path to session_state.pkl file
        wd: Widget dictionary containing all dashboard widgets

    Returns:
        bool: True if successful, False otherwise
    """
    global EXPERIMENT_STACK, STACK_RESULTS, TRAINED_CACHE, SESSION_NAME, CURRENT_RESULTS

    try:
        # Show loading message
        with wd['status_output']:
            clear_output(wait=True)
            print(f"📂 Loading session state from: {filepath}")
            print("   Please wait...")

        # Load pickled state
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Restore global state variables
        SESSION_NAME = state.get('session_name', 'Unknown')
        EXPERIMENT_STACK = state.get('stack', [])
        STACK_RESULTS = state.get('results', [])
        TRAINED_CACHE = state.get('cache', {})

        # Update stack display widget
        if EXPERIMENT_STACK:
            wd['stack_display'].options = [
                c.get('experiment_name', f"Exp #{i + 1}")
                for i, c in enumerate(EXPERIMENT_STACK)
            ]
            with wd['status_output']:
                print(f"   ✓ Loaded {len(EXPERIMENT_STACK)} experiments in stack")

        # CRITICAL FIX: Properly restore and display results
        if STACK_RESULTS:
            # Rebuild CURRENT_RESULTS dictionary from loaded results
            CURRENT_RESULTS = {}
            for i, res in enumerate(STACK_RESULTS):
                if hasattr(res, 'config'):
                    exp_name = res.config.get('experiment_name', f"Exp #{i + 1}")
                else:
                    exp_name = f"Exp #{i + 1}"
                CURRENT_RESULTS[exp_name] = res

            with wd['status_output']:
                print(f"   ✓ Loaded {len(STACK_RESULTS)} completed results")

            # CRITICAL: Force update the results display
            try:
                # Clear existing display first
                wd['results_summary'].value = "<h3>⏳ Loading results...</h3>"

                # Call the update function to refresh all visualizations
                update_results_display(wd, CURRENT_RESULTS)

                with wd['status_output']:
                    print(f"   ✓ Results display updated successfully")

            except Exception as e:
                with wd['status_output']:
                    print(f"   ⚠️ Display update warning: {e}")
                    import traceback
                    traceback.print_exc()

        # Display final success message
        with wd['status_output']:
            clear_output(wait=True)
            print("=" * 70)
            print(f"✅ SESSION RESTORED: {SESSION_NAME}")
            print("=" * 70)
            print(f"   Experiments in stack: {len(EXPERIMENT_STACK)}")
            print(f"   Completed results: {len(STACK_RESULTS)}")
            print(f"   Cached experiments: {len(TRAINED_CACHE)}")
            print(f"   Timestamp: {state.get('timestamp', 'Unknown')}")
            print("=" * 70)
            print()
            print("📊 Check the 'Results & Analysis Dashboard' tab below!")
            print("   All plots, tables, and analysis should now be visible.")
            print()
            if STACK_RESULTS:
                print(f"💡 Loaded experiments:")
                for i, res in enumerate(STACK_RESULTS[:5]):  # Show first 5
                    name = res.config.get('experiment_name', f"Exp #{i + 1}") if hasattr(res,
                                                                                         'config') else f"Exp #{i + 1}"
                    print(f"   {i + 1}. {name}")
                if len(STACK_RESULTS) > 5:
                    print(f"   ... and {len(STACK_RESULTS) - 5} more")
            print()

        return True

    except FileNotFoundError:
        with wd['status_output']:
            clear_output(wait=True)
            print(f"❌ File not found: {filepath}")
        return False

    except Exception as e:
        with wd['status_output']:
            clear_output(wait=True)
            print(f"❌ Failed to load session state: {e}")
            print()
            print("Detailed error:")
            import traceback
            traceback.print_exc()
        return False


def on_load_results_browser(b, wd):
    """
    Browse and load saved results with improved display refresh.

    Provides a file browser interface to select and load either:
    - session_state.pkl: Complete session with stack, results, and cache
    - results_*.pkl: Individual experiment results

    Args:
        b: Button widget that triggered this callback
        wd: Widget dictionary containing all dashboard widgets
    """
    base = wd['output_dir'].value

    # Search for saved result files
    files = []
    for r, _, fs in os.walk(base):
        for f in fs:
            if (f.startswith('results_') and f.endswith('.pkl')) or f == 'session_state.pkl':
                full_path = os.path.join(r, f)
                rel_path = os.path.relpath(full_path, base)
                # Prioritize session_state files (priority 0)
                priority = 0 if f == 'session_state.pkl' else 1
                mod_time = os.path.getmtime(full_path)
                files.append((priority, rel_path, full_path, mod_time))

    if not files:
        with wd['status_output']:
            clear_output(wait=True)
            print("⚠️ No saved results found in output directory")
            print(f"   Searched in: {base}")
            print()
            print("💡 Run an experiment first, then save results using:")
            print("   - 'Save Results' button")
            print("   - Auto-save will create session_state.pkl")
        return

    # Sort by priority (session_state first), then by date (newest first)
    files.sort(key=lambda x: (x[0], -x[3]))

    # Create file options with labels
    file_options = []
    for priority, rel_name, full_path, mod_time in files:
        file_type = "SESSION" if 'session_state' in rel_name else "RESULTS"
        # Format modification time
        mod_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M')
        label = f"[{file_type}] {rel_name} ({mod_str})"
        file_options.append((label, full_path))

    # Create file selector widget
    sel = widgets.Dropdown(
        options=file_options,
        description='File:',
        layout=widgets.Layout(width='700px'),
        style={'description_width': '50px'}
    )

    # Create load button
    btn = widgets.Button(
        description='Load & Restore',
        button_style='success',
        layout=widgets.Layout(width='180px'),
        icon='upload',
        tooltip='Load selected file and restore results'
    )

    # Info text
    info = widgets.HTML(
        value="""
        <div style='padding: 10px; background-color: #e8f4f8; border-left: 4px solid #2196F3; margin-bottom: 10px;'>
            <b>💡 Tip:</b> <code>session_state.pkl</code> restores <b>EVERYTHING</b>: 
            <ul style='margin: 5px 0;'>
                <li>Experiment stack (configurations)</li>
                <li>All completed results</li>
                <li>Trained model cache</li>
                <li>All plots and visualizations</li>
            </ul>
        </div>
        """
    )

    def load(b):
        """Handle load button click."""
        filepath = sel.value
        btn.disabled = True
        btn.description = 'Loading...'
        btn.icon = 'hourglass'

        try:
            with wd['status_output']:
                clear_output(wait=True)
                print(f"📂 Loading from: {os.path.basename(filepath)}")
                print(f"   Full path: {filepath}")

            # Check if file is session state or results
            if filepath.endswith('session_state.pkl'):
                # Load complete session state
                success = load_state(filepath, wd)  # ← This now works because load_state is defined above
                if not success:
                    raise Exception("Session state loading failed - see error above")

            else:
                # Load individual results file
                with wd['status_output']:
                    print("   Loading results file...")

                with open(filepath, 'rb') as f:
                    loaded = pickle.load(f)

                global STACK_RESULTS, CURRENT_RESULTS

                # Handle different result formats
                if isinstance(loaded, list):
                    # List of experiment results
                    STACK_RESULTS = loaded
                    CURRENT_RESULTS = {
                        res.config.get('experiment_name', f"Exp #{i + 1}"): res
                        for i, res in enumerate(STACK_RESULTS)
                    }
                elif isinstance(loaded, dict):
                    # Dictionary of results
                    CURRENT_RESULTS = loaded
                    STACK_RESULTS = list(loaded.values())
                else:
                    # Single result object
                    CURRENT_RESULTS = {'Loaded Result': loaded}
                    STACK_RESULTS = [loaded]

                # CRITICAL: Update the results display
                with wd['status_output']:
                    print("   Updating visualization...")

                update_results_display(wd, CURRENT_RESULTS)

                # Success message
                with wd['status_output']:
                    clear_output(wait=True)
                    print("=" * 70)
                    print(f"✅ Results loaded successfully!")
                    print("=" * 70)
                    print(f"   File: {os.path.basename(filepath)}")
                    print(f"   Experiments loaded: {len(STACK_RESULTS)}")
                    print("=" * 70)
                    print()
                    print("📊 Check the 'Results & Analysis Dashboard' tab below!")
                    print("   All plots and tables should now be visible.")
                    print()

        except FileNotFoundError:
            with wd['status_output']:
                clear_output(wait=True)
                print(f"❌ File not found: {filepath}")
                print("   The file may have been moved or deleted.")

        except Exception as e:
            with wd['status_output']:
                clear_output(wait=True)
                print(f"❌ Load failed: {e}")
                print()
                print("Detailed error information:")
                import traceback
                traceback.print_exc()
                print()
                print("💡 Troubleshooting tips:")
                print("   1. Check if the file is corrupted")
                print("   2. Ensure it was saved with the same code version")
                print("   3. Try loading a different file")

        finally:
            # Re-enable button
            btn.disabled = False
            btn.description = 'Load & Restore'
            btn.icon = 'upload'

    # Connect button to load function
    btn.on_click(load)

    # Display file browser interface
    with wd['status_output']:
        clear_output(wait=True)
        print("=" * 70)
        print("📁 LOAD SAVED RESULTS")
        print("=" * 70)
        print()
        display(widgets.VBox([
            info,
            widgets.HBox([sel, btn], layout=widgets.Layout(margin='10px 0'))
        ]))
        print()
        print(f"Found {len(files)} saved file(s) in: {base}")
        print()

def on_load_results_browser(b, wd):
    """
    Browse and load saved results with improved display refresh.

    Provides a file browser interface to select and load either:
    - session_state.pkl: Complete session with stack, results, and cache
    - results_*.pkl: Individual experiment results

    Args:
        b: Button widget that triggered this callback
        wd: Widget dictionary containing all dashboard widgets
    """
    base = wd['output_dir'].value

    # Search for saved result files
    files = []
    for r, _, fs in os.walk(base):
        for f in fs:
            if (f.startswith('results_') and f.endswith('.pkl')) or f == 'session_state.pkl':
                full_path = os.path.join(r, f)
                rel_path = os.path.relpath(full_path, base)
                # Prioritize session_state files (priority 0)
                priority = 0 if f == 'session_state.pkl' else 1
                mod_time = os.path.getmtime(full_path)
                files.append((priority, rel_path, full_path, mod_time))

    if not files:
        with wd['status_output']:
            clear_output(wait=True)
            print("⚠️ No saved results found in output directory")
            print(f"   Searched in: {base}")
            print()
            print("💡 Run an experiment first, then save results using:")
            print("   - 'Save Results' button")
            print("   - Auto-save will create session_state.pkl")
        return

    # Sort by priority (session_state first), then by date (newest first)
    files.sort(key=lambda x: (x[0], -x[3]))

    # Create file options with labels
    file_options = []
    for priority, rel_name, full_path, mod_time in files:
        file_type = "SESSION" if 'session_state' in rel_name else "RESULTS"
        # Format modification time
        mod_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M')
        label = f"[{file_type}] {rel_name} ({mod_str})"
        file_options.append((label, full_path))

    # Create file selector widget
    sel = widgets.Dropdown(
        options=file_options,
        description='File:',
        layout=widgets.Layout(width='700px'),
        style={'description_width': '50px'}
    )

    # Create load button
    btn = widgets.Button(
        description='Load & Restore',
        button_style='success',
        layout=widgets.Layout(width='180px'),
        icon='upload',
        tooltip='Load selected file and restore results'
    )

    # Info text
    info = widgets.HTML(
        value="""
        <div style='padding: 10px; background-color: #e8f4f8; border-left: 4px solid #2196F3; margin-bottom: 10px;'>
            <b>💡 Tip:</b> <code>session_state.pkl</code> restores <b>EVERYTHING</b>: 
            <ul style='margin: 5px 0;'>
                <li>Experiment stack (configurations)</li>
                <li>All completed results</li>
                <li>Trained model cache</li>
                <li>All plots and visualizations</li>
            </ul>
        </div>
        """
    )

    def load(b):
        """Handle load button click."""
        filepath = sel.value
        btn.disabled = True
        btn.description = 'Loading...'
        btn.icon = 'hourglass'

        try:
            with wd['status_output']:
                clear_output(wait=True)
                print(f"📂 Loading from: {os.path.basename(filepath)}")
                print(f"   Full path: {filepath}")

            # Check if file is session state or results
            if filepath.endswith('session_state.pkl'):
                # Load complete session state
                success = load_state(filepath, wd)
                if not success:
                    raise Exception("Session state loading failed - see error above")

            else:
                # Load individual results file
                with wd['status_output']:
                    print("   Loading results file...")

                with open(filepath, 'rb') as f:
                    loaded = pickle.load(f)

                global STACK_RESULTS, CURRENT_RESULTS

                # Handle different result formats
                if isinstance(loaded, list):
                    # List of experiment results
                    STACK_RESULTS = loaded
                    CURRENT_RESULTS = {
                        res.config.get('experiment_name', f"Exp #{i + 1}"): res
                        for i, res in enumerate(STACK_RESULTS)
                    }
                elif isinstance(loaded, dict):
                    # Dictionary of results
                    CURRENT_RESULTS = loaded
                    STACK_RESULTS = list(loaded.values())
                else:
                    # Single result object
                    CURRENT_RESULTS = {'Loaded Result': loaded}
                    STACK_RESULTS = [loaded]

                # CRITICAL: Update the results display
                with wd['status_output']:
                    print("   Updating visualization...")

                update_results_display(wd, CURRENT_RESULTS)

                # Success message
                with wd['status_output']:
                    clear_output(wait=True)
                    print("=" * 70)
                    print(f"✅ Results loaded successfully!")
                    print("=" * 70)
                    print(f"   File: {os.path.basename(filepath)}")
                    print(f"   Experiments loaded: {len(STACK_RESULTS)}")
                    print("=" * 70)
                    print()
                    print("📊 Check the 'Results & Analysis Dashboard' tab below!")
                    print("   All plots and tables should now be visible.")
                    print()

        except FileNotFoundError:
            with wd['status_output']:
                clear_output(wait=True)
                print(f"❌ File not found: {filepath}")
                print("   The file may have been moved or deleted.")

        except Exception as e:
            with wd['status_output']:
                clear_output(wait=True)
                print(f"❌ Load failed: {e}")
                print()
                print("Detailed error information:")
                import traceback
                traceback.print_exc()
                print()
                print("💡 Troubleshooting tips:")
                print("   1. Check if the file is corrupted")
                print("   2. Ensure it was saved with the same code version")
                print("   3. Try loading a different file")

        finally:
            # Re-enable button
            btn.disabled = False
            btn.description = 'Load & Restore'
            btn.icon = 'upload'

    # Connect button to load function
    btn.on_click(load)

    # Display file browser interface
    with wd['status_output']:
        clear_output(wait=True)
        print("=" * 70)
        print("📁 LOAD SAVED RESULTS")
        print("=" * 70)
        print()
        display(widgets.VBox([
            info,
            widgets.HBox([sel, btn], layout=widgets.Layout(margin='10px 0'))
        ]))
        print()
        print(f"Found {len(files)} saved file(s) in: {base}")
        print()

# === UI CALLBACKS ===
def on_phase_mode_change(c, wd): wd['phase_bits'].disabled = (c['new'] != 'discrete')
def on_K_change(c, wd): wd['M'].max = c['new']; wd['M'].value = min(wd['M'].value, c['new']); update_param_count_preview(wd)
def on_M_change(c, wd): wd['M'].value = min(c['new'], wd['K'].value)
def on_probe_category_change(c, wd):
    opts = ['continuous', 'binary', '2bit'] if c['new'] == 'Physics-Based' else ['hadamard', 'sobol', 'halton']
    wd['probe_type'].options = opts
    if wd['probe_type'].value not in opts: wd['probe_type'].value = opts[0]
def on_model_preset_change(c, wd):
    is_custom = (c['new'] == 'Custom')
    wd['num_layers'].disabled = not is_custom
    if is_custom: update_layer_size_widgets(wd['num_layers'].value, wd)
    else: wd['layer_sizes_container'].children = []
    update_param_count_preview(wd)
def on_num_layers_change(c, wd): update_layer_size_widgets(c['new'], wd); update_param_count_preview(wd)
def on_compare_models_change(c, wd): wd['models_to_compare'].disabled = not c['new']
def on_multi_seed_change(c, wd): wd['num_seeds'].disabled = not c['new']

def update_layer_size_widgets(n, wd):
    ws = [widgets.IntText(value=max(32, 512//(2**i)), description=f'L{i+1}:', layout=widgets.Layout(width='200px')) for i in range(n)]
    for w in ws: w.observe(lambda c: update_param_count_preview(wd), names='value')
    wd['layer_sizes_container'].children = ws

def get_current_hidden_sizes(wd):
    return [w.value for w in wd['layer_sizes_container'].children] if wd['model_preset'].value == 'Custom' else get_model_architecture(wd['model_preset'].value)

def update_param_count_preview(wd):
    try:
        K, total, prev = wd['K'].value, 0, 2*wd['K'].value
        for s in get_current_hidden_sizes(wd): total += prev*s + s; prev = s
        total += prev*K + K
        wd['param_count_display'].value = f"<b>Params:</b> {total:,}"
    except: pass

# === STACK MANAGEMENT ===
def on_add_to_stack(b, wd):
    global EXPERIMENT_STACK
    cfg = config_to_dict(wd)
    idx = len(EXPERIMENT_STACK) + 1
    if 'custom_exp_name' in wd and wd['custom_exp_name'].value.strip():
        name = wd['custom_exp_name'].value.strip()
        cfg['experiment_name'] = name
        wd['custom_exp_name'].value = ""
    else:
        cat = wd['probe_category'].value.split()[0][:4]
        model = cfg['model_preset'][:12]
        name = f"[{model}] {cat}:{cfg['probe_type']} M={cfg['M']}"
        cfg['experiment_name'] = name
    cfg['stack_index'] = idx - 1
    EXPERIMENT_STACK.append(cfg)
    wd['stack_display'].options = [c['experiment_name'] for c in EXPERIMENT_STACK]
    wd['transfer_source'].options = ['None'] + [c['experiment_name'] for c in EXPERIMENT_STACK]
    wd['transfer_source'].disabled = False
    paths = ensure_paths(wd)
    save_state(paths)  # Auto-save
    with wd['status_output']: print(f"✅ Added: {name}")

def on_remove_from_stack(b, wd):
    global EXPERIMENT_STACK
    if not wd['stack_display'].value:
        with wd['status_output']: print("⚠️ Select item"); return
    sel = wd['stack_display'].value
    EXPERIMENT_STACK = [c for c in EXPERIMENT_STACK if c['experiment_name'] != sel]
    for i, c in enumerate(EXPERIMENT_STACK): c['stack_index'] = i
    wd['stack_display'].options = [c['experiment_name'] for c in EXPERIMENT_STACK]
    if EXPERIMENT_STACK:
        wd['transfer_source'].options = ['None'] + [c['experiment_name'] for c in EXPERIMENT_STACK]
    else:
        wd['transfer_source'].options, wd['transfer_source'].value = ['None'], 'None'
        wd['transfer_source'].disabled = True
    if SESSION_PATHS: save_state(SESSION_PATHS)
    with wd['status_output']: print(f"🗑️ Removed: {sel}")

def on_move_up(b, wd):
    global EXPERIMENT_STACK
    if not wd['stack_display'].value: return
    sel = wd['stack_display'].value
    idx = next((i for i, c in enumerate(EXPERIMENT_STACK) if c['experiment_name'] == sel), None)
    if idx and idx > 0:
        EXPERIMENT_STACK[idx], EXPERIMENT_STACK[idx-1] = EXPERIMENT_STACK[idx-1], EXPERIMENT_STACK[idx]
        for i, c in enumerate(EXPERIMENT_STACK): c['stack_index'] = i
        wd['stack_display'].options = [c['experiment_name'] for c in EXPERIMENT_STACK]
        wd['stack_display'].value = EXPERIMENT_STACK[idx-1]['experiment_name']
        if SESSION_PATHS: save_state(SESSION_PATHS)

def on_move_down(b, wd):
    global EXPERIMENT_STACK
    if not wd['stack_display'].value: return
    sel = wd['stack_display'].value
    idx = next((i for i, c in enumerate(EXPERIMENT_STACK) if c['experiment_name'] == sel), None)
    if idx is not None and idx < len(EXPERIMENT_STACK)-1:
        EXPERIMENT_STACK[idx], EXPERIMENT_STACK[idx+1] = EXPERIMENT_STACK[idx+1], EXPERIMENT_STACK[idx]
        for i, c in enumerate(EXPERIMENT_STACK): c['stack_index'] = i
        wd['stack_display'].options = [c['experiment_name'] for c in EXPERIMENT_STACK]
        wd['stack_display'].value = EXPERIMENT_STACK[idx+1]['experiment_name']
        if SESSION_PATHS: save_state(SESSION_PATHS)

def on_clear_stack(b, wd):
    global EXPERIMENT_STACK, STACK_RESULTS
    EXPERIMENT_STACK, STACK_RESULTS = [], []
    wd['stack_display'].options = []
    wd['transfer_source'].options, wd['transfer_source'].value = ['None'], 'None'
    wd['transfer_source'].disabled = True
    if SESSION_PATHS: save_state(SESSION_PATHS)
    with wd['status_output']: print("🗑️ Cleared")

# === SAVE/LOAD ===
def on_save_stack(b, wd):
    if not EXPERIMENT_STACK:
        with wd['status_output']: print("⚠️ Empty"); return
    paths = ensure_paths(wd)
    fname = Path(paths['stacks'])/f"stack_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(fname, 'w') as f: json.dump({'stack': EXPERIMENT_STACK, 'count': len(EXPERIMENT_STACK)}, f, indent=2)
        with wd['status_output']: print(f"💾 Saved: {fname.name}")
    except Exception as e:
        with wd['status_output']: print(f"❌ {e}")

def on_load_stack_browser(b, wd):
    global EXPERIMENT_STACK
    base = wd['output_dir'].value
    files = [(os.path.relpath(os.path.join(r,f), base), os.path.join(r,f))
             for r,_,fs in os.walk(base) for f in fs if f.startswith('stack_') and f.endswith('.json')]
    if not files:
        with wd['status_output']: print("⚠️ No stacks"); return
    sel = widgets.Dropdown(options=files, description='Stack:', layout=widgets.Layout(width='450px'))
    btn = widgets.Button(description='Load', button_style='success')
    def load(b):
        try:
            with open(sel.value) as f: data = json.load(f)
            EXPERIMENT_STACK.clear(); EXPERIMENT_STACK.extend(data.get('stack', []))
            wd['stack_display'].options = [c['experiment_name'] for c in EXPERIMENT_STACK]
            if EXPERIMENT_STACK:
                wd['transfer_source'].options = ['None'] + [c['experiment_name'] for c in EXPERIMENT_STACK]
                wd['transfer_source'].disabled = False
            print(f"✅ Loaded {len(EXPERIMENT_STACK)} experiments")
        except Exception as e: print(f"❌ {e}")
    btn.on_click(load)
    with wd['status_output']:
        clear_output(wait=True)
        display(widgets.VBox([sel, btn]))

def on_save_config_click(b, wd):
    cfg = config_to_dict(wd)
    paths = ensure_paths(wd)
    fname = Path(paths['configs'])/f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(fname, 'w') as f: json.dump(cfg, f, indent=2)
        with wd['status_output']: print(f"💾 {fname.name}")
    except Exception as e:
        with wd['status_output']: print(f"❌ {e}")

def on_load_config_browser(b, wd):
    base = wd['output_dir'].value
    files = [(os.path.relpath(os.path.join(r,f), base), os.path.join(r,f))
             for r,_,fs in os.walk(base) for f in fs if f.startswith('config_') and f.endswith('.json')]
    if not files:
        with wd['status_output']: print("⚠️ No configs"); return
    sel = widgets.Dropdown(options=files, description='Config:', layout=widgets.Layout(width='450px'))
    btn = widgets.Button(description='Load', button_style='success')
    def load(b):
        try:
            with open(sel.value) as f: cfg = json.load(f)
            dict_to_widgets(cfg, wd)
            print(f"✅ Loaded")
        except Exception as e: print(f"❌ {e}")
    btn.on_click(load)
    with wd['status_output']:
        clear_output(wait=True)
        display(widgets.VBox([sel, btn]))

# === PLOT-ONLY & RESULTS LOADING ===
def on_plot_only_mode(b, wd):
    global CURRENT_RESULTS
    if not (STACK_RESULTS or CURRENT_RESULTS):
        with wd['status_output']: print("⚠️ No results"); return
    with wd['status_output']: print("🎨 Plotting...")
    if STACK_RESULTS:
        CURRENT_RESULTS = {res.config.get('experiment_name', f"Exp #{i+1}"): res for i, res in enumerate(STACK_RESULTS)}
    update_results_display(wd, CURRENT_RESULTS)
    with wd['status_output']: print("✅ Done!")

def on_save_results(b, wd):
    if not STACK_RESULTS:
        with wd['status_output']: print("⚠️ No results"); return
    paths = ensure_paths(wd)
    fname = Path(paths['results'])/f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    try:
        with open(fname, 'wb') as f: pickle.dump(STACK_RESULTS, f)
        with wd['status_output']: print(f"💾 {fname.name}")
    except Exception as e:
        with wd['status_output']: print(f"❌ {e}")

def on_load_results_browser(b, wd):
    base = wd['output_dir'].value
    # Look for both results and session_state files
    files = []
    for r,_,fs in os.walk(base):
        for f in fs:
            if (f.startswith('results_') and f.endswith('.pkl')) or f == 'session_state.pkl':
                full_path = os.path.join(r, f)
                rel_path = os.path.relpath(full_path, base)
                # Prioritize session_state files
                priority = 0 if f == 'session_state.pkl' else 1
                files.append((priority, rel_path, full_path))

    if not files:
        with wd['status_output']:
            clear_output(wait=True)
            print("⚠️ No saved results found")
        return

    # Sort by priority (session_state first)
    files.sort()
    file_options = [(name, path) for _, name, path in files]

    sel = widgets.Dropdown(
        options=file_options,
        description='File:',
        layout=widgets.Layout(width='500px'),
        style={'description_width': '60px'}
    )
    btn = widgets.Button(description='Load & Restore', button_style='success', layout=widgets.Layout(width='150px'))
    info = widgets.HTML(value="<i>💡 Tip: session_state.pkl restores EVERYTHING (stack + results + cache)</i>")

    def load(b):
        filepath = sel.value
        try:
            # Try loading as session state first
            if filepath.endswith('session_state.pkl'):
                load_state(filepath, wd)
            else:
                # Load just results
                with open(filepath, 'rb') as f: loaded = pickle.load(f)
                global STACK_RESULTS, CURRENT_RESULTS

                with wd['status_output']:
                    clear_output(wait=True)
                    print("📂 Loading results...")

                if isinstance(loaded, list):
                    STACK_RESULTS = loaded
                    CURRENT_RESULTS = {res.config.get('experiment_name', f"Exp #{i+1}"): res
                                     for i, res in enumerate(STACK_RESULTS)}
                else:
                    CURRENT_RESULTS = loaded

                # Update display
                update_results_display(wd, CURRENT_RESULTS)

                with wd['status_output']:
                    clear_output(wait=True)
                    print(f"✅ Results loaded!")
                    print(f"   Experiments: {len(STACK_RESULTS) if isinstance(loaded, list) else 1}")
                    print()
                    print("📊 Check Results & Analysis Dashboard below!")
        except Exception as e:
            with wd['status_output']:
                clear_output(wait=True)
                print(f"❌ Load error: {e}")
            import traceback
            traceback.print_exc()

    btn.on_click(load)

    with wd['status_output']:
        clear_output(wait=True)
        print("📂 Select a file to load:")
        print()
        display(widgets.VBox([
            info,
            widgets.HBox([sel, btn])
        ], layout=widgets.Layout(padding='10px')))

def on_pause_training(b, wd):
    global PAUSE_REQUESTED
    PAUSE_REQUESTED = True
    with wd['status_output']:
        print("\n⏸️ PAUSE REQUESTED - Will stop after current experiment completes")
        print("   (Current experiment will finish, then execution stops)\n")

def on_resume_training(b, wd):
    global PAUSE_REQUESTED
    PAUSE_REQUESTED = False
    with wd['status_output']: print("▶️ Resume (not fully implemented)")

# === EXPERIMENT EXECUTION ===
def on_run_stack(b, wd):
    global STACK_RESULTS, CURRENT_RESULTS
    if not EXPERIMENT_STACK:
        with wd['status_output']: print("⚠️ Empty stack"); return
    paths = ensure_paths(wd)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    wd['status_output'].clear_output()
    STACK_RESULTS = []

    with wd['status_output']:
        print(f"🚀 Running {len(EXPERIMENT_STACK)} experiments...\n")
        for i, cfg in enumerate(EXPERIMENT_STACK):
            exp_name = cfg.get('experiment_name', f"Exp #{i+1}")
            print(f"▶️ [{i+1}/{len(EXPERIMENT_STACK)}] {exp_name}")

            # Check cache first
            cached = get_cached(cfg)
            if cached:
                print(f"   ⚡ Using cached result (skip training)")
                STACK_RESULTS.append(cached)
                continue

            # Check transfer learning
            initial_weights = None
            trans = cfg.get('transfer_source', 'None')
            if trans not in ['None', None]:
                try:
                    src_name = trans
                    src_idx = next((j for j, c in enumerate(EXPERIMENT_STACK) if c['experiment_name'] == src_name), None)
                    if src_idx is not None and src_idx < len(STACK_RESULTS):
                        print(f"   ↪ Transfer from: {src_name}")
                        initial_weights = STACK_RESULTS[src_idx].model_state
                except: pass

            try:
                def progress_cb(epoch, total, metrics):
                    wd['progress_bar'].value = int((epoch/total)*100)
                    wd['live_metrics'].value = (f"<b>{exp_name}</b><br>Epoch {epoch}/{total}<br>"
                                               f"Loss: {metrics.get('val_loss',0):.4f}<br>"
                                               f"η: {metrics.get('val_eta',0):.4f}")

                # Try with progress_callback, fall back if not supported
                try:
                    result = run_single_experiment(cfg, progress_callback=progress_cb,
                                                 initial_weights=initial_weights, verbose=False)
                except TypeError:
                    # Old version without progress_callback support
                    result = run_single_experiment(cfg, initial_weights=initial_weights, verbose=False)
                STACK_RESULTS.append(result)
                cache(cfg, result)  # Cache for future

                # Save model with custom name
                safe_name = "".join(c if c.isalnum() or c in '_-' else '_' for c in exp_name)
                model_file = Path(paths['models'])/f"{safe_name}_{timestamp}.pt"
                torch.save(result.model_state, model_file)

                # FIXED: Check if evaluation exists before accessing
                if result.evaluation is not None and hasattr(result.evaluation, 'accuracy_top1'):
                    if result.evaluation.accuracy_top1 < 0.01:
                        print(f"   ⚠️ Low accuracy ({result.evaluation.accuracy_top1:.3f})")
                    else:
                        print(f"   ✅ η={result.evaluation.eta_top1:.4f} | Acc={result.evaluation.accuracy_top1:.3f}\n")
                else:
                    print(f"   ⚠️ Experiment failed - no evaluation\n")


            except Exception as e:
                print(f"   ❌ Failed: {e}\n")
                import traceback
                traceback.print_exc()

        # Update display
        CURRENT_RESULTS = {res.config.get('experiment_name', f"Exp #{i+1}"): res
                          for i, res in enumerate(STACK_RESULTS)}
        update_results_display(wd, CURRENT_RESULTS)

        # Auto-save everything
        results_file = Path(paths['results'])/f"results_{timestamp}.pkl"
        with open(results_file, 'wb') as f: pickle.dump(STACK_RESULTS, f)
        save_state(paths)

        print(f"\n🏁 Complete! {len(STACK_RESULTS)}/{len(EXPERIMENT_STACK)} successful")
        print(f"   Results: {results_file.name}")
        print(f"   Cached: {len(TRAINED_CACHE)} experiments for future use")


def on_run_single(b, wd):
    global CURRENT_RESULTS
    wd['status_output'].clear_output()
    wd['results_summary'].value = "<i>Running...</i>"

    with wd['status_output']:
        try:
            cfg = config_to_dict(wd)

            # PHASE 1: Print physics configuration
            print("=" * 70)
            print("EXPERIMENT CONFIGURATION")
            print("=" * 70)
            print(f"Physics Source: {cfg.get('channel_source', 'python_synthetic')}")
            print(f"Realism Profile: {cfg.get('realism_profile', 'ideal')}")
            if cfg.get('use_custom_impairments'):
                print("Custom Impairments: ENABLED")
            print("=" * 70)
            print()

            def progress_cb(epoch, total, metrics):
                wd['progress_bar'].value = int((epoch / total) * 100)
                wd['live_metrics'].value = (f"<b>Training...</b><br>Epoch {epoch}/{total}<br>"
                                            f"Loss: {metrics.get('val_loss', 0):.4f}")

            if cfg.get('compare_multiple_models'):
                models = list(cfg.get('models_to_compare', []))
                if len(models) < 2:
                    print("WARNING: Select 2+ models");
                    return
                CURRENT_RESULTS = run_multi_model_comparison(cfg, models, progress_cb)
            else:
                CURRENT_RESULTS = run_single_experiment(cfg, progress_callback=progress_cb)

            update_results_display(wd, CURRENT_RESULTS)
            print("\nDone!")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

# === RESULTS DISPLAY ===
def update_results_display(wd, results):
    if not results:
        wd['results_summary'].value = "<div style='padding: 20px;'><i>No results yet. Run experiments or load saved results.</i></div>"
        return

    try:
        # Clear output areas first
        wd['results_plots_training'].clear_output()
        wd['results_plots_analysis'].clear_output()

        # Summary table
        if isinstance(results, dict):
            summary = "<h3>📊 Experiment Results</h3>"
            summary += "<table border='1' style='border-collapse: collapse; width: 100%;'>"
            summary += "<tr style='background-color: #f0f0f0;'>"
            summary += "<th>Experiment</th><th>Model</th><th>Probe</th><th>M</th><th>η top-1</th><th>Acc</th><th>Time(s)</th></tr>"
            for name, res in results.items():
                cfg, ev = res.config, res.evaluation
                # FIXED: Handle None evaluation
                if ev is not None and hasattr(ev, 'eta_top1'):
                    summary += (f"<tr><td><b>{name}</b></td>"
                                f"<td>{cfg.get('model_preset', '?')}</td>"
                                f"<td>{cfg.get('probe_type', '?')}</td>"
                                f"<td>{cfg.get('M', '?')}</td>"
                                f"<td>{ev.eta_top1:.4f}</td>"
                                f"<td>{ev.accuracy_top1:.3f}</td>"
                                f"<td>{res.execution_time:.1f}</td></tr>")
                else:
                    summary += (f"<tr><td><b>{name}</b></td>"
                                f"<td>{cfg.get('model_preset', '?')}</td>"
                                f"<td>{cfg.get('probe_type', '?')}</td>"
                                f"<td>{cfg.get('M', '?')}</td>"
                                f"<td colspan='2' style='color: red;'>FAILED</td>"
                                f"<td>{res.execution_time:.1f}</td></tr>")
            summary += "</table>"
            summary += f"<p style='margin-top: 10px;'><i>Loaded {len(results)} experiments</i></p>"
            wd['results_summary'].value = summary
        else:
            ev, cfg = results.evaluation, results.config
            wd['results_summary'].value = (f"<h3>📊 Results</h3>"
                f"<p><b>Model:</b> {cfg.get('model_preset','?')}</p>"
                f"<p><b>η:</b> {ev.eta_top1:.4f} | <b>Acc:</b> {ev.accuracy_top1:.3f}</p>"
                f"<p><b>Time:</b> {results.execution_time:.1f}s</p>")

        # Generate plots
        try:
            generate_plots(wd, results)
        except Exception as e:
            with wd['status_output']:
                print(f"⚠️ Plot generation error: {e}")
                print("   Results table is still available above")
    except Exception as e:
        with wd['status_output']:
            print(f"❌ Display error: {e}")
        import traceback
        traceback.print_exc()

def generate_plots(wd, results):
    wd['results_plots_training'].clear_output()
    wd['results_plots_analysis'].clear_output()

    TRAINING_PLOTS = ['training_curves']
    set_plot_style(wd['color_palette'].value)
    paths = ensure_paths(wd)
    save_plots, fig_format, dpi = wd['save_plots'].value, wd['figure_format'].value, wd['dpi'].value

    for plot_name in wd['selected_plots'].value:
        target = wd['results_plots_training'] if plot_name in TRAINING_PLOTS else wd['results_plots_analysis']
        with target:
            try:
                if isinstance(results, dict):
                    generate_multi_plot(plot_name, results)
                else:
                    generate_single_plot(plot_name, results)

                if save_plots:
                    fname = Path(paths['plots'])/f"{plot_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{fig_format}"
                    plt.savefig(fname, dpi=dpi, bbox_inches='tight')
                    print(f"💾 {fname.name}")

                plt.show()
                plt.close('all')
            except Exception as e:
                print(f"❌ {plot_name}: {e}")
                plt.close('all')

def generate_single_plot(plot_name, result):
    if plot_name == 'training_curves':
        EXTENDED_PLOT_REGISTRY['training_curves'](result.training_history)
    elif plot_name == 'eta_distribution':
        EXTENDED_PLOT_REGISTRY['eta_distribution'](result.evaluation)
    elif plot_name == 'top_m_comparison':
        EXTENDED_PLOT_REGISTRY['top_m_comparison'](result.evaluation)
    elif plot_name == 'baseline_comparison':
        EXTENDED_PLOT_REGISTRY['baseline_comparison'](result.evaluation)
    elif plot_name == 'cdf':
        EXTENDED_PLOT_REGISTRY['cdf'](result.evaluation)
    else:
        print(f"ℹ️ {plot_name}: Multi-experiment only")

def generate_multi_plot(plot_name, results_dict):
    if plot_name == 'training_curves':
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for name, res in results_dict.items():
            hist = res.training_history
            axes[0].plot(hist.train_loss, label=f'{name} Train', alpha=0.7)
            axes[0].plot(hist.val_loss, label=f'{name} Val', alpha=0.7, linestyle='--')
            axes[1].plot(hist.val_eta, label=name, alpha=0.7)
        axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].set_title('η'); axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
    elif plot_name in ['eta_distribution', 'baseline_comparison', 'cdf', 'heatmap']:
        # Pass full dict - these plots now support multi-experiment
        EXTENDED_PLOT_REGISTRY[plot_name](results_dict)
    elif plot_name in EXTENDED_PLOT_REGISTRY:
        EXTENDED_PLOT_REGISTRY[plot_name](results_dict)
    else:
        print(f"⚠️ {plot_name} not found")

# === EXPORT ===
def export_results_data(results, wd, fmt):
    if not results:
        with wd['status_output']: print("⚠️ No results"); return
    paths = ensure_paths(wd)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    if isinstance(results, dict):
        rows = [{
            'Experiment': name,
            'Model': res.config.get('model_preset'),
            'Probe': res.config.get('probe_type'),
            'M': res.config.get('M'),
            'eta': res.evaluation.eta_top1,
            'accuracy': res.evaluation.accuracy_top1,
            'time': res.execution_time
        } for name, res in results.items()]
    else:
        rows = [{'Model': results.config.get('model_preset'), 'eta': results.evaluation.eta_top1,
                'accuracy': results.evaluation.accuracy_top1}]

    df = pd.DataFrame(rows)
    try:
        if fmt == 'csv':
            fname = Path(paths['exports'])/f"results_{ts}.csv"
            df.to_csv(fname, index=False)
        elif fmt == 'json':
            fname = Path(paths['exports'])/f"results_{ts}.json"
            df.to_json(fname, orient='records', indent=2)
        elif fmt == 'latex':
            fname = Path(paths['exports'])/f"results_{ts}.tex"
            df.to_latex(fname, index=False, float_format="%.4f")
        with wd['status_output']: print(f"✅ Exported: {fname.name}")
    except Exception as e:
        with wd['status_output']: print(f"❌ {e}")

def save_ml_model(results, wd):
    if not results:
        with wd['status_output']: print("⚠️ No model"); return
    if isinstance(results, dict):
        with wd['status_output']: print("⚠️ Multiple models"); return
    paths = ensure_paths(wd)
    fname = Path(paths['models'])/f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    try:
        torch.save(results.model_state, fname)
        with wd['status_output']: print(f"💾 {fname.name}")
    except Exception as e:
        with wd['status_output']: print(f"❌ {e}")

def reset_to_defaults(wd):
    wd['N'].value, wd['K'].value, wd['M'].value = 32, 64, 8
    wd['P_tx'].value, wd['sigma_h_sq'].value, wd['sigma_g_sq'].value = 1.0, 1.0, 1.0
    wd['phase_mode'].value, wd['probe_category'].value, wd['probe_type'].value = 'continuous', 'Physics-Based', 'continuous'
    wd['model_preset'].value = 'Baseline_MLP'
    wd['n_train'].value, wd['n_val'].value, wd['n_test'].value = 50000, 5000, 5000
    wd['learning_rate'].value, wd['n_epochs'].value = 1e-3, 50
    with wd['status_output']: print("✅ Reset")

# === SETUP CALLBACKS ===
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

def setup_experiment_handlers(wd):
    # Clear existing handlers
    for btn in ['button_add_to_stack', 'button_remove_from_stack', 'button_move_up', 'button_move_down',
                'button_clear_stack', 'button_save_stack', 'button_load_stack', 'button_run_experiment',
                'button_run_stack', 'button_save_config', 'button_load_config', 'button_reset_defaults',
                'button_plot_only', 'button_save_results', 'button_load_results',
                'button_pause_training', 'button_resume_training',
                'button_export_csv', 'button_export_json', 'button_export_latex', 'button_save_model']:
        if btn in wd:
            wd[btn]._click_handlers.callbacks = []

    # Stack
    wd['button_add_to_stack'].on_click(lambda b: on_add_to_stack(b, wd))
    wd['button_remove_from_stack'].on_click(lambda b: on_remove_from_stack(b, wd))
    wd['button_move_up'].on_click(lambda b: on_move_up(b, wd))
    wd['button_move_down'].on_click(lambda b: on_move_down(b, wd))
    wd['button_clear_stack'].on_click(lambda b: on_clear_stack(b, wd))
    wd['button_save_stack'].on_click(lambda b: on_save_stack(b, wd))
    wd['button_load_stack'].on_click(lambda b: on_load_stack_browser(b, wd))

    # Execution
    wd['button_run_experiment'].on_click(lambda b: on_run_single(b, wd))
    wd['button_run_stack'].on_click(lambda b: on_run_stack(b, wd))

    # Config
    wd['button_save_config'].on_click(lambda b: on_save_config_click(b, wd))
    wd['button_load_config'].on_click(lambda b: on_load_config_browser(b, wd))
    wd['button_reset_defaults'].on_click(lambda b: reset_to_defaults(wd))

    # Plots
    wd['button_plot_only'].on_click(lambda b: on_plot_only_mode(b, wd))
    wd['button_save_results'].on_click(lambda b: on_save_results(b, wd))
    wd['button_load_results'].on_click(lambda b: on_load_results_browser(b, wd))

    # Pause/Resume
    wd['button_pause_training'].on_click(lambda b: on_pause_training(b, wd))
    wd['button_resume_training'].on_click(lambda b: on_resume_training(b, wd))

    # Export
    # NEW CODE (handles missing buttons):
    if 'button_export_csv' in wd:
        wd['button_export_csv'].on_click(lambda b: export_results_data(CURRENT_RESULTS, wd, 'csv'))
    if 'button_export_json' in wd:
        wd['button_export_json'].on_click(lambda b: export_results_data(CURRENT_RESULTS, wd, 'json'))
    if 'button_export_latex' in wd:
        wd['button_export_latex'].on_click(lambda b: export_results_data(CURRENT_RESULTS, wd, 'latex'))
    if 'button_save_model' in wd:
        wd['button_save_model'].on_click(lambda b: save_ml_model(CURRENT_RESULTS, wd))

    # Interactive plots (with debouncing)
    def safe_plot_update(change):
        if PLOT_UPDATE_ENABLED and CURRENT_RESULTS:
            try: update_results_display(wd, CURRENT_RESULTS)
            except: pass

    wd['selected_plots'].observe(safe_plot_update, names='value')
    wd['color_palette'].observe(safe_plot_update, names='value')