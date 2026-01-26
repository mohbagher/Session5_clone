"""
Experiment Runner - DASHBOARD BRIDGE FIXED
==========================================
1. Accepts 'progress_callback' from callbacks.py (The Critical Link).
2. Updates 'progress_bar' and 'live_metrics' correctly.
3. Keeps the text log alive.
4. FIX: Imports 'List' correctly to prevent NameError.
"""
import logging
import traceback
import sys
import time
from datetime import datetime
from typing import Dict, Optional, Any, List  # <--- FIX: Added 'List' here
from dataclasses import dataclass

from src.ris_platform.experiments.factories import create_physics_model, create_backend, create_probing_strategy
from src.ris_platform.experiments.pipeline import generate_channels, compute_physics_features
from src.ris_platform.experiments.training import train_model_phase2

try:
    from evaluation.evaluator import evaluate_model, EvaluationResults
except ImportError:
    from src.ris_platform.experiments.evaluation import evaluate_model, EvaluationResults

# --- ROBUST DATA WRAPPER ---
class DataWrapper:
    """
    Hybrid Object/Dict wrapper.
    Allows accessing data via .attribute OR ['key'] to satisfy both Dashboard and Tests.
    """
    def __init__(self, d):
        self._d = d
        for k, v in d.items():
            setattr(self, k, v)

    def __getitem__(self, item):
        return self._d[item]

    def get(self, item, default=None):
        return self._d.get(item, default)

class ConfigWrapper:
    def __init__(self, d):
        self.training = DataWrapper({'device': d.get('device', 'cpu')})
        self.system = DataWrapper({'K': d['K'], 'M': d['M']})
        self.eval = DataWrapper({'top_m_values': d.get('top_m_values', [1,2,4,8])})
        self.data = DataWrapper({'seed': d.get('seed', 42)})

@dataclass
class ExperimentResult:
    config: Dict
    model_state: Any
    training_history: Any
    evaluation: Any
    execution_time: float
    metadata: Dict

def update_ui(widget_dict: Optional[Dict], text: str = None, progress: int = None, status: str = None):
    """
    Unified UI Updater synchronized with your callbacks.py keys.
    """
    if not widget_dict:
        if text: print(text); sys.stdout.flush()
        return

    # 1. Update Execution Log (Text Box)
    if text and 'status_output' in widget_dict:
        with widget_dict['status_output']:
            print(text)
            sys.stdout.flush()

    # 2. Update Progress Bar (Matching your callbacks.py key)
    if 'progress_bar' in widget_dict and progress is not None:
        widget_dict['progress_bar'].value = progress

    # 3. Update Live Metrics (Matching your callbacks.py key)
    if 'live_metrics' in widget_dict and status is not None:
        # We use .value because live_metrics is an HTML or Label widget
        widget_dict['live_metrics'].value = status

def run_single_experiment(config: Dict, widget_dict: Optional[Dict] = None, progress_callback=None, **kwargs) -> ExperimentResult:
    start_time = datetime.now()
    try:
        # 1. SETUP
        update_ui(widget_dict, text="1. Initializing Modules...", progress=5, status="Initializing...")
        physics = create_physics_model(config)
        backend = create_backend(config)
        probing = create_probing_strategy(config)

        # 2. DATA
        update_ui(widget_dict, text="2. Generating Channels...", progress=10, status="Generating Data...")
        h_all, g_all, meta = generate_channels(backend, config)
        if h_all is None: raise RuntimeError("Backend returned None.")

        # 3. PHYSICS
        update_ui(widget_dict, text="3. Physics Simulation...", progress=20, status="Simulating Physics...")
        train_ds, val_ds, test_data = compute_physics_features(physics, probing, h_all, g_all, config)

        # 4. TRAINING
        n_epochs = int(config.get('n_epochs', 10))
        update_ui(widget_dict, text=f"4. Training Model ({n_epochs} Epochs)...", progress=30, status="Training Started...")

        # --- THE BRIDGE ---
        def on_epoch_end(epoch, logs):
            # PRIORITY 1: Use the callback provided by callbacks.py
            if progress_callback:
                progress_callback(epoch, n_epochs, logs)

            # PRIORITY 2: Fallback to local logic if no callback provided
            else:
                prog_val = 30 + int((epoch / n_epochs) * 60)
                loss = logs.get('train_loss', 0)
                acc = logs.get('val_acc', 0)
                status_html = f"<b>Training...</b><br>Epoch {epoch}/{n_epochs}<br>Loss: {loss:.4f}<br>Acc: {acc:.2%}"
                update_ui(widget_dict, text=None, progress=prog_val, status=status_html)

            # ALWAYS: Update the text log
            loss = logs.get('train_loss', 0)
            acc = logs.get('val_acc', 0)
            log_msg = f"  Epoch {epoch:02d}/{n_epochs} | Loss: {loss:.4f} | Acc: {acc:.2%}"
            update_ui(widget_dict, text=log_msg)

            # Yield Control
            time.sleep(0.01)

        model, history = train_model_phase2(
            train_ds,
            val_ds,
            config,
            callback=on_epoch_end
        )

        # 5. EVALUATION
        update_ui(widget_dict, text="5. Evaluating...", progress=95, status="Evaluating...")
        cfg_obj = ConfigWrapper(config)

        eval_results = evaluate_model(
            model=model,
            test_loader=test_data['loader'],
            config=cfg_obj,
            powers_full=test_data['powers_full'],
            labels=test_data['labels'],
            observed_indices=test_data['observed_indices'],
            optimal_powers=test_data['optimal_powers']
        )

        # 6. FINISH
        acc = eval_results.accuracy_top1
        update_ui(widget_dict, text=f"✅ Done! Final Acc: {acc:.2%}", progress=100, status="Complete")

        # Wrap Outputs
        history_obj = DataWrapper(history)
        eval_obj = DataWrapper(eval_results) if isinstance(eval_results, dict) else eval_results

        return ExperimentResult(
            config=config,
            model_state=model.state_dict(),
            training_history=history_obj,
            evaluation=eval_obj,
            execution_time=(datetime.now() - start_time).total_seconds(),
            metadata={'success': True}
        )

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        update_ui(widget_dict, text=error_msg, status="Error")
        traceback.print_exc(file=sys.stdout)

        dummy_hist = DataWrapper({'train_loss':[], 'val_loss':[], 'val_acc':[]})
        class DummyEval: accuracy_top1 = 0.0; eta_top1 = 0.0
        return ExperimentResult(config, None, dummy_hist, DummyEval(), 0.0, {'success': False, 'error': str(e)})

# Helpers
run_experiment_stack = lambda stack, wd: [run_single_experiment(c, wd) for c in stack]

# FIX: Added type hint 'List[str]' which required the import
def run_multi_model_comparison(base_config: Dict, models: List[str], progress_callback=None):
    results = {}
    for m in models:
        cfg = base_config.copy(); cfg['model_preset'] = m
        results[m] = run_single_experiment(cfg, progress_callback=progress_callback)
    return results

def run_multi_seed_experiment(base_config: Dict, seeds: List[int], widget_dict: Optional[Dict] = None):
    return [run_single_experiment({**base_config, 'seed': s}, widget_dict) for s in seeds]