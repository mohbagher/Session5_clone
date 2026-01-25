"""
Tab: Ablation Study (Scientific Validation)
===========================================
Compares Ideal Physics vs. Realistic Phase 2 Physics.
Generates proof plots for research papers.
"""

import ipywidgets as widgets
import matplotlib.pyplot as plt
from src.ris_platform.experiments.runner import run_single_experiment


def render(global_config):
    """
    Renders the Ablation Study Tab.
    """
    # 1. UI Elements
    btn_run = widgets.Button(
        description="RUN ABLATION STUDY (Ideal vs Real)",
        button_style='danger',  # Red button for visibility
        icon='balance-scale',
        layout=widgets.Layout(width='350px', height='50px')
    )

    # Description
    desc_html = widgets.HTML(
        "<h3>🔬 Scientific Validation: Ablation Study</h3>"
        "<p>This tool runs two identical experiments back-to-back:</p>"
        "<ul>"
        "<li><b>Experiment A (Blue):</b> Ideal Physics (Perfect RIS, No Coupling)</li>"
        "<li><b>Experiment B (Red):</b> Phase 2 Realistic Physics (Varactors + Coupling + Aging)</li>"
        "</ul>"
        "<p><i>Use this to generate the 'Performance Gap' plot for your paper.</i></p>"
    )

    output_log = widgets.Output(layout={'border': '1px solid #ddd', 'height': '200px', 'overflow_y': 'scroll'})
    output_plot = widgets.Output(layout={'border': '1px solid #ddd', 'min_height': '400px'})

    # 2. The Logic
    def on_click_run(b):
        output_plot.clear_output()
        output_log.clear_output()

        with output_log:
            print("🚀 Starting Ablation Study...")

            # Define Base Config (Fast but meaningful)
            base_config = {
                'N': 16, 'K': 32, 'M': 8,
                'n_train': 2000,
                'n_val': 500,
                'n_test': 500,
                'physics_backend': 'python',
                'probe_type': 'sobol',
                'seed': 42,
                'epochs': 20,
                'batch_size': 32,
                'learning_rate': 0.001
            }

            # Config A: Ideal
            cfg_ideal = base_config.copy()
            cfg_ideal['realism_profile'] = 'ideal'

            # Config B: Realistic
            cfg_real = base_config.copy()
            cfg_real['realism_profile'] = 'realistic'
            cfg_real['unit_cell_type'] = 'varactor'
            cfg_real['varactor_coupling_strength'] = 0.3
            cfg_real['coupling_type'] = 'geometric'
            cfg_real['enable_aging'] = True

            # Run A
            print(f"⚡ Running Configuration A: Ideal Physics...")
            try:
                res_ideal = run_single_experiment(cfg_ideal, verbose=False)
                print(f"   -> Final Accuracy: {res_ideal.evaluation['accuracy']:.2%}")
            except Exception as e:
                print(f"❌ Ideal Run Failed: {e}")
                return

            # Run B
            print(f"⚡ Running Configuration B: Realistic Physics (Phase 2)...")
            try:
                res_real = run_single_experiment(cfg_real, verbose=False)
                print(f"   -> Final Accuracy: {res_real.evaluation['accuracy']:.2%}")
            except Exception as e:
                print(f"❌ Realistic Run Failed: {e}")
                return

            print("✅ Study Complete. Generating Plot...")

        # 3. Plotting
        with output_plot:
            plt.figure(figsize=(10, 6))

            # Plot Accuracy (if available)
            if 'val_acc' in res_ideal.training_history:
                plt.plot(res_ideal.training_history['val_acc'], label='Ideal Physics (Baseline)',
                         linewidth=2, marker='o', color='blue')
                plt.plot(res_real.training_history['val_acc'], label='Realistic Physics (Phase 2)',
                         linewidth=2, marker='x', color='red')
                plt.ylabel("Validation Accuracy")
            else:
                # Fallback to Loss
                plt.plot(res_ideal.training_history['train_loss'], label='Ideal Loss', linestyle='--')
                plt.plot(res_real.training_history['train_loss'], label='Realistic Loss', linestyle='-')
                plt.ylabel("Training Loss")

            plt.title(f"Ablation Study: Hardware Impairments Impact\n(N={base_config['N']}, K={base_config['K']})")
            plt.xlabel("Epochs")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

    btn_run.on_click(on_click_run)

    # 4. Layout
    layout = widgets.VBox([
        desc_html,
        widgets.HTML("<hr>"),
        btn_run,
        widgets.HTML("<br>"),
        widgets.HBox([output_log, output_plot])
    ], layout=widgets.Layout(padding='20px'))

    # Store widgets for main loop collection
    layout._widgets = {'btn_ablation_run': btn_run}

    return layout