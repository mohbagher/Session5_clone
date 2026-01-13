"""
Extended Plot Registry for RIS PhD Dashboard.
Supports Single, Multi-Model, and Sweep plotting.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def set_plot_style(palette='viridis'):
    sns.set_theme(style="whitegrid")
    sns.set_palette(palette)

# --- SWEEP PLOTS (The Logic you asked for) ---
def plot_eta_vs_M(results_dict):
    """
    Automatically extracts M vs Eta from a stack of experiments.
    Expects results_dict = {'Exp1': res1, 'Exp2': res2...}
    """
    data = []
    for name, res in results_dict.items():
        # Check if result has config M and evaluation eta
        if hasattr(res, 'config') and hasattr(res, 'evaluation'):
            m_val = res.config['M'] if isinstance(res.config, dict) else res.config.system.M
            eta = res.evaluation.eta_top1
            data.append({'M': m_val, 'η': eta, 'Model': name})

    if not data:
        print("⚠️ No valid data found for Sweep Plot.")
        return

    df = pd.DataFrame(data).sort_values('M')

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='M', y='η', marker='o', linewidth=2.5)
    plt.title("η (Power Efficiency) vs Sensing Budget (M)", fontsize=14)
    plt.xlabel("Number of Probes (M)", fontsize=12)
    plt.ylabel("η (Normalized Power)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

def plot_training_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    ax1.plot(history.train_loss, label='Train')
    ax1.plot(history.val_loss, label='Val')
    ax1.set_title('Loss History')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend()

    # Accuracy/Eta
    if hasattr(history, 'val_acc'):
        ax2.plot(history.val_acc, label='Accuracy', color='green')
    if hasattr(history, 'val_eta'):
        ax2.plot(history.val_eta, label='η (Efficiency)', color='orange')
    ax2.set_title('Metrics History')
    ax2.set_xlabel('Epoch')
    ax2.legend()

def plot_top_m_comparison(data):
    # Handles both Single (EvaluationResults) and Multi (Dict)
    plt.figure(figsize=(10, 6))

    if isinstance(data, dict): # Multi-model comparison
        rows = []
        for name, res in data.items():
            ev = res.evaluation
            rows.append({'Model': name, 'Top-1': ev.accuracy_top1, 'Top-2': ev.accuracy_top2, 'Top-4': ev.accuracy_top4})
        df = pd.DataFrame(rows).melt(id_vars='Model', var_name='Metric', value_name='Accuracy')
        sns.barplot(data=df, x='Model', y='Accuracy', hue='Metric')
        plt.xticks(rotation=45)
    else: # Single EvaluationResults
        # Extract top-m values dynamically if available, else standard
        m_vals = ['Top-1', 'Top-2', 'Top-4', 'Top-8']
        accs = [data.accuracy_top1, data.accuracy_top2, data.accuracy_top4, data.accuracy_top8]
        sns.barplot(x=m_vals, y=accs)
        plt.ylim(0, 1.0)
        plt.title("Top-m Accuracy Analysis")

def plot_heatmap(probe_bank):
    # Visualizes the first probe in the bank
    phases = probe_bank.phases[0].cpu().numpy().reshape(-1) # Flatten
    N = len(phases)
    side = int(np.sqrt(N))
    if side * side == N:
        grid = phases.reshape(side, side)
        plt.figure(figsize=(8, 6))
        sns.heatmap(grid, cmap='twilight', annot=False)
        plt.title(f"Probe Phase Configuration ({side}x{side})")
    else:
        print("⚠️ RIS elements (N) not a perfect square, skipping heatmap.")

# --- REGISTRY ---
EXTENDED_PLOT_REGISTRY = {
    'training_curves': plot_training_curves,
    'eta_vs_M': plot_eta_vs_M, # New Sweep Handler
    'top_m_comparison': plot_top_m_comparison,
    'heatmap': plot_heatmap,
    # Add placeholders for others to prevent crashes if selected
    'eta_distribution': lambda x: print("Plot placeholder: eta_distribution"),
    'cdf': lambda x: print("Plot placeholder: cdf"),
    'baseline_comparison': lambda x: print("Plot placeholder: baseline_comparison"),
    'violin': lambda x: print("Plot placeholder: violin"),
    'box': lambda x: print("Plot placeholder: box"),
    'scatter': lambda x, y: print("Plot placeholder: scatter"),
}