"""
Extended Plot Registry - ULTIMATE V2.0
=======================================
All plots properly implemented with custom experiment naming support
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any

def set_plot_style(palette='viridis'):
    """Set consistent plot style."""
    sns.set_theme(style="whitegrid")
    sns.set_palette(palette)
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'

def get_exp_name(res, idx=0):
    """Extract experiment name from result."""
    if hasattr(res, 'config'):
        return res.config.get('experiment_name', f"Exp #{idx+1}")
    return f"Exp #{idx+1}"

# =============================================================================
# SWEEP PLOTS (Multi-Experiment)
# =============================================================================
def plot_eta_vs_M(results_dict: Dict[str, Any]):
    """Plot η vs sensing budget M with custom names."""
    data = []
    for name, res in results_dict.items():
        if hasattr(res, 'config') and hasattr(res, 'evaluation'):
            m_val = res.config.get('M', res.config.get('system', {}).get('M', None))
            if m_val is not None:
                eta = res.evaluation.eta_top1
                exp_name = get_exp_name(res)
                data.append({'M': m_val, 'η': eta, 'Experiment': exp_name})

    if not data:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No valid M sweep data.\nAdd experiments with different M values.',
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        return

    df = pd.DataFrame(data).sort_values('M')

    if df['M'].nunique() == 1:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f'All experiments have M={df["M"].iloc[0]}\n\nTip: Add experiments with different M values\n(e.g., M=4, M=8, M=12, M=16)',
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df['M'], df['η'], marker='o', linewidth=2.5, markersize=8)
    plt.title("η (Power Efficiency) vs Sensing Budget (M)", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Probes Measured (M)", fontsize=12)
    plt.ylabel("η (Normalized Power)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

def plot_eta_vs_K(results_dict: Dict[str, Any]):
    """Plot η vs total probes K."""
    data = []
    for name, res in results_dict.items():
        if hasattr(res, 'config') and hasattr(res, 'evaluation'):
            k_val = res.config.get('K', res.config.get('system', {}).get('K', None))
            if k_val is not None:
                eta = res.evaluation.eta_top1
                data.append({'K': k_val, 'η': eta, 'Experiment': name})

    if not data:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No K sweep data available.', ha='center', va='center', fontsize=14)
        plt.axis('off')
        return

    df = pd.DataFrame(data).sort_values('K')

    if df['K'].nunique() == 1:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f'All experiments have K={df["K"].iloc[0]}', ha='center', va='center', fontsize=14)
        plt.axis('off')
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df['K'], df['η'], marker='s', linewidth=2.5, markersize=8, color='coral')
    plt.title("η (Power Efficiency) vs Codebook Size (K)", fontsize=14, fontweight='bold')
    plt.xlabel("Total Probes in Codebook (K)", fontsize=12)
    plt.ylabel("η (Normalized Power)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

def plot_eta_vs_N(results_dict: Dict[str, Any]):
    """Plot η vs RIS elements N."""
    data = []
    for name, res in results_dict.items():
        if hasattr(res, 'config') and hasattr(res, 'evaluation'):
            n_val = res.config.get('N', res.config.get('system', {}).get('N', None))
            if n_val is not None:
                eta = res.evaluation.eta_top1
                data.append({'N': n_val, 'η': eta, 'Experiment': name})

    if not data:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No N sweep data available.', ha='center', va='center', fontsize=14)
        plt.axis('off')
        return

    df = pd.DataFrame(data).sort_values('N')

    if df['N'].nunique() == 1:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f'All experiments have N={df["N"].iloc[0]}', ha='center', va='center', fontsize=14)
        plt.axis('off')
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df['N'], df['η'], marker='^', linewidth=2.5, markersize=8, color='green')
    plt.title("η (Power Efficiency) vs RIS Elements (N)", fontsize=14, fontweight='bold')
    plt.xlabel("Number of RIS Elements (N)", fontsize=12)
    plt.ylabel("η (Normalized Power)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

# =============================================================================
# TRAINING PLOTS
# =============================================================================
def plot_training_curves(history):
    """Plot training and validation curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(history.train_loss) + 1)
    ax1.plot(epochs, history.train_loss, label='Train Loss', linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, history.val_loss, label='Val Loss', linewidth=2, marker='s', markersize=4)
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    if hasattr(history, 'val_acc') and len(history.val_acc) > 0:
        ax2.plot(epochs, history.val_acc, label='Accuracy', color='green', linewidth=2, marker='o', markersize=4)
    if hasattr(history, 'val_eta') and len(history.val_eta) > 0:
        ax2.plot(epochs, history.val_eta, label='η (Efficiency)', color='orange', linewidth=2, marker='s', markersize=4)

    ax2.set_title('Performance Metrics', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

# =============================================================================
# ANALYSIS PLOTS
# =============================================================================
def plot_top_m_comparison(data):
    """Plot top-m accuracy comparison with custom names and ALL top-m values."""
    plt.figure(figsize=(12, 6))

    # Distinctive color palette
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    if isinstance(data, dict):
        # Multi-model comparison
        names = []
        accuracy_by_m = {}  # {m_value: [acc1, acc2, ...]}

        # Collect all available top-m accuracies
        first_res = list(data.values())[0]
        available_m = []
        for attr in dir(first_res.evaluation):
            if attr.startswith('accuracy_top'):
                try:
                    m_val = int(attr.replace('accuracy_top', ''))
                    available_m.append(m_val)
                except:
                    pass
        available_m.sort()

        # Collect data for each experiment
        for key, res in data.items():
            exp_name = get_exp_name(res)
            names.append(exp_name)

            for m in available_m:
                attr_name = f'accuracy_top{m}'
                if hasattr(res.evaluation, attr_name):
                    if m not in accuracy_by_m:
                        accuracy_by_m[m] = []
                    accuracy_by_m[m].append(getattr(res.evaluation, attr_name))

        x = np.arange(len(names))
        width = 0.8 / len(accuracy_by_m)  # Dynamic width based on number of top-m values

        # Plot bars for each top-m value
        for i, (m, accuracies) in enumerate(sorted(accuracy_by_m.items())):
            offset = (i - len(accuracy_by_m)/2 + 0.5) * width
            color = COLORS[i % len(COLORS)]
            plt.bar(x + offset, accuracies, width, label=f'Top-{m}', alpha=0.85, color=color, edgecolor='black', linewidth=0.5)

        plt.xlabel('Experiment', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
        plt.title('Top-m Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x, names, rotation=45, ha='right')
        plt.legend(fontsize=10, loc='best')
        plt.ylim(0, 1.0)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
    else:
        # Single result - show all available top-m
        metrics = []
        values = []
        colors = []

        for attr in dir(data):
            if attr.startswith('accuracy_top'):
                try:
                    m_val = int(attr.replace('accuracy_top', ''))
                    metrics.append(f'Top-{m_val}')
                    values.append(getattr(data, attr))
                    colors.append(COLORS[len(metrics)-1 % len(COLORS)])
                except:
                    pass

        plt.bar(metrics, values, alpha=0.85, color=colors, edgecolor='black', linewidth=1.5)
        plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
        plt.title('Top-m Accuracy', fontsize=14, fontweight='bold')
        plt.ylim(0, 1.0)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

def plot_eta_distribution(evaluation_or_dict):
    """Plot distribution of η values - supports single or multi-experiment."""
    plt.figure(figsize=(10, 6))

    # Distinctive colors
    COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

    # Check if multi-experiment
    if isinstance(evaluation_or_dict, dict):
        # Multi-experiment: overlay distributions
        for i, (name, res) in enumerate(evaluation_or_dict.items()):
            exp_name = get_exp_name(res)
            eta_dist = res.evaluation.eta_top1_distribution
            color = COLORS[i % len(COLORS)]
            plt.hist(eta_dist, bins=30, alpha=0.6, label=exp_name, edgecolor='black', linewidth=0.5, color=color)

        plt.xlabel('η (Power Efficiency)', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title('Distribution of η Values (All Experiments)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    else:
        # Single experiment
        eta_dist = evaluation_or_dict.eta_top1_distribution
        plt.hist(eta_dist, bins=50, alpha=0.7, edgecolor='black', color='#377eb8', linewidth=1)
        plt.axvline(np.mean(eta_dist), color='#e41a1c', linestyle='--', linewidth=2.5, label=f'Mean: {np.mean(eta_dist):.4f}')
        plt.axvline(np.median(eta_dist), color='#4daf4a', linestyle='--', linewidth=2.5, label=f'Median: {np.median(eta_dist):.4f}')

        plt.xlabel('η (Power Efficiency)', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title('Distribution of η Values', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

def plot_baseline_comparison(evaluation_or_dict):
    """Compare ML model against baselines - supports multi-experiment."""

    # Distinctive colors for baselines
    BASELINE_COLORS = ['#2ca02c', '#ff7f0e', '#d62728', '#9467bd']  # Green, Orange, Red, Purple

    # Check if multi-experiment
    if isinstance(evaluation_or_dict, dict):
        # Multi-experiment: show all side by side
        fig, ax = plt.subplots(figsize=(14, 6))

        experiments = []
        ml_scores = []
        best_obs_scores = []
        random_m_scores = []
        random_1_scores = []

        for name, res in evaluation_or_dict.items():
            exp_name = get_exp_name(res)
            ev = res.evaluation
            experiments.append(exp_name)
            ml_scores.append(ev.eta_top1)
            best_obs_scores.append(ev.eta_best_observed)
            random_m_scores.append(ev.eta_random_M)
            random_1_scores.append(ev.eta_random_1)

        x = np.arange(len(experiments))
        width = 0.2

        ax.bar(x - 1.5*width, ml_scores, width, label='ML Model', color=BASELINE_COLORS[0], alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.bar(x - 0.5*width, best_obs_scores, width, label='Best of Observed', color=BASELINE_COLORS[1], alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.bar(x + 0.5*width, random_m_scores, width, label='Random M', color=BASELINE_COLORS[2], alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.bar(x + 1.5*width, random_1_scores, width, label='Random 1', color=BASELINE_COLORS[3], alpha=0.85, edgecolor='black', linewidth=0.5)

        ax.set_ylabel('η (Power Efficiency)', fontsize=12, fontweight='bold')
        ax.set_title('ML Model vs Baselines (All Experiments)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(experiments, rotation=45, ha='right')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
    else:
        # Single experiment
        evaluation = evaluation_or_dict
        plt.figure(figsize=(10, 6))

        methods = ['ML Model\n(Top-1)', 'Best of\nObserved', 'Random M', 'Random 1']
        values = [
            evaluation.eta_top1,
            evaluation.eta_best_observed,
            evaluation.eta_random_M,
            evaluation.eta_random_1
        ]

        bars = plt.bar(methods, values, alpha=0.85, color=BASELINE_COLORS, edgecolor='black', linewidth=1.5)

        for bar, val in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.ylabel('η (Power Efficiency)', fontsize=12, fontweight='bold')
        plt.title('ML Model vs Baselines', fontsize=14, fontweight='bold')
        plt.ylim(0, max(values) * 1.15)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

def plot_heatmap(probe_bank_or_dict):
    """Visualize probe phase configuration - shows first experiment if multi."""

    # Handle multi-experiment (use first)
    if isinstance(probe_bank_or_dict, dict):
        first_res = list(probe_bank_or_dict.values())[0]
        exp_name = get_exp_name(first_res)
        if hasattr(first_res, 'probe_bank'):
            probe_bank = first_res.probe_bank
        else:
            print("⚠️ Heatmap: Probe bank not available in results")
            return
        title_suffix = f" - {exp_name}"
    else:
        probe_bank = probe_bank_or_dict
        title_suffix = ""

    phases = probe_bank.phases[0]
    if hasattr(phases, 'cpu'):
        phases = phases.cpu().numpy()
    phases = phases.reshape(-1)

    N = len(phases)
    side = int(np.sqrt(N))

    if side * side == N:
        grid = phases.reshape(side, side)

        plt.figure(figsize=(8, 6))
        sns.heatmap(grid, cmap='twilight', annot=False, cbar_kws={'label': 'Phase (radians)'})
        plt.title(f"Probe Phase Configuration ({side}x{side}){title_suffix}", fontsize=14, fontweight='bold')
        plt.xlabel('Element Index', fontsize=12)
        plt.ylabel('Element Index', fontsize=12)
        plt.tight_layout()
    else:
        plt.figure(figsize=(12, 4))
        plt.plot(phases, marker='o', markersize=3, linewidth=1)
        plt.xlabel('Element Index', fontsize=12)
        plt.ylabel('Phase (radians)', fontsize=12)
        plt.title(f"Probe Phase Configuration (N={N}){title_suffix}", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

def plot_violin(results_dict):
    """Violin plot of η distributions."""
    data = []
    for name, res in results_dict.items():
        eta_dist = res.evaluation.eta_top1_distribution
        for val in eta_dist:
            data.append({'Experiment': name, 'η': val})

    if not data:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No distribution data available.', ha='center', va='center', fontsize=14)
        plt.axis('off')
        return

    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x='Experiment', y='η', inner='box')
    plt.xticks(rotation=45, ha='right')
    plt.title('η Distribution Across Experiments', fontsize=14, fontweight='bold')
    plt.ylabel('η (Power Efficiency)', fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

def plot_box(results_dict):
    """Box plot of η distributions."""
    data = []
    for name, res in results_dict.items():
        eta_dist = res.evaluation.eta_top1_distribution
        for val in eta_dist:
            data.append({'Experiment': name, 'η': val})

    if not data:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No distribution data available.', ha='center', va='center', fontsize=14)
        plt.axis('off')
        return

    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Experiment', y='η')
    plt.xticks(rotation=45, ha='right')
    plt.title('η Distribution (Box Plot)', fontsize=14, fontweight='bold')
    plt.ylabel('η (Power Efficiency)', fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

def plot_scatter(results_dict):
    """Scatter plot of accuracy vs efficiency."""
    data = []
    for name, res in results_dict.items():
        data.append({
            'Experiment': name,
            'η': res.evaluation.eta_top1,
            'Accuracy': res.evaluation.accuracy_top1,
            'M': res.config.get('M', 0)
        })

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['Accuracy'], df['η'],
                         s=df['M']*20, alpha=0.6,
                         c=range(len(df)), cmap='viridis',
                         edgecolors='black', linewidth=1.5)

    for _, row in df.iterrows():
        plt.annotate(row['Experiment'],
                    (row['Accuracy'], row['η']),
                    fontsize=8, alpha=0.7)

    plt.xlabel('Accuracy (Top-1)', fontsize=12)
    plt.ylabel('η (Power Efficiency)', fontsize=12)
    plt.title('Accuracy vs Efficiency Trade-off', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Experiment Index')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def plot_cdf(evaluation_or_dict):
    """Plot cumulative distribution function - supports multi-experiment."""
    plt.figure(figsize=(10, 6))

    # Distinctive colors
    COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

    # Check if multi-experiment
    if isinstance(evaluation_or_dict, dict):
        # Multi-experiment: overlay CDFs
        for i, (name, res) in enumerate(evaluation_or_dict.items()):
            exp_name = get_exp_name(res)
            eta_dist = res.evaluation.eta_top1_distribution
            sorted_eta = np.sort(eta_dist)
            cdf = np.arange(1, len(sorted_eta)+1) / len(sorted_eta)
            color = COLORS[i % len(COLORS)]
            plt.plot(sorted_eta, cdf, linewidth=2.5, label=exp_name, alpha=0.9, color=color)

        plt.xlabel('η (Power Efficiency)', fontsize=12, fontweight='bold')
        plt.ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
        plt.title('Cumulative Distribution Function (All Experiments)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    else:
        # Single experiment
        eta_dist = evaluation_or_dict.eta_top1_distribution
        sorted_eta = np.sort(eta_dist)
        cdf = np.arange(1, len(sorted_eta)+1) / len(sorted_eta)

        plt.plot(sorted_eta, cdf, linewidth=2.5, color='#377eb8')
        plt.axvline(np.median(sorted_eta), color='#e41a1c', linestyle='--',
                    label=f'Median: {np.median(sorted_eta):.4f}', linewidth=2)

        plt.xlabel('η (Power Efficiency)', fontsize=12, fontweight='bold')
        plt.ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
        plt.title('Cumulative Distribution Function (CDF) of η', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

def plot_top_m_efficiency(evaluation_or_dict):
    """NEW: Plot efficiency (η) for different top-m selections."""
    plt.figure(figsize=(10, 6))

    # Distinctive colors
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    if isinstance(evaluation_or_dict, dict):
        # Multi-experiment comparison
        fig, ax = plt.subplots(figsize=(12, 6))

        experiments = []
        eta_by_m = {}  # {m_value: [eta1, eta2, ...]}

        # Collect all available top-m efficiencies
        first_res = list(evaluation_or_dict.values())[0]
        available_m = []
        for attr in dir(first_res.evaluation):
            if attr.startswith('eta_top') and not attr.endswith('distribution'):
                try:
                    m_val = int(attr.replace('eta_top', ''))
                    available_m.append(m_val)
                except:
                    pass
        available_m.sort()

        # Collect data
        for name, res in evaluation_or_dict.items():
            exp_name = get_exp_name(res)
            experiments.append(exp_name)

            for m in available_m:
                attr_name = f'eta_top{m}'
                if hasattr(res.evaluation, attr_name):
                    if m not in eta_by_m:
                        eta_by_m[m] = []
                    eta_by_m[m].append(getattr(res.evaluation, attr_name))

        x = np.arange(len(experiments))
        width = 0.8 / len(eta_by_m)

        # Plot bars
        for i, (m, etas) in enumerate(sorted(eta_by_m.items())):
            offset = (i - len(eta_by_m)/2 + 0.5) * width
            color = COLORS[i % len(COLORS)]
            ax.bar(x + offset, etas, width, label=f'Top-{m}', alpha=0.85, color=color, edgecolor='black', linewidth=0.5)

        ax.set_ylabel('η (Power Efficiency)', fontsize=12, fontweight='bold')
        ax.set_title('Efficiency (η) for Different Top-m Selections', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(experiments, rotation=45, ha='right')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

    else:
        # Single experiment
        evaluation = evaluation_or_dict
        m_values = []
        eta_values = []
        colors = []

        for attr in dir(evaluation):
            if attr.startswith('eta_top') and not attr.endswith('distribution'):
                try:
                    m_val = int(attr.replace('eta_top', ''))
                    m_values.append(m_val)
                    eta_values.append(getattr(evaluation, attr))
                    colors.append(COLORS[len(m_values)-1 % len(COLORS)])
                except:
                    pass

        # Sort by m value
        sorted_data = sorted(zip(m_values, eta_values, colors))
        m_values, eta_values, colors = zip(*sorted_data) if sorted_data else ([], [], [])

        m_labels = [f'Top-{m}' for m in m_values]
        plt.bar(m_labels, eta_values, alpha=0.85, color=colors, edgecolor='black', linewidth=1.5)
        plt.ylabel('η (Power Efficiency)', fontsize=12, fontweight='bold')
        plt.title('Efficiency for Different Top-m', fontsize=14, fontweight='bold')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

# =============================================================================
# PLOT REGISTRY - ALL IMPLEMENTED
# =============================================================================
EXTENDED_PLOT_REGISTRY = {
    # Training plots
    'training_curves': plot_training_curves,

    # Sweep plots (multi-experiment)
    'eta_vs_M': plot_eta_vs_M,
    'eta_vs_K': plot_eta_vs_K,
    'eta_vs_N': plot_eta_vs_N,

    # Analysis plots - ALL IMPLEMENTED
    'top_m_comparison': plot_top_m_comparison,
    'top_m_efficiency': plot_top_m_efficiency,  # NEW: Top-m vs Efficiency
    'eta_distribution': plot_eta_distribution,
    'baseline_comparison': plot_baseline_comparison,
    'heatmap': plot_heatmap,
    'cdf': plot_cdf,

    # Statistical plots (multi-experiment)
    'violin': plot_violin,
    'box': plot_box,
    'scatter': plot_scatter,
}