import json
import os

# Define the notebook content structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_cell(source_code, cell_type="code"):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source_code.split("\n")]
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    notebook["cells"].append(cell)

# ---------------------------------------------------------
# CELL 1: Imports and Setup
# ---------------------------------------------------------
add_cell("""# %load_ext autoreload
# %autoreload 2

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

# Add parent directory to path to see project modules
sys.path.append(os.path.abspath(os.path.join('..')))

# Import Project Modules
from config import Config
from data_generation import (
    create_masked_input, 
    compute_probe_powers, 
    compute_optimal_power, 
    select_probing_subset
)
# Note: We import ProbeBank from experiments.probe_generators as per the latest fix
from experiments.probe_generators import (
    ProbeBank,
    generate_sobol_probes,
    generate_hadamard_probes,
    generate_binary_probes,
    generate_continuous_probes
)
from model import create_model, LimitedProbingMLP
from training import train
from evaluation import evaluate_model
from utils import plot_training_history

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
print("✅ Project Environment Configured.")""")

# ---------------------------------------------------------
# CELL 2: Master Dashboard (Config)
# ---------------------------------------------------------
add_cell("""# ==========================================
# 🎛️ MASTER EXPERIMENT DASHBOARD
# ==========================================

# 1. Hardware & Physics
N = 32               # Number of RIS elements
K = 64               # Total Codebook Size
M = 8                # Sensing Budget (Probes measured)
PHASE_BITS = 1       # Phase Resolution: 1=Binary, 2=Quad(0,π/2..), None=Continuous

# 2. Channel Environment
CHANNEL_TYPE = "Sparse"   # Options: "Rayleigh" (Rich Scattering) or "Sparse" (mmWave)
SPARSITY_PATHS = 3        # Number of dominant paths (Only for Sparse mode)

# 3. Probe Strategy
PROBE_TYPE = "sobol"      # Options: "sobol", "hadamard", "binary", "random"

# 4. Training
N_TRAIN = 20000
N_TEST = 2000
BATCH_SIZE = 128
EPOCHS = 30

# 5. Model Competition
# Define the list of models you want to compare in this run
MODELS_TO_COMPARE = {
    "Baseline_MLP": [256, 128],         # Standard model
    "Deep_MLP":     [512, 512, 256],    # High capacity model
    "Tiny_MLP":     [64, 32]            # Hardware efficient model
}

print(f"🚀 Dashboard Ready: {CHANNEL_TYPE} Channel | {PROBE_TYPE.upper()} Probes ({PHASE_BITS}-bit)")""")

# ---------------------------------------------------------
# CELL 3: Custom Physics Functions
# ---------------------------------------------------------
add_cell("""# ==========================================
# 🔧 CUSTOM PHYSICS ENGINES
# ==========================================

def quantize_phases(phases, n_bits):
    \"\"\"
    Quantizes continuous phases to n_bits resolution.
    \"\"\"
    if n_bits is None:
        return phases
    
    n_levels = 2 ** n_bits
    step = (2 * np.pi) / n_levels
    
    # Quantize to nearest level
    quantized = np.round(phases / step) * step
    return quantized

def generate_sparse_channel(N, n_paths, rng):
    \"\"\"
    Generates a Sparse (mmWave-like) channel.
    \"\"\"
    h = np.zeros(N, dtype=np.complex64)
    g = np.zeros(N, dtype=np.complex64)
    
    # Select random dominant paths
    path_indices = rng.choice(N, size=n_paths, replace=False)
    
    # Assign strong gains to these paths
    h[path_indices] = (rng.randn(n_paths) + 1j * rng.randn(n_paths)) * 5.0
    g[path_indices] = (rng.randn(n_paths) + 1j * rng.randn(n_paths)) * 5.0
    
    # Add small background noise
    h += (rng.randn(N) + 1j * rng.randn(N)) * 0.1
    g += (rng.randn(N) + 1j * rng.randn(N)) * 0.1
    
    return h, g

def custom_data_generator(probe_bank, n_samples, M, channel_type, n_paths, seed=42):
    \"\"\"
    Custom data generator supporting 'Sparse' and 'Rayleigh' modes.
    \"\"\"
    rng = np.random.RandomState(seed)
    K = probe_bank.K
    
    masked_powers = np.zeros((n_samples, K), dtype=np.float32)
    masks = np.zeros((n_samples, K), dtype=np.float32)
    powers_full = np.zeros((n_samples, K), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int64)
    optimal_powers = np.zeros(n_samples, dtype=np.float32)
    observed_indices = np.zeros((n_samples, M), dtype=np.int64)

    print(f"   Generating {n_samples} samples ({channel_type} mode)...")
    
    for i in range(n_samples):
        # 1. Generate Channel
        if channel_type == "Sparse":
            h, g = generate_sparse_channel(probe_bank.N, n_paths, rng)
        else:
            h = (rng.randn(probe_bank.N) + 1j * rng.randn(probe_bank.N))
            g = (rng.randn(probe_bank.N) + 1j * rng.randn(probe_bank.N))

        # 2. Compute Powers
        p_full = compute_probe_powers(h, g, probe_bank)
        powers_full[i] = p_full
        labels[i] = np.argmax(p_full)
        optimal_powers[i] = compute_optimal_power(h, g)

        # 3. Mask Input
        obs_idx = select_probing_subset(K, M, rng)
        observed_indices[i] = obs_idx
        mp, m = create_masked_input(p_full, obs_idx, K)
        masked_powers[i] = mp
        masks[i] = m
        
    return {
        'masked_powers': masked_powers, 'masks': masks, 
        'powers_full': powers_full, 'labels': labels, 
        'observed_indices': observed_indices, 'optimal_powers': optimal_powers
    }

print("✅ Custom Physics Engines Loaded.")""")

# ---------------------------------------------------------
# CELL 4: Probe Generation
# ---------------------------------------------------------
add_cell("""# ==========================================
# 📡 PROBE GENERATION & VIZ
# ==========================================

# 1. Generate Base Probes
if PROBE_TYPE == "sobol":
    raw_phases = generate_sobol_probes(N, K)
elif PROBE_TYPE == "hadamard":
    raw_phases = generate_hadamard_probes(N, K)
elif PROBE_TYPE == "continuous":
    raw_phases = generate_continuous_probes(N, K)
else:
    raw_phases = generate_binary_probes(N, K)

# 2. Apply Custom Quantization
final_phases = quantize_phases(raw_phases, PHASE_BITS)

# 3. Create ProbeBank Object
custom_bank = ProbeBank(phases=final_phases, K=K, N=N)

# 4. Visualize
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.heatmap(final_phases[:40], cmap="viridis", cbar=True)
plt.title(f"{PROBE_TYPE.title()} Probes ({PHASE_BITS}-bit)\\n(First 40 Probes)")
plt.xlabel("RIS Elements")
plt.ylabel("Probe Index")

plt.subplot(1, 2, 2)
# Diversity Metric: Variance of phases across elements
div_metric = np.var(final_phases, axis=0)
plt.bar(range(N), div_metric, color='purple', alpha=0.7)
plt.title("Probe Diversity (Element-wise Variance)")
plt.xlabel("RIS Element Index")
plt.ylabel("Phase Variance")
plt.tight_layout()
plt.show()""")

# ---------------------------------------------------------
# CELL 5: Data Generation
# ---------------------------------------------------------
add_cell("""# ==========================================
# 🏭 DATA FACTORY
# ==========================================

# Generate Train Data
train_raw = custom_data_generator(custom_bank, N_TRAIN, M, CHANNEL_TYPE, SPARSITY_PATHS, seed=42)

# Generate Test Data (Different Seed)
test_raw  = custom_data_generator(custom_bank, N_TEST,  M, CHANNEL_TYPE, SPARSITY_PATHS, seed=999)

# Wrap in PyTorch Dataset
class QuickDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict):
        self.inputs = torch.FloatTensor(np.concatenate([data_dict['masked_powers'], data_dict['masks']], axis=1))
        self.labels = torch.LongTensor(data_dict['labels'])
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.inputs[idx], self.labels[idx]

train_dataset = QuickDataset(train_raw)
test_dataset  = QuickDataset(test_raw)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"✅ Data Ready: {len(train_dataset)} Train, {len(test_dataset)} Test")""")

# ---------------------------------------------------------
# CELL 6: Training Loop
# ---------------------------------------------------------
add_cell("""# ==========================================
# 🥊 MODEL COMPETITION LOOP
# ==========================================

competition_results = {}

for model_name, hidden_layers in MODELS_TO_COMPARE.items():
    print(f"\\n🥊 Training Model: {model_name} {hidden_layers}...")
    
    # 1. Config for this specific model
    current_config = Config()
    current_config.system.N = N
    current_config.system.K = K
    current_config.system.M = M
    current_config.model.hidden_sizes = hidden_layers
    current_config.training.n_epochs = EPOCHS
    current_config.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Create & Train
    model = create_model(current_config)
    
    # Dummy metadata for validation hook
    meta = {'val_powers_full': test_raw['powers_full'], 'val_labels': test_raw['labels']}
    
    trained_model, history = train(model, train_loader, test_loader, current_config, meta)
    
    # 3. Evaluate
    eval_results = evaluate_model(
        trained_model, test_loader, current_config,
        test_raw['powers_full'], test_raw['labels'], 
        test_raw['observed_indices'], test_raw['optimal_powers']
    )
    
    competition_results[model_name] = {
        'results': eval_results,
        'history': history,
        'eta_top1': eval_results.eta_top1
    }
    print(f"🏅 {model_name} Final η (Top-1): {eval_results.eta_top1:.4f}")""")

# ---------------------------------------------------------
# CELL 7: Visualization
# ---------------------------------------------------------
add_cell("""# ==========================================
# 📊 FINAL RESULTS DASHBOARD
# ==========================================

names = list(competition_results.keys())
scores = [res['eta_top1'] for res in competition_results.values()]
baselines = competition_results[names[0]]['results'].eta_best_observed

plt.figure(figsize=(14, 6))

# 1. Bar Chart Comparison
plt.subplot(1, 2, 1)
x = np.arange(len(names))
plt.bar(x, scores, 0.5, label='ML Model', color='steelblue')
plt.axhline(baselines, color='orange', linestyle='--', linewidth=2, label='Best Observed (Baseline)')
plt.ylabel('Power Ratio (η)')
plt.title(f'Model Accuracy (Top-1)\\n{CHANNEL_TYPE} Channel | {PROBE_TYPE.title()}')
plt.xticks(x, names)
plt.legend()
plt.ylim(0, 1.1)

# 2. CDF Curves
plt.subplot(1, 2, 2)
for name, data in competition_results.items():
    sorted_eta = np.sort(data['results'].eta_top1_distribution)
    cdf = np.arange(1, len(sorted_eta) + 1) / len(sorted_eta)
    plt.plot(sorted_eta, cdf, linewidth=2, label=f"{name}")

# Baseline CDF
base_eta = np.sort(competition_results[names[0]]['results'].eta_best_observed_distribution)
base_cdf = np.arange(1, len(base_eta) + 1) / len(base_eta)
plt.plot(base_eta, base_cdf, '--', color='orange', label='Best Observed')

plt.xlabel('Power Efficiency (η)')
plt.ylabel('CDF')
plt.title(f'Reliability Analysis (CDF)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()""")

# ---------------------------------------------------------
# Save the notebook
# ---------------------------------------------------------
output_path = os.path.join("notebooks", "Master_Experiment.ipynb")
os.makedirs("notebooks", exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"✅ Notebook successfully created at: {output_path}")
print("Run 'jupyter notebook' and open notebooks/Master_Experiment.ipynb")
