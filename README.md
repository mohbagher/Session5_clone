# RIS Probe-Based Control with Limited Probing

A machine learning framework for optimizing Reconfigurable Intelligent Surface (RIS) phase configurations using limited probe measurements.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Problem Overview

In RIS-assisted wireless communications:
- **N** RIS elements (reflective surface) - default: 32
- **K** pre-defined probe configurations (phase patterns) - default: 64
- **M** sensing budget (probes we can actually measure) - default: 8

**Challenge**: Predict the best probe among all K options while only observing M << K measurements.

**Our Solution**: Train a neural network to learn patterns from limited observations and predict the optimal probe configuration.

## 🏗️ Architecture

```
Input: [masked_powers, binary_mask] ∈ ℝ^{2K}
      ↓
   MLP (512 → 256 → 128)
      ↓
Output: logits ∈ ℝ^K (probability over all probes)
```

The model uses a **Masked Vector Approach**:
- `masked_powers`: K-dimensional vector with observed powers at their indices, zeros elsewhere
- `binary_mask`: K-dimensional vector with 1s at observed indices, 0s elsewhere

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/mohbagher/Probe_Based_ML_Session5.git
cd Probe_Based_ML_Session5

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- Python ≥ 3.8
- PyTorch ≥ 1.12.0
- NumPy ≥ 1.21.0
- Matplotlib ≥ 3.5.0
- Seaborn ≥ 0.11.0
- SciPy ≥ 1.7.0
- tqdm ≥ 4.62.0
- pandas ≥ 1.3.0

## 🚀 Quick Start

### Option 1: Using the New Experiment Toolkit (Recommended)

```python
from experiment_toolkit import ConfigBuilder, ExperimentRunner, PlotGallery

# Fluent API for configuration
config = (ConfigBuilder()
    .system(N=32, K=64, M=8, probe_type='continuous')
    .data(n_train=50000, n_val=5000, n_test=5000, seed=42)
    .model_by_name('Baseline_MLP')  # Use pre-defined architecture
    .training(epochs=50, batch_size=128, lr=1e-3)
    .build())

# Run complete experiment
experiment = ExperimentRunner.run(config, verbose=True)

# Generate plots
PlotGallery.show('cdf', experiment['results'])
PlotGallery.show(['eta_distribution', 'training_curves'], experiment['results'], experiment['history'])
```

### Option 2: Interactive Menu

```bash
python experiment_runner.py
```

This opens an interactive menu where you can:
- Select individual tasks (1-14)
- Run all tasks (0)
- Change parameters (S)
- Custom task selection (99)

When running multiple tasks in sequence, the runner waits 5 seconds between tasks (press Enter to skip the wait).

### Option 3: Command Line

```bash
# Run specific tasks
python experiment_runner.py --task 1,3,6 --N 32 --K 64 --M 8

# Run all tasks
python experiment_runner.py --task 0

# Custom parameters
python experiment_runner.py --task 6 --N 64 --K 128 --M 16 --seed 42
```

### Option 4: Python API

```python
from experiments.tasks.task_a1_binary import run_task_a1
from experiments.tasks.task_b1_m_variation import run_task_b1

# Run binary probe analysis
result = run_task_a1(N=32, K=64, M=8, seed=42)

# Run M variation study
result = run_task_b1(N=32, K=64, M=8, seed=42)
```

### Option 5: Original Training Script

```bash
python main.py --N 32 --K 64 --M 8 --epochs 100
```

## 📋 Experiment Tasks

The framework includes 14 organized research tasks:

### Phase A: Probe Design
| Task | Command | Description |
|------|---------|-------------|
| A1 | `--task 1` | Binary (1-bit) probe generation and analysis |
| A2 | `--task 2` | Hadamard structured probe patterns |
| A3 | `--task 3` | Probe diversity comparison across all types |
| A4 | `--task 4` | Sobol low-discrepancy probe analysis |
| A5 | `--task 5` | Halton low-discrepancy probe analysis |

### Phase B: Limited Probing Analysis
| Task | Command | Description |
|------|---------|-------------|
| B1 | `--task 6` | M variation study (M ∈ {2,4,8,16,32}) |
| B2 | `--task 7` | Top-m selection performance |
| B3 | `--task 8` | ML vs baseline comparison |

### Phase C: Scaling Study
| Task | Command | Description |
|------|---------|-------------|
| C1 | `--task 9` | Scale K (number of probes) |
| C2 | `--task 10` | Phase resolution comparison |

### Phase D: Quality Control
| Task | Command | Description |
|------|---------|-------------|
| D1 | `--task 11` | Seed variation (reproducibility) |
| D2 | `--task 12` | Training sanity checks |

### Phase E: Documentation
| Task | Command | Description |
|------|---------|-------------|
| E1 | `--task 13` | Generate one-page summary |
| E2 | `--task 14` | Generate comparison plots |

## 📊 Key Metrics

- **η (eta)**: Power ratio = P_selected / P_best_probe
  - η = 1.0 → Perfect (found the best probe)
  - η = 0.5 → Selected probe gives half the optimal power

- **Top-m Accuracy**: Fraction where oracle best probe is in top-m predictions

- **Baselines**:
  - Random 1/K: Pick 1 probe randomly
  - Random M/K: Best of random M probes
  - Best Observed: Best among actually measured M probes
  - Oracle: Theoretical best (η = 1.0)

## 📁 Project Structure

```
Probe_Based_ML_Session5/
├── experiment_runner.py      # 🎯 Main interactive experiment runner
├── main.py                   # Original training script
├── run.py                    # Quick experiment scripts
│
├── config.py                 # Configuration dataclasses
├── model.py                  # MLP neural network architecture
├── model_registry.py         # 🆕 Model architecture registry
├── plot_registry.py          # 🆕 Centralized plotting functions
├── experiment_toolkit.py     # 🆕 High-level experiment API
├── training.py               # Training loop with early stopping
├── evaluation.py             # Metrics computation & baselines
├── data_generation.py        # Channel simulation & datasets
├── utils.py                  # Visualization & utilities
│
├── experiments/              # 🧪 Experiment framework
│   ├── __init__.py
│   ├── probe_generators.py   # All probe types (continuous, binary, Hadamard)
│   ├── diversity_analysis.py # Diversity metrics
│   └── tasks/                # Individual task implementations
│       ├── task_a1_binary.py
│       ├── task_a2_hadamard.py
│       ├── task_a3_diversity.py
│       ├── task_a4_sobol.py
│       ├── task_a5_halton.py
│       ├── task_b1_m_variation.py
│       ├── task_b2_top_m.py
│       ├── task_b3_baselines.py
│       └── ... (more tasks)
│
├── results/                  # 📈 Output directory (auto-created)
│   ├── A1_binary_probes/
│   ├── A2_hadamard_probes/
│   ├── B1_M_variation/
│   └── ...
│
├── EXPERIMENT_RUNNER.md      # Detailed experiment documentation
├── IMPLEMENTATION_SUMMARY.md # Technical implementation details
├── requirements.txt          # Dependencies
├── LICENSE                   # MIT License
└── README.md                 # This file
```

## 🆕 New Features: Registry System

### Model Registry

The framework now includes a centralized model architecture registry for easy extension:

```python
from model_registry import list_models, get_model_architecture, register_model

# List available models
models = list_models()  
# ['Baseline_MLP', 'Deep_MLP', 'Tiny_MLP', 'Ultra_Deep', 'Wide_Deep', ...]

# Get a model architecture
arch = get_model_architecture('Baseline_MLP')  # Returns [256, 128]

# Register your own custom model
register_model('My_Custom_MLP', [512, 256, 128, 64])
```

**Pre-defined Models:**
- **Baseline_MLP**: [256, 128] - Standard model
- **Deep_MLP**: [512, 512, 256] - High capacity
- **Tiny_MLP**: [64, 32] - Lightweight
- **Ultra_Deep**: [1024, 512, 256, 128, 64] - Very deep
- **Wide_Deep**: [1024, 1024, 512, 512] - Wide architecture
- **Lightweight**: [128, 64] - Efficient
- **Minimal**: [32, 16] - Minimal capacity
- **Experimental_A**: [512, 256, 256, 128] - Research model
- **Experimental_B**: [768, 384, 192, 96] - Research model

### Plot Registry

Unified interface to all visualization functions:

```python
from plot_registry import list_plots, get_plot_function

# List available plot types
plots = list_plots()
# ['training_curves', 'eta_distribution', 'cdf', 'violin', 'heatmap', ...]

# Use a plot function
plot_func = get_plot_function('cdf')
plot_func(results, save_path='my_cdf.png')
```

**Available Plots:**
- `training_curves` - Training history visualization
- `eta_distribution` - Performance distribution
- `top_m_comparison` - Top-m accuracy comparison
- `baseline_comparison` - ML vs baselines
- `cdf` - Cumulative distribution function
- `violin` - Violin plot for multiple models
- `heatmap` - Probe phase heatmap
- `scatter` - Scatter comparison plot
- `box` - Box plot comparison
- `correlation_matrix` - Probe similarity matrix

### Experiment Toolkit

High-level fluent API for running experiments:

```python
from experiment_toolkit import ConfigBuilder, ExperimentRunner, PlotGallery

# Build configuration with fluent interface
config = (ConfigBuilder()
    .system(N=32, K=64, M=8, probe_type='hadamard')
    .data(n_train=50000, seed=42)
    .model_by_name('Deep_MLP')
    .training(epochs=50, batch_size=128)
    .build())

# Run experiment
results = ExperimentRunner.run(config, verbose=True)

# Compare multiple models
results = ExperimentRunner.compare_models(
    config_base=config,
    model_names=['Baseline_MLP', 'Deep_MLP', 'Tiny_MLP']
)

# Show plots
PlotGallery.show(['cdf', 'heatmap'], results)
PlotGallery.show('violin', results_dict)
```

## ⚠️ Important Changes

### Deprecated Parameters

- **`probe_bank_method`** in `SystemConfig` is deprecated. Use **`probe_type`** instead.
  - Old: `system={'probe_bank_method': 'hadamard'}`
  - New: `system={'probe_type': 'hadamard'}`
  - The old parameter still works with a deprecation warning for backward compatibility.

### Parameter Naming Standardization

- Training configuration now uses **`n_epochs`** instead of `num_epochs`
  - Old: `training={'num_epochs': 50}`
  - New: `training={'n_epochs': 50}`

### ProbeBank Import Location

- `ProbeBank` class is now only in `experiments.probe_generators`
  - Import: `from experiments.probe_generators import ProbeBank, get_probe_bank`
  - The duplicate definition in `data_generation.py` has been removed

## 📈 Results Structure

Each task saves results to its own folder:

```
results/
├── A1_binary_probes/
│   ├── plots/
│   │   ├── phase_heatmap.png
│   │   └── phase_histogram.png
│   └── metrics.txt
├── B1_M_variation/
│   ├── plots/
│   │   └── eta_vs_M.png
│   └── metrics.txt
└── comparison_all_models/
    ├── master_comparison.png
    └── summary.txt
```

## 🔬 Probe Types

The framework supports 4 probe types:

| Type | Phases | Description | Use Case |
|------|--------|-------------|----------|
| Continuous | [0, 2π) | Random phases | Baseline, theoretical limit |
| Binary (1-bit) | {0, π} | Two-level quantization | Simple hardware |
| 2-bit | {0, π/2, π, 3π/2} | Four-level quantization | Balanced complexity |
| Hadamard | {0, π} structured | Orthogonal patterns | Maximum diversity |
| Sobol | [0, 2π) low-discrepancy | Quasi-random coverage | Uniform space-filling |
| Halton | [0, 2π) low-discrepancy | Quasi-random coverage | Uniform space-filling |

## 💻 Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | None | Task(s) to run (1-14, 0=all, or comma-separated) |
| `--N` | 32 | Number of RIS elements |
| `--K` | 64 | Total probes in bank |
| `--M` | 8 | Sensing budget |
| `--seed` | 42 | Random seed |
| `--results-dir` | results | Base directory for outputs |

## 📚 Documentation

- **[EXPERIMENT_RUNNER.md](EXPERIMENT_RUNNER.md)**: Complete guide to the experiment framework
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)**: Technical implementation details

## 🎓 For PhD Research

### Recommended Experiment Sequence

1. **Quick validation**: `python experiment_runner.py --task 1`
2. **Probe design analysis**: `python experiment_runner.py --task 1,2,3,4,5`
3. **Core ML experiments**: `python experiment_runner.py --task 6,7,8`
4. **Full study**: `python experiment_runner.py --task 0`

### Key Results for Publications

- **η vs M curve**: Shows performance vs measurement overhead
- **ML vs Baselines**: Demonstrates ML advantage
- **Probe diversity**: Justifies structured probe design

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@software{ris_probe_ml,
  author = {Bagher, Mohammad},
  title = {RIS Probe-Based Control with Limited Probing},
  year = {2024},
  url = {https://github.com/mohbagher/Probe_Based_ML_Session5}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or collaboration, please open an issue on GitHub.
