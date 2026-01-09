# RIS PhD Ultimate Dashboard - Documentation

## Overview

The RIS PhD Ultimate Dashboard is a comprehensive, fully customizable Jupyter notebook interface for conducting research on Reconfigurable Intelligent Surface (RIS) probe-based machine learning systems. It provides a professional-grade platform with complete control over all experimental parameters.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Parameter Reference](#parameter-reference)
- [Features](#features)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## Installation

### Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab

### Dependencies

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Required packages:
- numpy >= 1.21.0
- torch >= 1.12.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- tqdm >= 4.62.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- ipywidgets >= 8.0.0
- pyyaml >= 6.0

### Setup

1. Clone the repository
2. Install dependencies
3. Launch Jupyter: `jupyter notebook`
4. Open `notebooks/RIS_PhD_Ultimate_Dashboard.ipynb`

## Quick Start

### Basic Workflow

1. **Setup** - Run Cells 1-2 to initialize the system
2. **Configure** - Use the 5-tab interface (Cell 3-7) to set parameters
3. **Run** - Click "🚀 RUN EXPERIMENT" button (Cell 8)
4. **View** - See results in Cells 9-10
5. **Export** - Use Cell 11 to save results

### Example: Run a Simple Experiment

```python
# After initializing (Cells 1-2)
# Set parameters via widgets or programmatically:
widgets_dict['N'].value = 32
widgets_dict['K'].value = 64
widgets_dict['M'].value = 8
widgets_dict['model_preset'].value = 'Baseline_MLP'

# Click "RUN EXPERIMENT" button or:
on_run_experiment_clicked(None)
```

## Architecture

### Module Structure

```
dashboard/
├── __init__.py            # Main module exports
├── widgets.py             # All widget definitions (5 tabs)
├── callbacks.py           # Widget event handlers
├── validators.py          # Input validation functions
├── config_manager.py      # Configuration save/load
├── experiment_runner.py   # Experiment execution engine
└── plots.py               # 25+ visualization functions
```

### Data Flow

```
Widgets → Config Dict → Validator → Experiment Runner → Results → Plots
   ↓                                      ↓
Config File                          Results File
```

## Parameter Reference

### Tab 1: System & Physics

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| **N** | int | 32 | 4-256 | Number of RIS elements. More elements provide better beamforming control but increase computational complexity. |
| **K** | int | 64 | 4-512 | Total number of probes in the codebook. Larger codebook provides better coverage but makes prediction harder. |
| **M** | int | 8 | 1-K | Sensing budget - number of probes measured per channel realization. The M/K ratio is critical for performance. |
| **P_tx** | float | 1.0 | 0.1-10.0 | Transmit power (normalized). Scales all received powers proportionally. |
| **sigma_h_sq** | float | 1.0 | 0.1-10.0 | BS-RIS channel variance. Controls fading strength of base station to RIS link. |
| **sigma_g_sq** | float | 1.0 | 0.1-10.0 | RIS-UE channel variance. Controls fading strength of RIS to user equipment link. |
| **phase_mode** | str | "continuous" | continuous/discrete | Phase quantization mode. "continuous" for theoretical studies, "discrete" for hardware-realistic scenarios. |
| **phase_bits** | int | 3 | 1-8 | Number of bits for phase quantization (only when phase_mode="discrete"). More bits = finer control. |
| **probe_type** | str | "continuous" | 6 options | Probe generation method: continuous, binary, 2bit, hadamard, sobol, halton. |

### Tab 2: Model Architecture

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **model_preset** | str | "Baseline_MLP" | Pre-defined architecture or "Custom" |
| **num_layers** | int | 3 | Number of hidden layers (Custom only) |
| **hidden_sizes** | List[int] | [512,256,128] | Size of each hidden layer |
| **dropout_prob** | float | 0.1 | Dropout probability (0.0-0.8) for regularization |
| **use_batch_norm** | bool | True | Whether to use batch normalization |
| **activation_function** | str | "ReLU" | Activation: ReLU, LeakyReLU, GELU, ELU, Tanh |
| **weight_init** | str | "xavier_uniform" | Weight initialization method |

### Tab 3: Training Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **n_train** | int | 50000 | Number of training samples |
| **n_val** | int | 5000 | Number of validation samples |
| **n_test** | int | 5000 | Number of test samples |
| **seed** | int | 42 | Random seed for reproducibility |
| **normalize_input** | bool | True | Whether to normalize input features |
| **normalization_type** | str | "mean" | Type: mean, std, or log |
| **batch_size** | int | 128 | Training batch size (32-512) |
| **learning_rate** | float | 1e-3 | Initial learning rate (1e-5 to 1e-1) |
| **weight_decay** | float | 1e-4 | L2 regularization strength |
| **n_epochs** | int | 50 | Maximum number of epochs |
| **early_stop_patience** | int | 10 | Epochs without improvement before stopping |
| **optimizer** | str | "Adam" | Optimizer: Adam, AdamW, SGD, RMSprop |
| **scheduler** | str | "ReduceLROnPlateau" | LR scheduler or "None" |

### Tab 4: Evaluation & Comparison

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **top_m_values** | List[int] | [1,2,4,8] | Top-m accuracy values to compute |
| **compare_multiple_models** | bool | False | Enable multi-model comparison |
| **models_to_compare** | List[str] | [] | List of model names to compare |
| **multi_seed_runs** | bool | False | Enable multi-seed experiments |
| **num_seeds** | int | 3 | Number of seeds for statistical analysis |
| **compute_confidence_intervals** | bool | False | Calculate 95% confidence intervals |

### Tab 5: Visualization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **selected_plots** | List[str] | see below | Plot types to generate |
| **figure_format** | str | "png" | Output format: png, pdf, svg |
| **dpi** | int | 150 | Resolution (72-300) |
| **color_palette** | str | "viridis" | Color scheme for plots |
| **save_plots** | bool | True | Save plots to disk |
| **output_dir** | str | "results/" | Output directory path |

## Features

### 1. Interactive Widget System

- **5 Tabs** for organized parameter access
- **Real-time validation** with error messages
- **Dynamic UI** - widgets enable/disable based on selections
- **Parameter preview** - see model size before training

### 2. Model Architectures

19 pre-defined architectures:

**Standard Models:**
- Baseline_MLP: [256, 128]
- Deep_MLP: [512, 512, 256]
- Tiny_MLP: [64, 32]

**High-Capacity Models:**
- Ultra_Deep: [1024, 512, 256, 128, 64]
- Wide_Deep: [1024, 1024, 512, 512]

**Efficient Models:**
- Lightweight: [128, 64]
- Minimal: [32, 16]

**Research Models:**
- Experimental_A/B
- ResNet_Style
- Pyramid
- Hourglass
- DoubleWide
- VeryDeep
- Bottleneck
- Asymmetric
- PhD_Custom_1/2

**Custom Architecture:**
- Define your own layer sizes

### 3. Probe Types

Six probe generation methods:

1. **Continuous** - Random phases [0, 2π)
2. **Binary** - Phases {0, π}
3. **2-bit** - Phases {0, π/2, π, 3π/2}
4. **Hadamard** - Structured orthogonal patterns
5. **Sobol** - Low-discrepancy quasi-random
6. **Halton** - Another quasi-random sequence

### 4. Visualization Suite

25+ plot types available:

**Training Analysis:**
- training_curves - Loss, accuracy, eta, LR over epochs
- learning_curve - Train vs validation curves
- convergence_analysis - Compare multiple models

**Performance Metrics:**
- eta_distribution - Histogram of power ratios
- cdf - Cumulative distribution
- top_m_comparison - Bar chart of top-m accuracies
- baseline_comparison - ML vs baselines

**Multi-Model Comparison:**
- violin - Violin plots
- box - Box plots
- scatter - Scatter plots
- radar_chart - Multi-metric radar
- model_size_vs_performance - Parameters vs performance
- pareto_front - Complexity-performance tradeoff

**Probe Analysis:**
- heatmap - Phase configuration heatmap
- correlation_matrix - Probe similarity
- probe_type_comparison - Compare probe types
- phase_bits_comparison - Quantization study

**Parameter Sweeps:**
- eta_vs_M - Performance vs sensing budget
- eta_vs_K - Performance vs codebook size
- eta_vs_N - Performance vs RIS elements
- 3d_surface - 3D parameter surface

**Advanced Analysis:**
- confusion_matrix - Prediction analysis
- error_analysis - Error distribution
- power_distribution - Power statistics
- channel_statistics - Channel properties

### 5. Experiment Modes

**Single Experiment:**
- Standard single run with one configuration

**Multi-Model Comparison:**
- Compare multiple architectures
- Automatic aggregation of results
- Comparison plots

**Multi-Seed Runs:**
- Statistical validation
- Confidence intervals
- Aggregated statistics

### 6. Configuration Management

**Save Configurations:**
- JSON format for easy editing
- YAML format for readability
- Timestamped filenames

**Load Configurations:**
- Auto-detect format
- List available configs
- One-click restore

### 7. Results Export

**CSV Export:**
- Tabular format for Excel/plotting tools
- Model comparison tables

**JSON Export:**
- Complete results with metadata
- Nested structure for complex experiments

**Model Checkpoints:**
- Save trained model weights
- Resume training or inference

**LaTeX Tables:**
- Publication-ready tables
- Properly formatted metrics

## Usage Examples

### Example 1: Quick Test

```python
# Minimal configuration
widgets_dict['N'].value = 16
widgets_dict['K'].value = 32
widgets_dict['M'].value = 4
widgets_dict['n_train'].value = 1000
widgets_dict['n_epochs'].value = 5

# Run
on_run_experiment_clicked(None)
```

### Example 2: Architecture Comparison

```python
# Enable multi-model comparison
widgets_dict['compare_multiple_models'].value = True
widgets_dict['models_to_compare'].value = (
    'Baseline_MLP', 
    'Deep_MLP', 
    'Lightweight'
)

# Select comparison plots
widgets_dict['selected_plots'].value = (
    'violin',
    'box',
    'radar_chart',
    'model_size_vs_performance'
)

# Run
on_run_experiment_clicked(None)
```

### Example 3: Statistical Validation

```python
# Enable multi-seed runs
widgets_dict['multi_seed_runs'].value = True
widgets_dict['num_seeds'].value = 5
widgets_dict['compute_confidence_intervals'].value = True

# Run
on_run_experiment_clicked(None)
```

### Example 4: Probe Type Study

```python
# Programmatic multi-experiment
probe_types = ['continuous', 'binary', 'hadamard', 'sobol']
results = {}

for probe_type in probe_types:
    widgets_dict['probe_type'].value = probe_type
    on_run_experiment_clicked(None)
    results[probe_type] = current_results
    
# Generate comparison plot
from dashboard.plots import plot_probe_type_comparison
plot_probe_type_comparison(results, save_path='probe_comparison.png')
```

### Example 5: Save and Load Configuration

```python
# Save current configuration
on_save_config_clicked(None)

# Load a previous configuration
config = load_config('configs/config_20240109_120000.json')
dict_to_widgets(config, widgets_dict)
```

## Troubleshooting

### Common Issues

**1. Import Errors**

```
ModuleNotFoundError: No module named 'dashboard'
```

**Solution:** Ensure you're running from repository root:
```python
import os
os.chdir('/path/to/Session5_clone')
```

**2. CUDA Out of Memory**

```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce `batch_size` (try 64 or 32)
- Use smaller model architecture
- Reduce `n_train`
- Clear GPU cache: `torch.cuda.empty_cache()`

**3. Slow Training**

**Solutions:**
- Reduce `n_epochs`
- Use smaller `n_train` (e.g., 10000)
- Choose simpler model (e.g., Tiny_MLP)
- Check if GPU is being used

**4. Poor Performance**

**Solutions:**
- Increase model capacity
- More training data (`n_train`)
- Adjust learning rate (try 1e-4 or 1e-2)
- Try different probe types
- Increase M/K ratio

**5. Widget Not Responding**

**Solution:** Restart kernel and re-run setup cells:
- Kernel → Restart & Clear Output
- Run Cells 1-2 again

## API Reference

### Main Functions

#### `run_single_experiment(config_dict, progress_callback=None, verbose=True)`

Run a single experiment.

**Parameters:**
- `config_dict`: Dictionary with all configuration parameters
- `progress_callback`: Optional callback function(epoch, total_epochs, metrics)
- `verbose`: Whether to print progress messages

**Returns:**
- `ExperimentResults` object with evaluation, training_history, and model_state

#### `run_multi_model_comparison(base_config_dict, model_names, ...)`

Compare multiple model architectures.

**Parameters:**
- `base_config_dict`: Base configuration
- `model_names`: List of model names to compare

**Returns:**
- Dictionary mapping model names to ExperimentResults

#### `run_multi_seed_experiment(config_dict, seeds, ...)`

Run with multiple random seeds for statistical analysis.

**Parameters:**
- `config_dict`: Configuration dictionary
- `seeds`: List of random seeds

**Returns:**
- List of ExperimentResults, one per seed

#### `aggregate_results(results_list)`

Compute aggregate statistics from multiple runs.

**Parameters:**
- `results_list`: List of ExperimentResults

**Returns:**
- Dictionary with mean, std, min, max, confidence intervals

### Configuration Functions

#### `config_to_dict(widgets_dict)`

Extract configuration from widgets.

#### `dict_to_widgets(config_dict, widgets_dict)`

Set widget values from configuration.

#### `save_config(config_dict, filepath, format='auto')`

Save configuration to file (JSON or YAML).

#### `load_config(filepath, format='auto')`

Load configuration from file.

### Validation Functions

#### `get_validation_errors(config_dict)`

Validate complete configuration.

**Returns:**
- Tuple of (is_valid, error_messages)

## Best Practices

### 1. Start Small, Scale Up

Begin with:
- Small N, K (e.g., 16, 32)
- Few training samples (1000-5000)
- Few epochs (5-10)
- Simple model (Baseline_MLP)

Then gradually increase complexity.

### 2. Use Version Control

Save configurations with descriptive names:
```python
# configs/experiment_v1_baseline.yaml
# configs/experiment_v2_deep_model.yaml
```

### 3. Document Experiments

Add markdown cells in notebook describing:
- Hypothesis
- Configuration choices
- Expected results
- Observations

### 4. Regular Checkpoints

For long experiments, save intermediate results:
- Use lower `n_epochs` first
- Check convergence
- Then run full training

### 5. Statistical Validation

For publication results:
- Run with 5-10 seeds
- Compute confidence intervals
- Report mean ± std

## Advanced Topics

### Custom Probe Types

To add new probe generation methods, see `EXTENSION_GUIDE.md`.

### Custom Model Architectures

Beyond pre-defined models, you can:
1. Use "Custom" preset
2. Adjust `num_layers` and layer sizes
3. Or modify `model.py` for completely custom architectures (CNN, LSTM, etc.)

### Batch Processing

For large parameter sweeps:

```python
import itertools

# Define sweep ranges
N_values = [16, 32, 64]
M_values = [4, 8, 16]

# Run all combinations
for N, M in itertools.product(N_values, M_values):
    widgets_dict['N'].value = N
    widgets_dict['M'].value = M
    on_run_experiment_clicked(None)
    # Save results...
```

### Distributed Experiments

For cluster/cloud execution, extract core functions:

```python
from dashboard.experiment_runner import create_config_from_dict, run_single_experiment

# Can be run anywhere with dependencies
config_dict = {...}
results = run_single_experiment(config_dict, verbose=True)
```

## Support

For issues, questions, or contributions:
- Check `EXTENSION_GUIDE.md` for extending the system
- See `USAGE_EXAMPLES.md` for more examples
- Review existing issues/discussions in repository

## License

See `LICENSE` file in repository root.

---

**Version:** 1.0.0  
**Last Updated:** 2024-01-09
