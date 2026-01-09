# Usage Examples - New Features

This document provides practical examples of using the new features added in the code cleaning and standardization update.

## Table of Contents
1. [Model Registry](#model-registry)
2. [Plot Registry](#plot-registry)
3. [Experiment Toolkit](#experiment-toolkit)
4. [Backward Compatibility](#backward-compatibility)
5. [Migration Examples](#migration-examples)

---

## Model Registry

The Model Registry provides centralized management of model architectures.

### List Available Models

```python
from model_registry import list_models

# Get all available models
models = list_models()
print(f"Available models: {models}")
# Output: ['Baseline_MLP', 'Deep_MLP', 'Tiny_MLP', 'Ultra_Deep', 'Wide_Deep', 'Lightweight', 'Minimal', 'Experimental_A', 'Experimental_B']
```

### Get Model Architecture

```python
from model_registry import get_model_architecture

# Get a specific model's architecture
arch = get_model_architecture('Baseline_MLP')
print(f"Baseline_MLP: {arch}")
# Output: [256, 128]

arch = get_model_architecture('Deep_MLP')
print(f"Deep_MLP: {arch}")
# Output: [512, 512, 256]
```

### Register Custom Models

```python
from model_registry import register_model, list_models

# Register your custom model
register_model('My_Research_Model', [768, 512, 256, 128])

# Verify it's registered
if 'My_Research_Model' in list_models():
    print("✓ Custom model registered successfully")

# Use it in experiments
from model_registry import get_model_architecture
my_arch = get_model_architecture('My_Research_Model')
print(f"My model: {my_arch}")
```

### Use in Configuration

```python
from config import get_config
from model_registry import get_model_architecture

# Old way - manual specification
config = get_config(
    model={'hidden_sizes': [512, 256, 128]}
)

# New way - using registry
config = get_config(
    model={'hidden_sizes': get_model_architecture('Deep_MLP')}
)
```

---

## Plot Registry

The Plot Registry provides unified access to all visualization functions.

### List Available Plots

```python
from plot_registry import list_plots

plots = list_plots()
print(f"Available plots: {plots}")
# Output: ['training_curves', 'eta_distribution', 'cdf', 'violin', 'heatmap', ...]
```

### Use Plot Functions

```python
from plot_registry import get_plot_function

# Get and use a plot function
plot_cdf = get_plot_function('cdf')
plot_cdf(results, save_path='output/cdf.png')

# Use violin plot for model comparison
plot_violin = get_plot_function('violin')
plot_violin(results_dict, save_path='output/model_comparison.png')

# Create probe heatmap
plot_heatmap = get_plot_function('heatmap')
plot_heatmap(probe_bank, save_path='output/probe_heatmap.png')
```

### Register Custom Plots

```python
from plot_registry import register_plot
import matplotlib.pyplot as plt
import numpy as np

def my_custom_plot(results, save_path=None):
    """My custom visualization."""
    plt.figure(figsize=(10, 6))
    plt.hist(results.eta_top1_distribution, bins=50)
    plt.xlabel('η')
    plt.ylabel('Frequency')
    plt.title('My Custom Plot')
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Register it
register_plot('my_custom', my_custom_plot)

# Use it
from plot_registry import get_plot_function
plot_func = get_plot_function('my_custom')
plot_func(results, save_path='output/custom.png')
```

---

## Experiment Toolkit

The Experiment Toolkit provides a high-level fluent API for running experiments.

### ConfigBuilder - Fluent Configuration

```python
from experiment_toolkit import ConfigBuilder

# Build configuration with fluent interface
config = (ConfigBuilder()
    .system(N=32, K=64, M=8, probe_type='continuous')
    .data(n_train=50000, n_val=5000, n_test=5000, seed=42)
    .model([512, 256, 128])  # Direct specification
    .training(epochs=50, batch_size=128, lr=1e-3)
    .build())

# Or use model registry
config = (ConfigBuilder()
    .system(N=32, K=64, M=8, probe_type='hadamard')
    .data(n_train=50000, seed=42)
    .model_by_name('Deep_MLP')  # Use registry
    .training(epochs=50)
    .build())
```

### ExperimentRunner - High-Level Execution

```python
from experiment_toolkit import ConfigBuilder, ExperimentRunner

# Configure
config = (ConfigBuilder()
    .system(N=32, K=64, M=8, probe_type='continuous')
    .data(n_train=50000, seed=42)
    .model_by_name('Baseline_MLP')
    .training(epochs=50)
    .build())

# Run experiment
results = ExperimentRunner.run(config, verbose=True)

# Access results
print(f"Model: {results['model']}")
print(f"Training history: {results['history']}")
print(f"Evaluation results: {results['results']}")
print(f"Test eta_top1: {results['results'].eta_top1:.4f}")
```

### Compare Multiple Models

```python
from experiment_toolkit import ConfigBuilder, ExperimentRunner

# Base configuration
base_config = (ConfigBuilder()
    .system(N=32, K=64, M=8, probe_type='continuous')
    .data(n_train=50000, seed=42)
    .training(epochs=50)
    .build())

# Compare multiple models
results = ExperimentRunner.compare_models(
    config_base=base_config,
    model_names=['Baseline_MLP', 'Deep_MLP', 'Tiny_MLP'],
    verbose=True
)

# Analyze results
for model_name, result in results.items():
    print(f"{model_name}: eta_top1 = {result['results'].eta_top1:.4f}")
```

### PlotGallery - Unified Plotting

```python
from experiment_toolkit import PlotGallery

# Show single plot
PlotGallery.show('cdf', results['results'])

# Show multiple plots
PlotGallery.show(
    ['eta_distribution', 'training_curves'], 
    results['results'], 
    results['history']
)

# List available plots
available = PlotGallery.list_available()
print(f"Available: {available}")
```

### Complete Example

```python
from experiment_toolkit import ConfigBuilder, ExperimentRunner, PlotGallery

# Configure experiment
config = (ConfigBuilder()
    .system(N=32, K=64, M=8, probe_type='hadamard')
    .data(n_train=50000, n_val=5000, n_test=5000, seed=42)
    .model_by_name('Deep_MLP')
    .training(epochs=50, batch_size=128, lr=1e-3, patience=10)
    .build())

# Run experiment
experiment = ExperimentRunner.run(config, verbose=True)

# Print summary
print("\nResults Summary:")
print(f"  Top-1 accuracy: {experiment['results'].accuracy_top1:.4f}")
print(f"  Top-1 eta: {experiment['results'].eta_top1:.4f}")
print(f"  vs Random: {experiment['results'].eta_random_1:.4f}")
print(f"  vs Best Observed: {experiment['results'].eta_best_observed:.4f}")

# Visualize
PlotGallery.show(['cdf', 'training_curves', 'heatmap'], 
                 experiment['results'], 
                 experiment['history'],
                 experiment['probe_bank'])
```

---

## Backward Compatibility

All old code continues to work with deprecation warnings.

### Using Deprecated Parameters

```python
from config import get_config
import warnings

# Old parameter (shows warning but works)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    config = get_config(
        system={'probe_bank_method': 'hadamard'}  # Deprecated
    )
    if len(w) > 0:
        print(f"Warning: {w[0].message}")
    # Automatically mapped to probe_type='hadamard'
    print(f"probe_type: {config.system.probe_type}")

# New parameter (recommended)
config = get_config(
    system={'probe_type': 'hadamard'}  # Use this
)
```

---

## Migration Examples

### Example 1: Simple Experiment

**Old Code:**
```python
from config import get_config
from data_generation import generate_probe_bank, create_dataloaders
from model import create_model
from training import train

config = get_config(
    system={'N': 32, 'K': 64, 'M': 8},
    training={'num_epochs': 50}  # Old parameter
)

probe_bank = generate_probe_bank(
    config.system.N, config.system.K, 
    config.data.seed
)

train_loader, val_loader, test_loader, metadata = create_dataloaders(config, probe_bank)
model = create_model(config)
trained_model, history = train(model, train_loader, val_loader, config, metadata)
```

**New Code:**
```python
from experiment_toolkit import ConfigBuilder, ExperimentRunner

config = (ConfigBuilder()
    .system(N=32, K=64, M=8, probe_type='continuous')
    .training(epochs=50)  # New parameter: n_epochs
    .build())

results = ExperimentRunner.run(config)
```

### Example 2: Model Comparison

**Old Code:**
```python
from config import get_config
from data_generation import generate_probe_bank, create_dataloaders
from model import create_model
from training import train

model_architectures = {
    'baseline': [256, 128],
    'deep': [512, 512, 256],
    'tiny': [64, 32]
}

results = {}
for name, arch in model_architectures.items():
    config = get_config(
        system={'N': 32, 'K': 64, 'M': 8},
        model={'hidden_sizes': arch},
        training={'num_epochs': 50}
    )
    
    probe_bank = generate_probe_bank(config.system.N, config.system.K, config.data.seed)
    train_loader, val_loader, test_loader, metadata = create_dataloaders(config, probe_bank)
    model = create_model(config)
    trained_model, history = train(model, train_loader, val_loader, config, metadata)
    
    results[name] = {'model': trained_model, 'history': history}
```

**New Code:**
```python
from experiment_toolkit import ConfigBuilder, ExperimentRunner
from model_registry import register_model

# Register models (or use pre-defined ones)
register_model('baseline', [256, 128])
register_model('deep', [512, 512, 256])
register_model('tiny', [64, 32])

base_config = (ConfigBuilder()
    .system(N=32, K=64, M=8)
    .training(epochs=50)
    .build())

results = ExperimentRunner.compare_models(
    config_base=base_config,
    model_names=['baseline', 'deep', 'tiny']
)
```

### Example 3: Custom Probe Types

**Old Code:**
```python
from config import get_config, SystemConfig
from data_generation import generate_probe_bank

config = get_config(
    system={
        'N': 32, 
        'K': 64, 
        'M': 8,
        'probe_bank_method': 'hadamard'  # Deprecated
    }
)
```

**New Code:**
```python
from config import get_config

config = get_config(
    system={
        'N': 32, 
        'K': 64, 
        'M': 8,
        'probe_type': 'hadamard'  # New parameter
    }
)

# Or with ConfigBuilder
from experiment_toolkit import ConfigBuilder

config = (ConfigBuilder()
    .system(N=32, K=64, M=8, probe_type='hadamard')
    .build())
```

---

## Tips and Best Practices

### 1. Use Model Registry for Reproducibility
```python
# Instead of hardcoding architectures
config = get_config(model={'hidden_sizes': [512, 256, 128]})

# Use registry for consistency
from model_registry import get_model_architecture
config = get_config(model={'hidden_sizes': get_model_architecture('Deep_MLP')})
```

### 2. Use ConfigBuilder for Readability
```python
# Cleaner and more readable
config = (ConfigBuilder()
    .system(N=32, K=64, M=8, probe_type='continuous')
    .data(n_train=50000, seed=42)
    .model_by_name('Baseline_MLP')
    .training(epochs=50, batch_size=128)
    .build())

# vs nested dictionaries
config = get_config(
    system={'N': 32, 'K': 64, 'M': 8, 'probe_type': 'continuous'},
    data={'n_train': 50000, 'seed': 42},
    model={'hidden_sizes': [256, 128]},
    training={'n_epochs': 50, 'batch_size': 128}
)
```

### 3. Use PlotGallery for Consistent Visualizations
```python
from experiment_toolkit import PlotGallery

# Consistent plotting across experiments
PlotGallery.show(['cdf', 'eta_distribution', 'training_curves'], 
                 results, history)
```

### 4. Register Domain-Specific Models
```python
from model_registry import register_model

# Register models for your specific use case
register_model('mmWave_Optimized', [1024, 512, 256])
register_model('Power_Efficient', [128, 64, 32])
register_model('High_Accuracy', [2048, 1024, 512, 256])

# Use them in experiments
from experiment_toolkit import ConfigBuilder
config = ConfigBuilder().model_by_name('mmWave_Optimized').build()
```

---

## Next Steps

1. Explore the [CHANGELOG.md](CHANGELOG.md) for detailed information about all changes
2. Check [README.md](README.md) for updated documentation
3. Run existing experiments to verify backward compatibility
4. Migrate to new API gradually using the examples above
5. Register your custom models and plots for better organization

For questions or issues, please open an issue on GitHub.
