# Extension Guide - RIS PhD Ultimate Dashboard

## Overview

This guide provides detailed instructions for extending the RIS PhD Ultimate Dashboard system with new features, probe types, models, plots, and more.

## Table of Contents

1. [Adding a New Probe Type](#1-adding-a-new-probe-type)
2. [Adding a New ML Model Architecture](#2-adding-a-new-ml-model-architecture)
3. [Adding a New Plot Type](#3-adding-a-new-plot-type)
4. [Adding New Physics Equations](#4-adding-new-physics-equations)
5. [Adding New Widgets](#5-adding-new-widgets)
6. [Adding New Metrics](#6-adding-new-metrics)
7. [Customizing the Workflow](#7-customizing-the-workflow)

---

## 1. Adding a New Probe Type

Probe types control how RIS phase configurations are generated. Follow these steps to add a custom probe generation method.

### Step 1: Create the Generator Function

Add your function to `experiments/probe_generators.py`:

```python
def generate_probe_bank_mynewtype(N: int, K: int, seed: Optional[int] = None) -> ProbeBank:
    """
    Generate probe bank with my new method.
    
    Args:
        N: Number of RIS elements
        K: Number of probes
        seed: Random seed for reproducibility
        
    Returns:
        ProbeBank with phases
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # Your custom phase generation logic here
    # Example: Fibonacci-based phases
    fibonacci = [1, 1]
    for i in range(2, max(N, K)):
        fibonacci.append(fibonacci[i-1] + fibonacci[i-2])
    
    phases = np.zeros((K, N))
    for k in range(K):
        for n in range(N):
            idx = (k * N + n) % len(fibonacci)
            phases[k, n] = (fibonacci[idx] % 360) * (2 * np.pi / 360)
    
    return ProbeBank(phases=phases, K=K, N=N, probe_type="mynewtype")
```

### Step 2: Register in Factory Function

Add to the `get_probe_bank()` function in `experiments/probe_generators.py`:

```python
def get_probe_bank(probe_type: str, N: int, K: int, seed: Optional[int] = None) -> ProbeBank:
    """Factory function to generate probe bank of specified type."""
    probe_type = probe_type.lower()
    
    if probe_type == "continuous":
        return generate_probe_bank_continuous(N, K, seed)
    elif probe_type == "binary":
        return generate_probe_bank_binary(N, K, seed)
    # ... existing types ...
    elif probe_type == "mynewtype":  # ADD THIS
        return generate_probe_bank_mynewtype(N, K, seed)
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")
```

### Step 3: Add to Widget Options

Update `dashboard/widgets.py`:

```python
widget_probe_type = widgets.Dropdown(
    options=['continuous', 'binary', '2bit', 'hadamard', 'sobol', 'halton', 'mynewtype'],  # ADD HERE
    value='continuous',
    description='Probe type:',
    # ... rest of configuration ...
)
```

### Step 4: Update Validator

Add to valid probe types in `dashboard/validators.py`:

```python
def validate_system_params(N: int, K: int, M: int, **kwargs) -> List[str]:
    # ... existing validation ...
    
    probe_type = kwargs.get('probe_type', 'continuous')
    valid_probe_types = ['continuous', 'binary', '2bit', 'hadamard', 'sobol', 'halton', 'mynewtype']  # ADD HERE
    if probe_type not in valid_probe_types:
        errors.append(f"probe_type must be one of {valid_probe_types} (got {probe_type})")
```

### Step 5: Update Documentation

Add description to parameter tables in documentation and notebook.

### Complete Example: Lattice-Based Probes

```python
# In experiments/probe_generators.py

def generate_probe_bank_lattice(N: int, K: int, seed: Optional[int] = None) -> ProbeBank:
    """
    Generate probe bank using lattice-based structured sampling.
    Distributes phases uniformly across a lattice in phase space.
    """
    import itertools
    
    # Create lattice points
    levels_per_dim = int(np.ceil(K ** (1/N)))
    phase_levels = np.linspace(0, 2*np.pi, levels_per_dim, endpoint=False)
    
    # Generate all lattice points
    lattice_points = list(itertools.product(phase_levels, repeat=N))
    
    # Select K points (use first K or sample if needed)
    if len(lattice_points) < K:
        # Repeat if needed
        lattice_points = lattice_points * (K // len(lattice_points) + 1)
    
    if seed is not None:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(lattice_points), K, replace=False)
    else:
        indices = np.arange(K)
    
    phases = np.array([lattice_points[i] for i in indices])
    
    return ProbeBank(phases=phases, K=K, N=N, probe_type="lattice")
```

---

## 2. Adding a New ML Model Architecture

You can add new architectures in two ways: simple (list of layer sizes) or advanced (custom PyTorch module).

### Method A: Simple Architecture (List of Sizes)

Just add to `model_registry.py`:

```python
MODEL_REGISTRY: Dict[str, List[int]] = {
    # ... existing models ...
    
    # Your new models
    "MyCustom_Model": [1024, 768, 512, 256],
    "ExtremelyWide": [4096, 2048, 1024],
    "ExtremelyDeep": [256, 256, 256, 256, 256, 256, 256, 256, 256, 256],
}
```

The dashboard will automatically pick it up!

### Method B: Advanced Custom Architecture

For completely custom architectures (e.g., CNN, LSTM, Transformer), you need to modify the core model class.

#### Step 1: Create New Model Class

In `model.py`:

```python
class LimitedProbingCNN(nn.Module):
    """
    CNN-based model for probe selection.
    
    Treats the masked input as a 1D signal and applies convolutions.
    """
    
    def __init__(self, K: int, num_filters: int = 64, kernel_size: int = 3):
        super(LimitedProbingCNN, self).__init__()
        
        self.K = K
        self.input_size = 2 * K
        
        # Treat [masked_powers, mask] as 1D signal with 2 channels
        self.conv1 = nn.Conv1d(2, num_filters, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size, padding=kernel_size//2)
        self.conv3 = nn.Conv1d(num_filters*2, num_filters, kernel_size, padding=kernel_size//2)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters, K)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, 2K] - [masked_powers, mask]
        Returns:
            logits: [batch_size, K]
        """
        batch_size = x.size(0)
        
        # Reshape to [batch_size, 2, K]
        x = x.view(batch_size, 2, self.K)
        
        # Apply convolutions
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # Global pooling
        x = self.pool(x).squeeze(-1)  # [batch_size, num_filters]
        
        # Output layer
        logits = self.fc(x)
        
        return logits
```

#### Step 2: Update Training Code

Modify `training.py` to support model selection:

```python
def create_model(model_type: str, config: Config) -> nn.Module:
    """Factory function to create models."""
    
    if model_type == "mlp":
        return LimitedProbingMLP(
            K=config.system.K,
            hidden_sizes=config.model.hidden_sizes,
            dropout_prob=config.model.dropout_prob,
            use_batch_norm=config.model.use_batch_norm
        )
    elif model_type == "cnn":
        return LimitedProbingCNN(
            K=config.system.K,
            num_filters=64,
            kernel_size=3
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

#### Step 3: Add Configuration Option

Update `config.py`:

```python
@dataclass
class ModelConfig:
    """Model architecture parameters."""
    model_type: str = "mlp"  # "mlp" or "cnn" or "lstm" etc.
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout_prob: float = 0.1
    use_batch_norm: bool = True
    # Add architecture-specific params
    cnn_num_filters: int = 64
    cnn_kernel_size: int = 3
```

---

## 3. Adding a New Plot Type

### Step 1: Create Plot Function

Add to `dashboard/plots.py`:

```python
def plot_my_new_visualization(results, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """
    My new custom plot for analyzing results.
    
    Args:
        results: EvaluationResults or dict of results
        save_path: Optional path to save figure
        dpi: Resolution for saved figure
        **kwargs: Additional plotting parameters
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Your plotting logic here
    if hasattr(results, 'eta_top1_distribution'):
        data = results.eta_top1_distribution
        ax.plot(data, 'o-')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('η')
        ax.set_title('My New Visualization')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig
```

### Step 2: Register in Plot Registry

Add to `dashboard/plots.py`:

```python
EXTENDED_PLOT_REGISTRY = {
    # ... existing plots ...
    'my_new_plot': plot_my_new_visualization,  # ADD THIS
}
```

### Step 3: Add to Widget Options

Update `dashboard/widgets.py`:

```python
ALL_PLOT_TYPES = [
    'training_curves', 'eta_distribution', # ... existing ...
    'my_new_plot',  # ADD THIS
]
```

### Complete Example: Timeline Plot

```python
def plot_eta_timeline(results_list, timestamps, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """
    Plot how performance evolves over time (e.g., over days of experiments).
    
    Args:
        results_list: List of ExperimentResults
        timestamps: List of datetime objects or strings
        save_path: Path to save figure
        dpi: Resolution
    """
    import matplotlib.dates as mdates
    from datetime import datetime
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract metrics
    eta_means = [np.mean(r.evaluation.eta_top1_distribution) for r in results_list]
    eta_stds = [np.std(r.evaluation.eta_top1_distribution) for r in results_list]
    
    # Convert timestamps if needed
    if isinstance(timestamps[0], str):
        timestamps = [datetime.strptime(t, '%Y-%m-%d') for t in timestamps]
    
    # Plot with error bars
    ax.errorbar(timestamps, eta_means, yerr=eta_stds, 
                marker='o', linewidth=2, capsize=5, label='η_top1')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=45)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('η (Power Ratio)')
    ax.set_title('Performance Timeline')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig
```

---

## 4. Adding New Physics Equations

To modify the physical model (channel generation, power calculation, etc.):

### Step 1: Modify Channel Generation

In `data_generation.py`:

```python
def generate_channel_realization_rician(N: int,
                                        sigma_h_sq: float = 1.0,
                                        sigma_g_sq: float = 1.0,
                                        K_factor: float = 3.0,  # Rician K-factor
                                        rng: Optional[np.random.RandomState] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate channel realization with Rician fading (instead of Rayleigh).
    
    Includes Line-of-Sight (LoS) component.
    """
    if rng is None:
        rng = np.random.RandomState()
    
    # LoS component (deterministic)
    h_los = np.sqrt(sigma_h_sq * K_factor / (K_factor + 1)) * np.ones(N, dtype=complex)
    g_los = np.sqrt(sigma_g_sq * K_factor / (K_factor + 1)) * np.ones(N, dtype=complex)
    
    # Scattered component (Rayleigh)
    h_scatter = np.sqrt(sigma_h_sq / (2 * (K_factor + 1))) * (rng.randn(N) + 1j * rng.randn(N))
    g_scatter = np.sqrt(sigma_g_sq / (2 * (K_factor + 1))) * (rng.randn(N) + 1j * rng.randn(N))
    
    # Total channel
    h = h_los + h_scatter
    g = g_los + g_scatter
    
    return h, g
```

### Step 2: Add Configuration

Update `config.py`:

```python
@dataclass
class SystemConfig:
    """Physical system parameters."""
    N: int = 32
    K: int = 64
    M: int = 8
    P_tx: float = 1.0
    sigma_h_sq: float = 1.0
    sigma_g_sq: float = 1.0
    phase_mode: str = "continuous"
    phase_bits: int = 3
    probe_type: str = "continuous"
    
    # New parameters
    channel_model: str = "rayleigh"  # "rayleigh" or "rician"
    rician_k_factor: float = 3.0  # For Rician fading
```

### Step 3: Update Generation Logic

Modify `generate_dataset()` in `data_generation.py` to use the new channel model based on configuration.

---

## 5. Adding New Widgets

### Step 1: Create Widget

In `dashboard/widgets.py`:

```python
# Add your new widget
widget_my_new_param = widgets.FloatSlider(
    value=5.0,
    min=1.0,
    max=10.0,
    step=0.5,
    description='My New Param:',
    style={'description_width': '180px'},
    layout=widgets.Layout(width='500px')
)
```

### Step 2: Add to Tab Layout

Add to appropriate tab in `create_tab_layout()`:

```python
def create_tab_layout():
    # ... existing code ...
    
    # Tab 1: System & Physics
    tab1_content = widgets.VBox([
        widgets.HTML("<h3>System & Physics Parameters</h3>"),
        widget_N,
        widget_K,
        # ... existing widgets ...
        widget_my_new_param,  # ADD HERE
    ], layout=widgets.Layout(padding='20px'))
    
    # ... rest of function ...
```

### Step 3: Add to Widgets Dictionary

Update `get_all_widgets()`:

```python
def get_all_widgets() -> Dict:
    """Return dictionary of all widgets for easy access."""
    return {
        # ... existing widgets ...
        'my_new_param': widget_my_new_param,  # ADD HERE
    }
```

### Step 4: Add Callback (if needed)

In `dashboard/callbacks.py`:

```python
def on_my_new_param_change(change, widgets_dict):
    """Handle changes to my new parameter."""
    new_value = change['new']
    
    # Your logic here
    # E.g., update related widgets, validate, etc.
    if new_value > 8.0:
        # Show warning
        with widgets_dict['status_output']:
            print(f"⚠️ Warning: my_new_param is high ({new_value})")


def setup_all_callbacks(widgets_dict):
    """Attach all callbacks to widgets."""
    # ... existing callbacks ...
    
    # Add new callback
    widgets_dict['my_new_param'].observe(
        lambda change: on_my_new_param_change(change, widgets_dict),
        names='value'
    )
```

### Step 5: Add to Config Manager

Update `dashboard/config_manager.py`:

```python
def config_to_dict(widgets_dict: Dict) -> Dict[str, Any]:
    """Extract configuration values from widgets dictionary."""
    config = {
        # ... existing params ...
        'my_new_param': widgets_dict['my_new_param'].value,  # ADD HERE
    }
    return config


def dict_to_widgets(config_dict: Dict[str, Any], widgets_dict: Dict):
    """Set widget values from configuration dictionary."""
    # ... existing code ...
    
    if 'my_new_param' in config_dict:
        widgets_dict['my_new_param'].value = config_dict['my_new_param']
```

### Step 6: Add Validation

In `dashboard/validators.py`:

```python
def validate_system_params(N: int, K: int, M: int, **kwargs) -> List[str]:
    """Validate system parameters."""
    errors = []
    
    # ... existing validation ...
    
    # Validate new parameter
    my_new_param = kwargs.get('my_new_param', 5.0)
    if not (1.0 <= my_new_param <= 10.0):
        errors.append(f"my_new_param must be between 1.0 and 10.0 (got {my_new_param})")
    
    return errors
```

---

## 6. Adding New Metrics

To add new evaluation metrics beyond eta and accuracy:

### Step 1: Update EvaluationResults

In `evaluation.py`:

```python
@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    # ... existing fields ...
    
    # Add new metrics
    spectral_efficiency: float = 0.0
    energy_efficiency: float = 0.0
    fairness_index: float = 0.0
    
    # Add distributions
    spectral_efficiency_distribution: np.ndarray = None
    
    def to_dict(self) -> Dict:
        result = {
            # ... existing fields ...
            'spectral_efficiency': self.spectral_efficiency,
            'energy_efficiency': self.energy_efficiency,
            'fairness_index': self.fairness_index,
        }
        return result
```

### Step 2: Compute in Evaluation

Update `evaluate_model()` in `evaluation.py`:

```python
def evaluate_model(model: nn.Module, test_loader: DataLoader, 
                  probe_bank: ProbeBank, config: Config,
                  verbose: bool = True) -> EvaluationResults:
    """Evaluate model and compute all metrics."""
    
    # ... existing evaluation code ...
    
    # Compute new metrics
    spectral_efficiency = compute_spectral_efficiency(powers_full, eta_top1_list, config)
    energy_efficiency = compute_energy_efficiency(powers_full, config)
    fairness_index = compute_fairness_index(eta_top1_list)
    
    return EvaluationResults(
        # ... existing fields ...
        spectral_efficiency=spectral_efficiency,
        energy_efficiency=energy_efficiency,
        fairness_index=fairness_index,
        # ... rest of fields ...
    )


def compute_spectral_efficiency(powers, eta_list, config):
    """Compute spectral efficiency metric."""
    # Your computation here
    # Example: SE = log2(1 + SNR)
    snr = np.array(eta_list) * config.system.P_tx
    se = np.mean(np.log2(1 + snr))
    return se


def compute_energy_efficiency(powers, config):
    """Compute energy efficiency (bits per joule)."""
    # Your computation
    total_power = config.system.P_tx + 0.1  # Add circuit power
    se = np.mean(np.log2(1 + powers.mean() * config.system.P_tx))
    ee = se / total_power
    return ee


def compute_fairness_index(eta_list):
    """Compute Jain's fairness index."""
    eta_array = np.array(eta_list)
    numerator = np.sum(eta_array) ** 2
    denominator = len(eta_array) * np.sum(eta_array ** 2)
    return numerator / denominator if denominator > 0 else 0.0
```

### Step 3: Add Visualizations

Create plots for new metrics in `dashboard/plots.py`:

```python
def plot_spectral_efficiency(results, save_path: Optional[str] = None, dpi: int = 150, **kwargs):
    """Plot spectral efficiency analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Your plotting logic
    se = results.spectral_efficiency
    ax.bar(['Spectral Efficiency'], [se])
    ax.set_ylabel('SE (bits/s/Hz)')
    ax.set_title('Spectral Efficiency')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig
```

---

## 7. Customizing the Workflow

### Custom Pre-processing

Add custom data transformations:

```python
# In data_generation.py

def apply_custom_preprocessing(data: torch.Tensor, method: str = 'standardize') -> torch.Tensor:
    """Apply custom preprocessing to data."""
    
    if method == 'standardize':
        mean = data.mean(dim=0, keepdim=True)
        std = data.std(dim=0, keepdim=True)
        return (data - mean) / (std + 1e-8)
    
    elif method == 'robust':
        median = data.median(dim=0, keepdim=True)[0]
        mad = torch.median(torch.abs(data - median), dim=0, keepdim=True)[0]
        return (data - median) / (mad + 1e-8)
    
    elif method == 'minmax':
        min_val = data.min(dim=0, keepdim=True)[0]
        max_val = data.max(dim=0, keepdim=True)[0]
        return (data - min_val) / (max_val - min_val + 1e-8)
    
    else:
        return data
```

### Custom Loss Functions

Add custom training objectives:

```python
# In training.py

def create_loss_function(loss_type: str):
    """Factory for loss functions."""
    
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    
    elif loss_type == "focal":
        # Focal loss for imbalanced data
        class FocalLoss(nn.Module):
            def __init__(self, gamma=2.0):
                super().__init__()
                self.gamma = gamma
                self.ce = nn.CrossEntropyLoss(reduction='none')
            
            def forward(self, input, target):
                ce_loss = self.ce(input, target)
                pt = torch.exp(-ce_loss)
                focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
                return focal_loss
        
        return FocalLoss()
    
    elif loss_type == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
```

### Custom Optimizers

```python
# In training.py

def create_optimizer(model, config, optimizer_type: str = "adam"):
    """Factory for optimizers."""
    
    if optimizer_type == "adam":
        return optim.Adam(model.parameters(), 
                         lr=config.training.learning_rate,
                         weight_decay=config.training.weight_decay)
    
    elif optimizer_type == "adamw":
        return optim.AdamW(model.parameters(),
                          lr=config.training.learning_rate,
                          weight_decay=config.training.weight_decay)
    
    elif optimizer_type == "sgd_momentum":
        return optim.SGD(model.parameters(),
                        lr=config.training.learning_rate,
                        momentum=0.9,
                        weight_decay=config.training.weight_decay)
    
    elif optimizer_type == "lamb":
        # LAMB optimizer (requires separate package)
        try:
            from pytorch_lamb import Lamb
            return Lamb(model.parameters(),
                       lr=config.training.learning_rate,
                       weight_decay=config.training.weight_decay)
        except ImportError:
            print("⚠️ LAMB optimizer requires pytorch-lamb package")
            print("   Falling back to AdamW")
            return optim.AdamW(model.parameters(),
                              lr=config.training.learning_rate,
                              weight_decay=config.training.weight_decay)
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
```

---

## Best Practices for Extensions

### 1. Maintain Backward Compatibility

When adding new features:
- Add new parameters with default values
- Don't remove or rename existing parameters
- Support both old and new APIs

```python
# Good: Optional new parameter with default
def my_function(required_param, new_optional_param=None):
    if new_optional_param is None:
        new_optional_param = default_value
    # ... rest of function

# Bad: Renamed parameter breaks existing code
# def my_function(required_param):  # Old
def my_function(different_name):  # New - breaks compatibility!
```

### 2. Document Everything

Add docstrings to all functions:

```python
def my_new_function(param1: int, param2: float) -> Dict:
    """
    Brief description of what this function does.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Dictionary with results
        
    Raises:
        ValueError: If param1 is negative
        
    Example:
        >>> result = my_new_function(10, 0.5)
        >>> print(result['metric'])
        15.5
    """
    # Implementation
```

### 3. Add Error Handling

Gracefully handle errors:

```python
def robust_function(data):
    """Function with comprehensive error handling."""
    
    try:
        # Main logic
        result = process_data(data)
        return result
        
    except ValueError as e:
        print(f"⚠️ Value error: {str(e)}")
        return default_value
        
    except RuntimeError as e:
        print(f"⚠️ Runtime error: {str(e)}")
        return None
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
```

### 4. Write Tests

Add tests for new functionality:

```python
def test_my_new_function():
    """Test my new function."""
    # Test normal case
    result = my_new_function(10, 0.5)
    assert result['metric'] > 0
    
    # Test edge cases
    result = my_new_function(0, 0.0)
    assert result is not None
    
    # Test error handling
    with pytest.raises(ValueError):
        my_new_function(-1, 0.5)
    
    print("✅ All tests passed")
```

### 5. Keep Code Organized

Follow the existing structure:
- Configuration in `config.py`
- Core functionality in main modules
- Extensions in `experiments/` or `dashboard/`
- Tests in `tests/` (if exists)
- Documentation in `docs/` or markdown files

---

## Example: Complete Extension

Here's a complete example adding a "diversity score" metric:

```python
# 1. Add to evaluation.py
@dataclass
class EvaluationResults:
    # ... existing fields ...
    diversity_score: float = 0.0


def evaluate_model(model, test_loader, probe_bank, config, verbose=True):
    # ... existing code ...
    
    # Compute diversity score
    diversity_score = compute_probe_diversity(probe_bank)
    
    return EvaluationResults(
        # ... existing fields ...
        diversity_score=diversity_score,
    )


def compute_probe_diversity(probe_bank):
    """Compute diversity score of selected probes."""
    from experiments.diversity_analysis import compute_cosine_similarity_matrix
    
    similarity = compute_cosine_similarity_matrix(probe_bank)
    # Diversity = 1 - average similarity
    diversity = 1.0 - np.mean(similarity)
    return diversity


# 2. Add visualization in dashboard/plots.py
def plot_diversity_score(results, save_path=None, dpi=150, **kwargs):
    """Plot diversity score."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    score = results.diversity_score
    ax.bar(['Diversity Score'], [score], color='steelblue')
    ax.set_ylim([0, 1])
    ax.set_ylabel('Diversity')
    ax.set_title('Probe Bank Diversity Score')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


# 3. Register plot
EXTENDED_PLOT_REGISTRY['diversity_score'] = plot_diversity_score


# 4. Add to widget options in dashboard/widgets.py
ALL_PLOT_TYPES = [
    # ... existing ...
    'diversity_score',
]
```

Done! Now "diversity_score" is fully integrated into the dashboard.

---

## Support

For questions about extending the system:
1. Check existing code for similar functionality
2. Review this guide
3. Consult `dashboard/README.md` for usage
4. Look at existing extensions in `experiments/`

## Contributing

When contributing extensions:
1. Follow existing code style
2. Add documentation
3. Add examples
4. Test thoroughly
5. Update relevant guides

---

**Version:** 1.0.0  
**Last Updated:** 2024-01-09
