"""
System Configuration
====================
Core configuration classes for the RIS research platform.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import torch
import logging

logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    """System-level configuration."""
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    log_level: str = 'INFO'

    def __post_init__(self):
        """Validate configuration."""
        if self.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU")
            self.device = 'cpu'


@dataclass
class DataConfig:
    """Data generation and loading configuration."""
    # Data generation
    num_samples: int = 10000
    M_values: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    N: int = 64
    K: int = 4

    # Data loading
    batch_size: int = 128
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    # Dataset options
    shuffle: bool = True
    normalize: bool = True

    def __post_init__(self):
        """Validate configuration."""
        assert self.train_split + self.val_split + self.test_split == 1.0, \
            "Data splits must sum to 1.0"
        assert all(m > 0 for m in self.M_values), "M values must be positive"
        assert self.N > 0 and self.K > 0, "N and K must be positive"


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_name: str = 'Baseline_MLP'
    hidden_layers: List[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = 'relu'
    dropout: float = 0.1
    use_batch_norm: bool = True

    # Attention-specific
    num_heads: int = 4

    # Residual-specific
    use_residual: bool = False

    def __post_init__(self):
        """Validate configuration."""
        assert self.dropout >= 0 and self.dropout < 1, "Dropout must be in [0, 1)"
        assert len(self.hidden_layers) > 0, "Must have at least one hidden layer"
        assert self.activation in ['relu', 'gelu', 'tanh', 'leaky_relu'], \
            f"Unknown activation: {self.activation}"


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    # Optimization
    optimizer: str = 'adam'
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100

    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_type: str = 'reduce_on_plateau'
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4

    # Loss function
    loss_function: str = 'mse'

    # Logging
    log_interval: int = 10
    save_checkpoint: bool = True
    checkpoint_interval: int = 10

    def __post_init__(self):
        """Validate configuration."""
        assert self.optimizer in ['adam', 'sgd', 'adamw'], \
            f"Unknown optimizer: {self.optimizer}"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.max_epochs > 0, "Max epochs must be positive"


@dataclass
class EvaluationConfig:
    """Evaluation and metrics configuration."""
    # Metrics to compute
    compute_mse: bool = True
    compute_mae: bool = True
    compute_r2: bool = True
    compute_capacity_gap: bool = True

    # Evaluation options
    eval_batch_size: int = 256
    save_predictions: bool = False

    # Visualization
    plot_predictions: bool = True
    plot_training_curves: bool = True
    plot_capacity_curves: bool = True


@dataclass
class PhysicsConfig:
    """Physics simulation configuration."""
    # Channel model
    channel_model: str = 'rayleigh'
    snr_db: float = 20.0

    # RIS parameters
    phase_quantization_bits: int = 0  # 0 = continuous
    amplitude_control: bool = False

    # Beamforming
    beamforming_method: str = 'matched_filter'

    # Optimization (for future use)
    optimization_method: str = 'gradient_descent'
    max_iterations: int = 100
    convergence_threshold: float = 1e-6

    def __post_init__(self):
        """Validate configuration."""
        assert self.channel_model in ['rayleigh', 'rician', 'los'], \
            f"Unknown channel model: {self.channel_model}"
        assert self.phase_quantization_bits >= 0, \
            "Phase quantization bits must be non-negative"


@dataclass
class ExperimentConfig:
    """High-level experiment configuration."""
    experiment_name: str = 'default_experiment'
    experiment_type: str = 'single_model'  # or 'compare_models', 'sweep_params'

    # Results
    results_dir: str = 'results'
    save_results: bool = True

    # Reproducibility
    deterministic: bool = True
    benchmark: bool = False

    def __post_init__(self):
        """Validate configuration."""
        assert self.experiment_type in ['single_model', 'compare_models', 'sweep_params'], \
            f"Unknown experiment type: {self.experiment_type}"


@dataclass
class Config:
    """
    Complete system configuration.

    This is the main configuration object that combines all sub-configs.
    """
    system: SystemConfig = field(default_factory=SystemConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def __post_init__(self):
        """Set up logging and validate all sub-configs."""
        logging.basicConfig(
            level=getattr(logging, self.system.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Set random seeds
        if self.experiment.deterministic:
            torch.manual_seed(self.system.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.system.seed)
                if self.experiment.benchmark:
                    torch.backends.cudnn.benchmark = True
                else:
                    torch.backends.cudnn.deterministic = True

    def summary(self) -> str:
        """Get a summary of the configuration."""
        lines = [
            "=" * 70,
            "CONFIGURATION SUMMARY",
            "=" * 70,
            "",
            "System:",
            f"  Device: {self.system.device}",
            f"  Seed: {self.system.seed}",
            "",
            "Data:",
            f"  Samples: {self.data.num_samples}",
            f"  M values: {self.data.M_values}",
            f"  N={self.data.N}, K={self.data.K}",
            f"  Batch size: {self.data.batch_size}",
            "",
            "Model:",
            f"  Architecture: {self.model.model_name}",
            f"  Hidden layers: {self.model.hidden_layers}",
            f"  Activation: {self.model.activation}",
            "",
            "Training:",
            f"  Optimizer: {self.training.optimizer}",
            f"  Learning rate: {self.training.learning_rate}",
            f"  Max epochs: {self.training.max_epochs}",
            f"  Early stopping: {self.training.early_stopping}",
            "",
            "Experiment:",
            f"  Name: {self.experiment.experiment_name}",
            f"  Type: {self.experiment.experiment_type}",
            "=" * 70
        ]
        return "\n".join(lines)


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def get_quick_test_config() -> Config:
    """Get configuration for quick testing."""
    config = Config()
    config.data.num_samples = 1000
    config.data.M_values = [4, 8]
    config.training.max_epochs = 10
    config.experiment.experiment_name = 'quick_test'
    return config


def get_full_experiment_config() -> Config:
    """Get configuration for full experiment."""
    config = Config()
    config.data.num_samples = 50000
    config.data.M_values = [4, 8, 16, 32, 64]
    config.training.max_epochs = 200
    config.experiment.experiment_name = 'full_experiment'
    return config


def get_phd_config() -> Config:
    """Get configuration for PhD-level experiments."""
    config = Config()
    config.data.num_samples = 100000
    config.data.M_values = [4, 8, 16, 32, 64, 128]
    config.model.hidden_layers = [1024, 512, 256, 128]
    config.training.max_epochs = 500
    config.training.learning_rate = 5e-4
    config.experiment.experiment_name = 'phd_experiment'
    config.experiment.deterministic = True
    return config


# Add these lines at the END of your config/system_config.py file
# (after the preset configuration functions)

# =============================================================================
# BACKWARDS COMPATIBILITY ALIASES
# =============================================================================

# Alias for backwards compatibility
EvalConfig = EvaluationConfig

# =============================================================================
# GLOBAL CONFIG MANAGEMENT
# =============================================================================

_global_config: Config = None


def get_config() -> Config:
    """
    Get the global configuration instance.

    Returns:
        Global Config instance

    Raises:
        RuntimeError: If config has not been set
    """
    global _global_config

    if _global_config is None:
        # Create default config if none exists
        _global_config = Config()

    return _global_config


def set_config(config: Config) -> None:
    """
    Set the global configuration instance.

    Args:
        config: Config instance to set as global
    """
    global _global_config
    _global_config = config
    logger.info(f"Global config set: {config.experiment.experiment_name}")


def reset_config() -> None:
    """Reset global config to default."""
    global _global_config
    _global_config = Config()
    logger.info("Global config reset to default")