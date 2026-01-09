"""
Configuration and hyperparameters for RIS probe-based control with limited probing.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class SystemConfig:
    """Physical system parameters."""
    N: int = 32                    # Number of RIS elements
    K: int = 64                    # Number of probes in the bank
    M: int = 8                     # Sensing budget (probes measured per sample)
    P_tx: float = 1.0              # Transmit power (normalized)
    sigma_h_sq: float = 1.0        # BS-RIS channel variance
    sigma_g_sq: float = 1.0        # RIS-UE channel variance
    phase_mode: str = "continuous"  # "continuous" or "discrete"
    phase_bits: int = 3            # Bits for discrete phase quantization
    probe_type: str = "continuous"  # Type of probe: "continuous", "binary", "2bit", "hadamard", "sobol", "halton"
    probe_bank_method: Optional[str] = None  # DEPRECATED: Use probe_type instead
    
    def __post_init__(self):
        """Handle backward compatibility for probe_bank_method."""
        # Handle deprecated probe_bank_method
        if self.probe_bank_method is not None:
            import warnings
            warnings.warn(
                "probe_bank_method is deprecated and will be removed in a future version. "
                "Use probe_type instead.",
                DeprecationWarning,
                stacklevel=2
            )
            # Map old values to new ones
            if self.probe_bank_method == "random":
                self.probe_type = "continuous"
            else:
                self.probe_type = self.probe_bank_method


@dataclass
class DataConfig:
    """Data generation parameters."""
    n_train: int = 50000           # Number of training samples
    n_val: int = 5000              # Number of validation samples
    n_test: int = 5000             # Number of test samples
    seed: int = 42                 # Random seed for reproducibility
    normalize_input: bool = True   # Whether to normalize observed powers
    normalization_type: str = "mean"  # "mean", "std", or "log"


@dataclass
class ModelConfig:
    """MLP architecture parameters."""
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout_prob: float = 0.1
    use_batch_norm: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 50
    early_stop_patience: int = 10
    eval_interval: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EvalConfig:
    """Evaluation parameters."""
    top_m_values: List[int] = field(default_factory=lambda: [1, 2, 4, 8])


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    system: SystemConfig = field(default_factory=SystemConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self):
        # Run SystemConfig's __post_init__ for backward compatibility
        if hasattr(self.system, '__post_init__'):
            self.system.__post_init__()
        
        # Ensure top_m values don't exceed K
        self.eval.top_m_values = [m for m in self.eval.top_m_values if m <= self.system.K]
        # Ensure M <= K
        if self.system.M > self.system.K:
            raise ValueError(f"M ({self.system.M}) cannot exceed K ({self.system.K})")
        # Validate phase configuration
        if self.system.phase_mode not in {"continuous", "discrete"}:
            raise ValueError("phase_mode must be 'continuous' or 'discrete'")
        if self.system.phase_mode == "discrete" and self.system.phase_bits <= 0:
            raise ValueError("phase_bits must be > 0 when phase_mode is 'discrete'")
        # Validate probe_type
        valid_probe_types = {"continuous", "binary", "2bit", "hadamard", "sobol", "halton"}
        if self.system.probe_type not in valid_probe_types:
            raise ValueError(f"probe_type must be one of: {', '.join(valid_probe_types)}")

    def print_config(self):
        """Print configuration summary."""
        print("\n" + "=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        print("\nSystem:")
        print(f"  N (RIS elements):     {self.system.N}")
        print(f"  K (total probes):     {self.system.K}")
        print(f"  M (sensing budget):   {self.system.M}")
        print(f"  M/K ratio:            {self.system.M/self.system.K:.2%}")
        print(f"  Phase mode:           {self.system.phase_mode}")
        if self.system.phase_mode == "discrete":
            print(f"  Phase bits:           {self.system.phase_bits}")
        print(f"  Probe type:           {self.system.probe_type}")

        print("\nData:")
        print(f"  Training samples:     {self.data.n_train}")
        print(f"  Validation samples:   {self.data.n_val}")
        print(f"  Test samples:          {self.data.n_test}")

        print("\nModel:")
        print(f"  Input size:           {2 * self.system.K} (masked vector)")
        print(f"  Hidden layers:        {self.model.hidden_sizes}")
        print(f"  Output size:          {self.system.K}")

        print("\nTraining:")
        print(f"  Batch size:           {self.training.batch_size}")
        print(f"  Learning rate:        {self.training.learning_rate}")
        print(f"  Device:               {self.training.device}")
        print("=" * 60)


def get_config(**kwargs) -> Config:
    """
    Create configuration with optional overrides.

    Example:
        config = get_config(system={'N': 64, 'K': 128, 'M': 16})
    """
    config = Config()

    for key, value in kwargs.items():
        if hasattr(config, key) and isinstance(value, dict):
            sub_config = getattr(config, key)
            for sub_key, sub_value in value.items():
                if hasattr(sub_config, sub_key):
                    setattr(sub_config, sub_key, sub_value)

    # Re-run post_init validation
    config.__post_init__()

    return config
