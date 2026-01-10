"""
Model Architecture Registry for RIS Probe-Based ML System.

Allows users to easily add new model architectures without modifying core code.
"""

from typing import Dict, List

# Pre-defined model architectures
MODEL_REGISTRY: Dict[str, List[int]] = {
    # Standard models
    "Baseline_MLP": [256, 128],
    "Deep_MLP": [512, 512, 256],
    "Tiny_MLP": [64, 32],
    
    # High-capacity models
    "Ultra_Deep": [1024, 512, 256, 128, 64],
    "Wide_Deep": [1024, 1024, 512, 512],
    
    # Efficient models
    "Lightweight": [128, 64],
    "Minimal": [32, 16],
    
    # Research models
    "Experimental_A": [512, 256, 256, 128],
    "Experimental_B": [768, 384, 192, 96],
    
    # NEW PhD research architectures
    "ResNet_Style": [512, 512, 512, 512],  # Same width, like ResNet
    "Pyramid": [1024, 512, 256, 128, 64, 32],  # Gradual reduction
    "Hourglass": [128, 256, 512, 256, 128],  # Expand then contract
    "DoubleWide": [2048, 1024],  # Very wide, shallow
    "VeryDeep": [256, 256, 256, 256, 256, 256, 256, 256],  # 8 layers same size
    "Bottleneck": [512, 64, 512],  # Information bottleneck
    "Asymmetric": [1024, 256, 512, 128],  # Asymmetric design
    "PhD_Custom_1": [768, 512, 384, 256, 128],  # Balanced deep
    "PhD_Custom_2": [512, 512, 256, 256, 128, 128],  # Paired layers
}

def register_model(name: str, hidden_layers: List[int]) -> None:
    """Register a new model architecture."""
    MODEL_REGISTRY[name] = hidden_layers

def get_model_architecture(name: str) -> List[int]:
    """Get model architecture by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]

def list_models() -> List[str]:
    """List all available model architectures."""
    return list(MODEL_REGISTRY.keys())
