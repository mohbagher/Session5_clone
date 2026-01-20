"""
Model Registry
==============
Central registry for all model architectures with preset configurations.
"""

import torch.nn as nn
from typing import Dict, Type, List
import logging

logger = logging.getLogger(__name__)


# Import all model classes
from models.base_models import (
    BaselineMLPPredictor,
    AttentionMLPPredictor,
    ResidualMLPPredictor
)


# =============================================================================
# MODEL ARCHITECTURE PRESETS
# =============================================================================

MODEL_ARCHITECTURES = {
    # Standard MLPs
    'Baseline_MLP': [512, 256, 128],
    'Deep_MLP': [512, 512, 256, 256, 128, 128],
    'Tiny_MLP': [128, 64],
    'Wide_Deep': [1024, 512, 256, 128],

    # Specialized architectures
    'Attention_MLP': [512, 256, 128],
    'Residual_MLP': [512, 256, 128],

    # Research architectures
    'PhD_Custom_1': [256, 256, 128],
    'PhD_Custom_2': [512, 256, 256, 128],
    'Pyramid': [512, 384, 256, 192, 128, 96, 64],
    'Hourglass': [512, 256, 128, 64, 128, 256, 512],
    'ResNet_Style': [256, 256, 256, 256],

    # Learnable M models
    'LearnedTopK_MLP': [512, 256, 128],
    'Gumbel_MLP': [512, 256, 128],
    'RL_MLP': [512, 256, 128],
}


# Model class registry
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    'Baseline_MLP': BaselineMLPPredictor,
    'Deep_MLP': BaselineMLPPredictor,
    'Tiny_MLP': BaselineMLPPredictor,
    'Wide_Deep': BaselineMLPPredictor,
    'Attention_MLP': AttentionMLPPredictor,
    'Residual_MLP': ResidualMLPPredictor,
    'PhD_Custom_1': BaselineMLPPredictor,
    'PhD_Custom_2': BaselineMLPPredictor,
    'Pyramid': BaselineMLPPredictor,
    'Hourglass': BaselineMLPPredictor,
    'ResNet_Style': ResidualMLPPredictor,
    'LearnedTopK_MLP': BaselineMLPPredictor,  # Will use learnable_m_models when available
    'Gumbel_MLP': BaselineMLPPredictor,
    'RL_MLP': BaselineMLPPredictor,
}


# =============================================================================
# PUBLIC API
# =============================================================================

def get_model_architecture(model_name: str) -> List[int]:
    """
    Get hidden layer sizes for a model preset.

    Args:
        model_name: Name of model preset

    Returns:
        List of hidden layer sizes

    Raises:
        ValueError: If model_name not found
    """
    if model_name not in MODEL_ARCHITECTURES:
        raise ValueError(
            f"Unknown model architecture: {model_name}. "
            f"Available: {list(MODEL_ARCHITECTURES.keys())}"
        )

    return MODEL_ARCHITECTURES[model_name]


def get_model_class(model_name: str) -> Type[nn.Module]:
    """
    Get model class from registry.

    Args:
        model_name: Name of model preset

    Returns:
        Model class

    Raises:
        ValueError: If model_name not found
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    return MODEL_REGISTRY[model_name]


def list_available_models() -> List[str]:
    """List all available model presets."""
    return list(MODEL_REGISTRY.keys())


def list_models() -> List[str]:
    """Alias for list_available_models()."""
    return list_available_models()


def register_model(name: str,
                   model_class: Type[nn.Module],
                   architecture: List[int] = None):
    """
    Register a new model class and architecture.

    Args:
        name: Model name
        model_class: Model class
        architecture: Hidden layer sizes (optional)
    """
    MODEL_REGISTRY[name] = model_class
    if architecture is not None:
        MODEL_ARCHITECTURES[name] = architecture
    logger.info(f"Registered model: {name}")


def get_model_info(model_name: str) -> Dict:
    """
    Get information about a model.

    Args:
        model_name: Name of model preset

    Returns:
        Dictionary with model information
    """
    if model_name not in MODEL_REGISTRY:
        return {'error': f'Model {model_name} not found'}

    return {
        'name': model_name,
        'class': MODEL_REGISTRY[model_name].__name__,
        'architecture': MODEL_ARCHITECTURES.get(model_name, []),
        'num_layers': len(MODEL_ARCHITECTURES.get(model_name, [])),
    }


# =============================================================================
# CATEGORY HELPERS
# =============================================================================

def get_models_by_category() -> Dict[str, List[str]]:
    """Get models organized by category."""
    return {
        'Standard MLPs': [
            'Baseline_MLP',
            'Deep_MLP',
            'Tiny_MLP',
            'Wide_Deep'
        ],
        'Attention-Based': [
            'Attention_MLP'
        ],
        'Residual': [
            'Residual_MLP',
            'ResNet_Style'
        ],
        'Research Architectures': [
            'PhD_Custom_1',
            'PhD_Custom_2',
            'Pyramid',
            'Hourglass'
        ],
        'Learnable M Selection': [
            'LearnedTopK_MLP',
            'Gumbel_MLP',
            'RL_MLP'
        ]
    }


def print_model_summary():
    """Print a summary of all available models."""
    categories = get_models_by_category()

    print("="*70)
    print("AVAILABLE MODEL ARCHITECTURES")
    print("="*70)

    for category, models in categories.items():
        print(f"\n{category}:")
        for model_name in models:
            arch = MODEL_ARCHITECTURES.get(model_name, [])
            print(f"  - {model_name:20s} {arch}")

    print("\n" + "="*70)
    print(f"Total: {len(MODEL_REGISTRY)} models available")
    print("="*70)