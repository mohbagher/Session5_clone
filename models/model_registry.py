"""
Model Registry
==============
Central registry for all model architectures.
"""

import torch.nn as nn
from typing import Dict, Type, Optional
import logging

logger = logging.getLogger(__name__)


# Import all model classes
from models.base_models import (
    BaselineMLPPredictor,
    AttentionMLPPredictor,
    ResidualMLPPredictor
)


# Model registry
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    'Baseline_MLP': BaselineMLPPredictor,
    'Attention_MLP': AttentionMLPPredictor,
    'Residual_MLP': ResidualMLPPredictor,
}


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


def list_available_models() -> list:
    """List all available model presets."""
    return list(MODEL_REGISTRY.keys())


def register_model(name: str, model_class: Type[nn.Module]):
    """
    Register a new model class.

    Args:
        name: Model name
        model_class: Model class
    """
    MODEL_REGISTRY[name] = model_class
    logger.info(f"Registered model: {name}")