"""
Models Module
=============
Neural network architectures for RIS beamforming.
"""

from models.base_models import (
    BaselineMLPPredictor,
    AttentionMLPPredictor,
    ResidualMLPPredictor
)

from models.model_registry import (
    get_model_class,
    list_available_models,
    register_model,
    MODEL_REGISTRY
)

__all__ = [
    # Model classes
    'BaselineMLPPredictor',
    'AttentionMLPPredictor',
    'ResidualMLPPredictor',

    # Registry functions
    'get_model_class',
    'list_available_models',
    'register_model',
    'MODEL_REGISTRY'
]