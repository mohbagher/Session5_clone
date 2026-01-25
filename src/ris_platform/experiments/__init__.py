"""
Experiments Package
===================
Exposes the main entry points for running and managing experiments.
"""

# Import the main runner functions so they are accessible directly
# from the package (e.g., 'from src.experiments import run_single_experiment')
from .runner import (
    run_single_experiment,
    run_experiment_stack,
    ExperimentResult,
    update_ui  # <-- Updated to match the new visual-enabled runner
)

# Optional: If you want to expose specific factories or tools widely
from .factories import create_physics_model
from .evaluation import evaluate_model

__all__ = [
    'run_single_experiment',
    'run_experiment_stack',
    'ExperimentResult',
    'update_ui',  # <-- Exporting the new UI updater
    'create_physics_model',
    'evaluate_model'
]