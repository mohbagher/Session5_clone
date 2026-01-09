"""
Shared defaults for experiment tasks to keep training comparisons consistent.
"""

from typing import Dict

from config import get_config


DEFAULT_DATA = {
    'n_train': 20000,
    'n_val': 2000,
    'n_test': 2000
}

DEFAULT_TRAINING = {
    'n_epochs': 50,
    'batch_size': 128
}


def build_task_config(N: int, K: int, M: int, seed: int) :
    """Create a standardized config for task training runs."""
    return get_config(
        system={'N': N, 'K': K, 'M': M},
        data={**DEFAULT_DATA, 'seed': seed},
        training=DEFAULT_TRAINING
    )
