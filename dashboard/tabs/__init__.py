"""
Dashboard Tabs Module
=====================
Each tab in a separate, maintainable file.
"""

from dashboard.tabs.tab_system import create_system_tab
from dashboard.tabs.tab_physics import create_physics_tab
from dashboard.tabs.tab_model import create_model_tab
from dashboard.tabs.tab_training import create_training_tab
from dashboard.tabs.tab_evaluation import create_evaluation_tab
from dashboard.tabs.tab_visualization import create_visualization_tab

__all__ = [
    'create_system_tab',
    'create_physics_tab',
    'create_model_tab',
    'create_training_tab',
    'create_evaluation_tab',
    'create_visualization_tab',
    'tab_ablation'  # <--- NEW EXPORT
]
