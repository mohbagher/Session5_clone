"""
Configuration Manager for RIS PhD Ultimate Dashboard.

Handles saving and loading experiment configurations in JSON and YAML formats.
"""

import json
import yaml
import os
from typing import Dict, Any
from pathlib import Path


def config_to_dict(widgets_dict: Dict) -> Dict[str, Any]:
    """
    Extract configuration values from widgets dictionary.
    
    Args:
        widgets_dict: Dictionary of all widgets
        
    Returns:
        Dictionary with all configuration values
    """
    from dashboard.callbacks import get_current_hidden_sizes
    
    config = {
        # System & Physics
        'N': widgets_dict['N'].value,
        'K': widgets_dict['K'].value,
        'M': widgets_dict['M'].value,
        'P_tx': widgets_dict['P_tx'].value,
        'sigma_h_sq': widgets_dict['sigma_h_sq'].value,
        'sigma_g_sq': widgets_dict['sigma_g_sq'].value,
        'phase_mode': widgets_dict['phase_mode'].value,
        'phase_bits': widgets_dict['phase_bits'].value,
        'probe_type': widgets_dict['probe_type'].value,
        
        # Model Architecture
        'model_preset': widgets_dict['model_preset'].value,
        'hidden_sizes': get_current_hidden_sizes(widgets_dict),
        'dropout_prob': widgets_dict['dropout_prob'].value,
        'use_batch_norm': widgets_dict['use_batch_norm'].value,
        'activation_function': widgets_dict['activation_function'].value,
        'weight_init': widgets_dict['weight_init'].value,
        
        # Training Configuration
        'n_train': widgets_dict['n_train'].value,
        'n_val': widgets_dict['n_val'].value,
        'n_test': widgets_dict['n_test'].value,
        'seed': widgets_dict['seed'].value,
        'normalize_input': widgets_dict['normalize_input'].value,
        'normalization_type': widgets_dict['normalization_type'].value,
        'batch_size': widgets_dict['batch_size'].value,
        'learning_rate': widgets_dict['learning_rate'].value,
        'weight_decay': widgets_dict['weight_decay'].value,
        'n_epochs': widgets_dict['n_epochs'].value,
        'early_stop_patience': widgets_dict['early_stop_patience'].value,
        'optimizer': widgets_dict['optimizer'].value,
        'scheduler': widgets_dict['scheduler'].value,
        
        # Evaluation & Comparison
        'top_m_values': list(widgets_dict['top_m_values'].value),
        'compare_multiple_models': widgets_dict['compare_multiple_models'].value,
        'models_to_compare': list(widgets_dict['models_to_compare'].value),
        'multi_seed_runs': widgets_dict['multi_seed_runs'].value,
        'num_seeds': widgets_dict['num_seeds'].value,
        'compute_confidence_intervals': widgets_dict['compute_confidence_intervals'].value,
        
        # Visualization
        'selected_plots': list(widgets_dict['selected_plots'].value),
        'figure_format': widgets_dict['figure_format'].value,
        'dpi': widgets_dict['dpi'].value,
        'color_palette': widgets_dict['color_palette'].value,
        'save_plots': widgets_dict['save_plots'].value,
        'output_dir': widgets_dict['output_dir'].value,

        # Backends and physocs
        'physics_backend': widgets_dict['physics_backend'].value,
        'matlab_scenario': widgets_dict['matlab_scenario'].value,
        'realism_profile': widgets_dict['realism_profile'].value,
        'carrier_frequency': widgets_dict['carrier_frequency'].value,
        'doppler_shift_matlab': widgets_dict['doppler_shift_matlab'].value,
        'use_custom_impairments': widgets_dict['use_custom_impairments'].value,
    }
    
    return config


def dict_to_widgets(config_dict: Dict[str, Any], widgets_dict: Dict):
    """
    Set widget values from configuration dictionary.
    
    Args:
        config_dict: Dictionary with configuration values
        widgets_dict: Dictionary of all widgets
    """
    # System & Physics
    if 'N' in config_dict:
        widgets_dict['N'].value = config_dict['N']
    if 'K' in config_dict:
        widgets_dict['K'].value = config_dict['K']
    if 'M' in config_dict:
        widgets_dict['M'].value = config_dict['M']
    if 'P_tx' in config_dict:
        widgets_dict['P_tx'].value = config_dict['P_tx']
    if 'sigma_h_sq' in config_dict:
        widgets_dict['sigma_h_sq'].value = config_dict['sigma_h_sq']
    if 'sigma_g_sq' in config_dict:
        widgets_dict['sigma_g_sq'].value = config_dict['sigma_g_sq']
    if 'phase_mode' in config_dict:
        widgets_dict['phase_mode'].value = config_dict['phase_mode']
    if 'phase_bits' in config_dict:
        widgets_dict['phase_bits'].value = config_dict['phase_bits']
    if 'probe_type' in config_dict:
        widgets_dict['probe_type'].value = config_dict['probe_type']
    
    # Model Architecture
    if 'model_preset' in config_dict:
        widgets_dict['model_preset'].value = config_dict['model_preset']
    if 'dropout_prob' in config_dict:
        widgets_dict['dropout_prob'].value = config_dict['dropout_prob']
    if 'use_batch_norm' in config_dict:
        widgets_dict['use_batch_norm'].value = config_dict['use_batch_norm']
    if 'activation_function' in config_dict:
        widgets_dict['activation_function'].value = config_dict['activation_function']
    if 'weight_init' in config_dict:
        widgets_dict['weight_init'].value = config_dict['weight_init']
    
    # If custom model and hidden_sizes provided
    if config_dict.get('model_preset') == 'Custom' and 'hidden_sizes' in config_dict:
        hidden_sizes = config_dict['hidden_sizes']
        widgets_dict['num_layers'].value = len(hidden_sizes)
        # Layer sizes will be updated by callbacks
    
    # Training Configuration
    if 'n_train' in config_dict:
        widgets_dict['n_train'].value = config_dict['n_train']
    if 'n_val' in config_dict:
        widgets_dict['n_val'].value = config_dict['n_val']
    if 'n_test' in config_dict:
        widgets_dict['n_test'].value = config_dict['n_test']
    if 'seed' in config_dict:
        widgets_dict['seed'].value = config_dict['seed']
    if 'normalize_input' in config_dict:
        widgets_dict['normalize_input'].value = config_dict['normalize_input']
    if 'normalization_type' in config_dict:
        widgets_dict['normalization_type'].value = config_dict['normalization_type']
    if 'batch_size' in config_dict:
        widgets_dict['batch_size'].value = config_dict['batch_size']
    if 'learning_rate' in config_dict:
        widgets_dict['learning_rate'].value = config_dict['learning_rate']
    if 'weight_decay' in config_dict:
        widgets_dict['weight_decay'].value = config_dict['weight_decay']
    if 'n_epochs' in config_dict:
        widgets_dict['n_epochs'].value = config_dict['n_epochs']
    if 'early_stop_patience' in config_dict:
        widgets_dict['early_stop_patience'].value = config_dict['early_stop_patience']
    if 'optimizer' in config_dict:
        widgets_dict['optimizer'].value = config_dict['optimizer']
    if 'scheduler' in config_dict:
        widgets_dict['scheduler'].value = config_dict['scheduler']
    
    # Evaluation & Comparison
    if 'top_m_values' in config_dict:
        widgets_dict['top_m_values'].value = tuple(config_dict['top_m_values'])
    if 'compare_multiple_models' in config_dict:
        widgets_dict['compare_multiple_models'].value = config_dict['compare_multiple_models']
    if 'models_to_compare' in config_dict:
        widgets_dict['models_to_compare'].value = tuple(config_dict['models_to_compare'])
    if 'multi_seed_runs' in config_dict:
        widgets_dict['multi_seed_runs'].value = config_dict['multi_seed_runs']
    if 'num_seeds' in config_dict:
        widgets_dict['num_seeds'].value = config_dict['num_seeds']
    if 'compute_confidence_intervals' in config_dict:
        widgets_dict['compute_confidence_intervals'].value = config_dict['compute_confidence_intervals']
    
    # Visualization
    if 'selected_plots' in config_dict:
        widgets_dict['selected_plots'].value = tuple(config_dict['selected_plots'])
    if 'figure_format' in config_dict:
        widgets_dict['figure_format'].value = config_dict['figure_format']
    if 'dpi' in config_dict:
        widgets_dict['dpi'].value = config_dict['dpi']
    if 'color_palette' in config_dict:
        widgets_dict['color_palette'].value = config_dict['color_palette']
    if 'save_plots' in config_dict:
        widgets_dict['save_plots'].value = config_dict['save_plots']
    if 'output_dir' in config_dict:
        widgets_dict['output_dir'].value = config_dict['output_dir']


def save_config_json(config_dict: Dict[str, Any], filepath: str) -> bool:
    """
    Save configuration to JSON file.
    
    Args:
        config_dict: Configuration dictionary
        filepath: Path to save JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving JSON config: {e}")
        return False


def save_config_yaml(config_dict: Dict[str, Any], filepath: str) -> bool:
    """
    Save configuration to YAML file.
    
    Args:
        config_dict: Configuration dictionary
        filepath: Path to save YAML file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving YAML config: {e}")
        return False


def load_config_json(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    
    return config_dict


def load_config_yaml(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        filepath: Path to YAML file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If file is not valid YAML
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return config_dict


def auto_detect_format(filepath: str) -> str:
    """
    Auto-detect configuration file format from extension.
    
    Args:
        filepath: Path to configuration file
        
    Returns:
        'json' or 'yaml'
    """
    ext = Path(filepath).suffix.lower()
    if ext in ['.yaml', '.yml']:
        return 'yaml'
    elif ext == '.json':
        return 'json'
    else:
        # Default to JSON
        return 'json'


def save_config(config_dict: Dict[str, Any], filepath: str, format: str = 'auto') -> bool:
    """
    Save configuration with auto-detection of format.
    
    Args:
        config_dict: Configuration dictionary
        filepath: Path to save file
        format: 'json', 'yaml', or 'auto' (default: 'auto')
        
    Returns:
        True if successful, False otherwise
    """
    if format == 'auto':
        format = auto_detect_format(filepath)
    
    if format == 'yaml':
        return save_config_yaml(config_dict, filepath)
    else:
        return save_config_json(config_dict, filepath)


def load_config(filepath: str, format: str = 'auto') -> Dict[str, Any]:
    """
    Load configuration with auto-detection of format.
    
    Args:
        filepath: Path to configuration file
        format: 'json', 'yaml', or 'auto' (default: 'auto')
        
    Returns:
        Configuration dictionary
    """
    if format == 'auto':
        format = auto_detect_format(filepath)
    
    if format == 'yaml':
        return load_config_yaml(filepath)
    else:
        return load_config_json(filepath)
