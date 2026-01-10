"""
Input Validation for RIS PhD Ultimate Dashboard.

Validates all experiment parameters before execution.
"""

from typing import Dict, List, Tuple, Optional


def validate_system_params(N: int, K: int, M: int, **kwargs) -> List[str]:
    """
    Validate system parameters.
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Validate N
    if not (4 <= N <= 256):
        errors.append(f"N must be between 4 and 256 (got {N})")
    
    # Validate K
    if not (4 <= K <= 512):
        errors.append(f"K must be between 4 and 512 (got {K})")
    
    # Validate M
    if not (1 <= M <= K):
        errors.append(f"M must be between 1 and K={K} (got {M})")
    
    # Validate P_tx
    P_tx = kwargs.get('P_tx', 1.0)
    if not (0.1 <= P_tx <= 10.0):
        errors.append(f"P_tx must be between 0.1 and 10.0 (got {P_tx})")
    
    # Validate sigma_h_sq
    sigma_h_sq = kwargs.get('sigma_h_sq', 1.0)
    if not (0.1 <= sigma_h_sq <= 10.0):
        errors.append(f"sigma_h_sq must be between 0.1 and 10.0 (got {sigma_h_sq})")
    
    # Validate sigma_g_sq
    sigma_g_sq = kwargs.get('sigma_g_sq', 1.0)
    if not (0.1 <= sigma_g_sq <= 10.0):
        errors.append(f"sigma_g_sq must be between 0.1 and 10.0 (got {sigma_g_sq})")
    
    # Validate phase_mode
    phase_mode = kwargs.get('phase_mode', 'continuous')
    if phase_mode not in ['continuous', 'discrete']:
        errors.append(f"phase_mode must be 'continuous' or 'discrete' (got {phase_mode})")
    
    # Validate phase_bits (if discrete)
    if phase_mode == 'discrete':
        phase_bits = kwargs.get('phase_bits', 3)
        if not (1 <= phase_bits <= 8):
            errors.append(f"phase_bits must be between 1 and 8 (got {phase_bits})")
    
    # Validate probe_type
    probe_type = kwargs.get('probe_type', 'continuous')
    valid_probe_types = ['continuous', 'binary', '2bit', 'hadamard', 'sobol', 'halton']
    if probe_type not in valid_probe_types:
        errors.append(f"probe_type must be one of {valid_probe_types} (got {probe_type})")
    
    return errors


def validate_model_config(hidden_sizes: List[int], dropout_prob: float, **kwargs) -> List[str]:
    """
    Validate model configuration.
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Validate hidden_sizes
    if not hidden_sizes:
        errors.append("hidden_sizes cannot be empty")
    else:
        for i, size in enumerate(hidden_sizes):
            if not isinstance(size, int) or size < 1:
                errors.append(f"Layer {i+1} size must be a positive integer (got {size})")
            if size > 4096:
                errors.append(f"Layer {i+1} size is very large ({size}). Consider reducing for memory.")
    
    # Validate dropout
    if not (0.0 <= dropout_prob <= 0.8):
        errors.append(f"dropout_prob must be between 0.0 and 0.8 (got {dropout_prob})")
    
    # Validate activation function
    activation = kwargs.get('activation_function', 'ReLU')
    valid_activations = ['ReLU', 'LeakyReLU', 'GELU', 'ELU', 'Tanh']
    if activation not in valid_activations:
        errors.append(f"activation_function must be one of {valid_activations} (got {activation})")
    
    # Validate weight init
    weight_init = kwargs.get('weight_init', 'xavier_uniform')
    valid_inits = ['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal']
    if weight_init not in valid_inits:
        errors.append(f"weight_init must be one of {valid_inits} (got {weight_init})")
    
    return errors


def validate_training_config(n_train: int, n_val: int, n_test: int, 
                            batch_size: int, learning_rate: float, 
                            weight_decay: float, n_epochs: int,
                            early_stop_patience: int, **kwargs) -> List[str]:
    """
    Validate training configuration.
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Validate dataset sizes
    if n_train < 100:
        errors.append(f"n_train should be at least 100 (got {n_train})")
    if n_val < 100:
        errors.append(f"n_val should be at least 100 (got {n_val})")
    if n_test < 100:
        errors.append(f"n_test should be at least 100 (got {n_test})")
    
    if n_train > 1_000_000:
        errors.append(f"n_train is very large ({n_train}). This may take a long time.")
    
    # Validate batch_size
    if batch_size < 1:
        errors.append(f"batch_size must be positive (got {batch_size})")
    if batch_size > n_train:
        errors.append(f"batch_size ({batch_size}) cannot exceed n_train ({n_train})")
    
    # Validate learning_rate
    if not (1e-6 <= learning_rate <= 1.0):
        errors.append(f"learning_rate should be between 1e-6 and 1.0 (got {learning_rate})")
    
    # Validate weight_decay
    if not (0.0 <= weight_decay <= 0.1):
        errors.append(f"weight_decay should be between 0.0 and 0.1 (got {weight_decay})")
    
    # Validate n_epochs
    if n_epochs < 1:
        errors.append(f"n_epochs must be at least 1 (got {n_epochs})")
    if n_epochs > 1000:
        errors.append(f"n_epochs is very large ({n_epochs}). This may take a long time.")
    
    # Validate early_stop_patience
    if early_stop_patience < 1:
        errors.append(f"early_stop_patience must be at least 1 (got {early_stop_patience})")
    if early_stop_patience > n_epochs:
        errors.append(f"early_stop_patience ({early_stop_patience}) should not exceed n_epochs ({n_epochs})")
    
    # Validate optimizer
    optimizer = kwargs.get('optimizer', 'Adam')
    valid_optimizers = ['Adam', 'AdamW', 'SGD', 'RMSprop']
    if optimizer not in valid_optimizers:
        errors.append(f"optimizer must be one of {valid_optimizers} (got {optimizer})")
    
    # Validate scheduler
    scheduler = kwargs.get('scheduler', 'ReduceLROnPlateau')
    valid_schedulers = ['ReduceLROnPlateau', 'CosineAnnealing', 'StepLR', 'None']
    if scheduler not in valid_schedulers:
        errors.append(f"scheduler must be one of {valid_schedulers} (got {scheduler})")
    
    # Validate normalization
    normalize_input = kwargs.get('normalize_input', True)
    if normalize_input:
        normalization_type = kwargs.get('normalization_type', 'mean')
        valid_norm_types = ['mean', 'std', 'log']
        if normalization_type not in valid_norm_types:
            errors.append(f"normalization_type must be one of {valid_norm_types} (got {normalization_type})")
    
    return errors


def validate_eval_config(top_m_values: List[int], K: int, **kwargs) -> List[str]:
    """
    Validate evaluation configuration.
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Validate top_m_values
    if not top_m_values:
        errors.append("top_m_values cannot be empty")
    else:
        for m in top_m_values:
            if not isinstance(m, int) or m < 1:
                errors.append(f"top_m value must be a positive integer (got {m})")
            if m > K:
                errors.append(f"top_m value ({m}) cannot exceed K ({K})")
    
    # Validate multi-seed runs
    multi_seed_runs = kwargs.get('multi_seed_runs', False)
    if multi_seed_runs:
        num_seeds = kwargs.get('num_seeds', 3)
        if not (1 <= num_seeds <= 10):
            errors.append(f"num_seeds must be between 1 and 10 (got {num_seeds})")
    
    # Validate model comparison
    compare_multiple_models = kwargs.get('compare_multiple_models', False)
    if compare_multiple_models:
        models_to_compare = kwargs.get('models_to_compare', [])
        if len(models_to_compare) < 2:
            errors.append("Must select at least 2 models for comparison")
    
    return errors


def validate_visualization_config(selected_plots: List[str], **kwargs) -> List[str]:
    """
    Validate visualization configuration.
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Validate selected_plots
    if not selected_plots:
        errors.append("Must select at least one plot type")
    
    # Validate figure_format
    figure_format = kwargs.get('figure_format', 'png')
    valid_formats = ['png', 'pdf', 'svg']
    if figure_format not in valid_formats:
        errors.append(f"figure_format must be one of {valid_formats} (got {figure_format})")
    
    # Validate DPI
    dpi = kwargs.get('dpi', 150)
    if not (72 <= dpi <= 300):
        errors.append(f"DPI must be between 72 and 300 (got {dpi})")
    
    # Validate color_palette
    color_palette = kwargs.get('color_palette', 'viridis')
    valid_palettes = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'seaborn']
    if color_palette not in valid_palettes:
        errors.append(f"color_palette must be one of {valid_palettes} (got {color_palette})")
    
    # Validate output_dir
    output_dir = kwargs.get('output_dir', 'results/')
    if not output_dir:
        errors.append("output_dir cannot be empty")
    
    return errors


def get_validation_errors(config_dict: Dict) -> Tuple[bool, List[str]]:
    """
    Run all validations and return consolidated errors.
    
    Args:
        config_dict: Dictionary with all configuration parameters
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    all_errors = []
    
    # Validate system parameters
    all_errors.extend(validate_system_params(
        N=config_dict.get('N', 32),
        K=config_dict.get('K', 64),
        M=config_dict.get('M', 8),
        P_tx=config_dict.get('P_tx', 1.0),
        sigma_h_sq=config_dict.get('sigma_h_sq', 1.0),
        sigma_g_sq=config_dict.get('sigma_g_sq', 1.0),
        phase_mode=config_dict.get('phase_mode', 'continuous'),
        phase_bits=config_dict.get('phase_bits', 3),
        probe_type=config_dict.get('probe_type', 'continuous')
    ))
    
    # Validate model configuration
    all_errors.extend(validate_model_config(
        hidden_sizes=config_dict.get('hidden_sizes', [256, 128]),
        dropout_prob=config_dict.get('dropout_prob', 0.1),
        activation_function=config_dict.get('activation_function', 'ReLU'),
        weight_init=config_dict.get('weight_init', 'xavier_uniform')
    ))
    
    # Validate training configuration
    all_errors.extend(validate_training_config(
        n_train=config_dict.get('n_train', 50000),
        n_val=config_dict.get('n_val', 5000),
        n_test=config_dict.get('n_test', 5000),
        batch_size=config_dict.get('batch_size', 128),
        learning_rate=config_dict.get('learning_rate', 1e-3),
        weight_decay=config_dict.get('weight_decay', 1e-4),
        n_epochs=config_dict.get('n_epochs', 50),
        early_stop_patience=config_dict.get('early_stop_patience', 10),
        optimizer=config_dict.get('optimizer', 'Adam'),
        scheduler=config_dict.get('scheduler', 'ReduceLROnPlateau'),
        normalize_input=config_dict.get('normalize_input', True),
        normalization_type=config_dict.get('normalization_type', 'mean')
    ))
    
    # Validate evaluation configuration
    all_errors.extend(validate_eval_config(
        top_m_values=config_dict.get('top_m_values', [1, 2, 4, 8]),
        K=config_dict.get('K', 64),
        multi_seed_runs=config_dict.get('multi_seed_runs', False),
        num_seeds=config_dict.get('num_seeds', 3),
        compare_multiple_models=config_dict.get('compare_multiple_models', False),
        models_to_compare=config_dict.get('models_to_compare', [])
    ))
    
    # Validate visualization configuration
    all_errors.extend(validate_visualization_config(
        selected_plots=config_dict.get('selected_plots', []),
        figure_format=config_dict.get('figure_format', 'png'),
        dpi=config_dict.get('dpi', 150),
        color_palette=config_dict.get('color_palette', 'viridis'),
        output_dir=config_dict.get('output_dir', 'results/')
    ))
    
    is_valid = len(all_errors) == 0
    return is_valid, all_errors
