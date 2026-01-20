"""
Experiment Runner
=================
Orchestrates experiment execution with backend selection support.
"""

import numpy as np
import torch
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


def run_single_experiment(config: Dict, widget_dict: Dict) -> Dict:
    """
    Run single experiment with Phase 2 MATLAB backend support.

    Args:
        config: Experiment configuration dictionary
        widget_dict: Dashboard widgets for status updates

    Returns:
        results: Experiment results with metadata
    """

    logger.info("=" * 70)
    logger.info("STARTING EXPERIMENT")
    logger.info("=" * 70)

    # Initialize results dictionary
    results = {
        'config': config.copy(),
        'metadata': {
            'start_time': datetime.now().isoformat(),
            'backend': config.get('physics_backend', 'python'),
        }
    }

    try:
        # ====================================================================
        # PHASE 2: BACKEND-AWARE CHANNEL GENERATION
        # ====================================================================

        backend = config.get('physics_backend', 'python')

        update_status(widget_dict, f"🔧 Initializing {backend.upper()} backend...")
        logger.info(f"Backend selected: {backend}")

        # Calculate total samples needed
        total_samples = config.get('n_train', 50000) + config.get('n_val', 5000) + config.get('n_test', 5000)

        if backend == 'matlab':
            # ================================================================
            # MATLAB BACKEND PATH
            # ================================================================

            h, g, channel_metadata = generate_channels_matlab(
                config=config,
                total_samples=total_samples,
                widget_dict=widget_dict
            )

            # Store MATLAB-specific metadata
            results['metadata']['channel_generation'] = channel_metadata
            results['metadata']['matlab_scenario'] = config.get('matlab_scenario')
            results['metadata']['matlab_version'] = channel_metadata.get('matlab_version')
            results['metadata']['toolbox'] = channel_metadata.get('toolbox')
            results['metadata']['reference'] = channel_metadata.get('reference')

        else:
            # ================================================================
            # PYTHON BACKEND PATH (DEFAULT)
            # ================================================================

            h, g, channel_metadata = generate_channels_python(
                config=config,
                total_samples=total_samples,
                widget_dict=widget_dict
            )

            # Store Python metadata
            results['metadata']['channel_generation'] = channel_metadata

        # ====================================================================
        # PHASE 1: APPLY IMPAIRMENTS (SAME FOR BOTH BACKENDS)
        # ====================================================================

        if config.get('use_custom_impairments', False) or config.get('realism_profile') != 'ideal':
            h, g = apply_impairments(h, g, config, widget_dict)
            results['metadata']['impairments_applied'] = True
        else:
            results['metadata']['impairments_applied'] = False

        # ====================================================================
        # CONTINUE WITH TRAINING (BACKEND-AGNOSTIC FROM HERE)
        # ====================================================================

        update_status(widget_dict, "📊 Generating probe bank...")
        probe_bank = generate_probe_bank(config)

        update_status(widget_dict, "🎯 Generating training data...")
        train_data = generate_training_data(h, g, probe_bank, config)

        update_status(widget_dict, "🧠 Training model...")
        model, training_history = train_model(train_data, config, widget_dict)

        update_status(widget_dict, "📈 Evaluating performance...")
        eval_results = evaluate_model(model, h, g, probe_bank, config)

        # Store results
        results['training_history'] = training_history
        results['evaluation'] = eval_results
        results['metadata']['end_time'] = datetime.now().isoformat()
        results['metadata']['success'] = True

        update_status(widget_dict, "✅ Experiment completed successfully!")

        return results

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        results['metadata']['success'] = False
        results['metadata']['error'] = str(e)
        results['metadata']['end_time'] = datetime.now().isoformat()

        update_status(widget_dict, f"❌ Experiment failed: {e}")

        return results


def generate_channels_matlab(
        config: Dict,
        total_samples: int,
        widget_dict: Dict
) -> Tuple[np.ndarray, np.ndarray, Dict]:  # FIXED: Added return type hint
    """
    Generate channels using MATLAB backend.

    Args:
        config: Configuration dictionary
        total_samples: Number of channel realizations needed
        widget_dict: Widgets for status updates

    Returns:
        h: BS-RIS channels (N, total_samples)
        g: RIS-UE channels (N, total_samples)
        metadata: Generation metadata
    """

    from physics.matlab_backend.matlab_source import MATLABEngineSource

    # Get parameters
    N = config['N']
    K = config['K']
    sigma_h_sq = config.get('sigma_h_sq', 1.0)
    sigma_g_sq = config.get('sigma_g_sq', 1.0)
    seed = config.get('seed', 42)

    matlab_scenario = config.get('matlab_scenario', 'rayleigh_basic')

    # MATLAB-specific parameters
    matlab_params = {}
    if matlab_scenario == 'cdl_ris':
        matlab_params['carrier_frequency'] = config.get('carrier_frequency', 28e9)
        matlab_params['delay_profile'] = config.get('delay_profile', 'CDL-C')
        matlab_params['doppler_shift'] = config.get('doppler_shift_matlab', 5.0)

    update_status(widget_dict, f"🔧 Starting MATLAB Engine ({matlab_scenario})...")

    try:
        # Create MATLAB source
        source = MATLABEngineSource(
            scenario=matlab_scenario,
            **matlab_params
        )

        update_status(widget_dict, f"📡 Generating {total_samples} channel realizations with MATLAB...")

        # Generate channels
        h, g, metadata = source.generate_channels(
            N=N,
            K=K,
            num_samples=total_samples,
            sigma_h_sq=sigma_h_sq,
            sigma_g_sq=sigma_g_sq,
            seed=seed,
            **matlab_params
        )

        logger.info(f"MATLAB channel generation completed")
        logger.info(f"  Toolbox: {metadata.get('toolbox')}")
        logger.info(f"  Scenario: {metadata.get('scenario')}")
        logger.info(f"  Function: {metadata.get('function')}")

        update_status(widget_dict, f"✅ MATLAB channels generated successfully")

        return h, g, metadata

    except Exception as e:
        logger.error(f"MATLAB backend failed: {e}")
        update_status(widget_dict, f"⚠️ MATLAB failed, falling back to Python: {e}")

        # Fallback to Python
        return generate_channels_python(config, total_samples, widget_dict)


def generate_channels_python(
        config: Dict,
        total_samples: int,
        widget_dict: Dict
) -> Tuple[np.ndarray, np.ndarray, Dict]:  # FIXED: Added return type hint
    """
    Generate channels using Python backend.

    Args:
        config: Configuration dictionary
        total_samples: Number of channel realizations needed
        widget_dict: Widgets for status updates

    Returns:
        h: BS-RIS channels (N, total_samples)
        g: RIS-UE channels (N, total_samples)
        metadata: Generation metadata
    """

    from physics import create_source_from_name

    # Get parameters
    N = config['N']
    K = config['K']
    sigma_h_sq = config.get('sigma_h_sq', 1.0)
    sigma_g_sq = config.get('sigma_g_sq', 1.0)
    seed = config.get('seed', 42)

    channel_source_name = config.get('channel_source', 'python_synthetic')

    update_status(widget_dict, f"📡 Generating {total_samples} channels with Python ({channel_source_name})...")

    # Create source
    source = create_source_from_name(channel_source_name)

    # Generate channels
    h, g = source.generate_channels(
        N=N,
        K=K,
        num_samples=total_samples,
        sigma_h_sq=sigma_h_sq,
        sigma_g_sq=sigma_g_sq,
        seed=seed
    )

    # Create metadata
    metadata = {
        'backend_name': 'python',
        'source': channel_source_name,
        'method': 'numpy_synthetic',
        'num_samples': total_samples,
        'N': N,
        'sigma_h_sq': sigma_h_sq,
        'sigma_g_sq': sigma_g_sq,
        'seed': seed
    }

    logger.info(f"Python channel generation completed")
    logger.info(f"  Source: {channel_source_name}")

    update_status(widget_dict, f"✅ Python channels generated successfully")

    return h, g, metadata


def apply_impairments(
        h: np.ndarray,
        g: np.ndarray,
        config: Dict,
        widget_dict: Dict
) -> Tuple[np.ndarray, np.ndarray]:  # FIXED: Added return type hint
    """
    Apply Phase 1 impairments to channels.

    Args:
        h: BS-RIS channels (N, num_samples)
        g: RIS-UE channels (N, num_samples)
        config: Configuration dictionary
        widget_dict: Widgets for status updates

    Returns:
        h_impaired: Impaired BS-RIS channels
        g_impaired: Impaired RIS-UE channels
    """

    from physics import create_pipeline_from_profile

    realism_profile = config.get('realism_profile', 'ideal')

    if realism_profile == 'ideal' and not config.get('use_custom_impairments', False):
        # No impairments
        return h, g

    update_status(widget_dict, f"🔨 Applying impairments ({realism_profile})...")

    # Create impairment pipeline
    pipeline = create_pipeline_from_profile(realism_profile)

    # Apply to all realizations
    num_samples = h.shape[1]
    h_impaired = np.zeros_like(h)
    g_impaired = np.zeros_like(g)

    for i in range(num_samples):
        h_impaired[:, i], g_impaired[:, i] = pipeline.apply(h[:, i], g[:, i], config)

        if (i + 1) % 1000 == 0:
            update_status(widget_dict, f"🔨 Applying impairments... {i + 1}/{num_samples}")

    logger.info(f"Impairments applied to {num_samples} channel realizations")

    return h_impaired, g_impaired


def generate_probe_bank(config: Dict) -> np.ndarray:  # FIXED: Added return type hint
    """Generate probe bank based on configuration."""

    from data.probe_generators import get_probe_bank  # FIXED: Added import

    K = config['K']
    N = config['N']
    probe_type = config.get('probe_type', 'continuous')
    seed = config.get('seed', 42)

    probe_bank = get_probe_bank(
        probe_type=probe_type,
        K=K,
        N=N,
        seed=seed
    )

    logger.info(f"Probe bank generated: {probe_type}, K={K}")

    return probe_bank


def generate_training_data(
        h: np.ndarray,
        g: np.ndarray,
        probe_bank: np.ndarray,
        config: Dict
) -> Dict:  # FIXED: Added return type hint
    """Generate training data from channels and probe bank."""

    from data.data_generation import generate_limited_probing_dataset

    N = config['N']
    K = config['K']
    M = config['M']
    P_tx = config.get('P_tx', 1.0)

    n_train = config.get('n_train', 50000)
    n_val = config.get('n_val', 5000)
    n_test = config.get('n_test', 5000)

    # Split channels
    h_train = h[:, :n_train]
    g_train = g[:, :n_train]

    h_val = h[:, n_train:n_train + n_val]
    g_val = g[:, n_train:n_train + n_val]

    h_test = h[:, n_train + n_val:]
    g_test = g[:, n_train + n_val:]

    # Generate datasets
    train_data = generate_limited_probing_dataset(
        h_train, g_train, probe_bank,
        N=N, K=K, M=M, P_tx=P_tx
    )

    val_data = generate_limited_probing_dataset(
        h_val, g_val, probe_bank,
        N=N, K=K, M=M, P_tx=P_tx
    )

    test_data = generate_limited_probing_dataset(
        h_test, g_test, probe_bank,
        N=N, K=K, M=M, P_tx=P_tx
    )

    logger.info(f"Training data generated: {n_train} train, {n_val} val, {n_test} test")

    return {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }


def train_model(
        train_data: Dict,
        config: Dict,
        widget_dict: Dict
) -> Tuple[Any, Dict]:  # FIXED: Added return type hint
    """Train the model."""

    from training.trainer import train_model as trainer_train_model
    from models.model_registry import get_model_class

    # Get model
    model_name = config.get('model_preset', 'Baseline_MLP')
    model_class = get_model_class(model_name)

    # Training parameters
    batch_size = config.get('batch_size', 128)
    learning_rate = config.get('learning_rate', 1e-3)
    n_epochs = config.get('n_epochs', 50)
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Create model instance
    input_size = 2 * config['K']
    output_size = config['K']
    hidden_sizes = config.get('hidden_sizes', [512, 256, 128])

    model = model_class(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size
    )

    # Train
    model, history = trainer_train_model(
        model=model,
        train_data=train_data['train'],
        val_data=train_data['val'],
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        device=device,
        widget_dict=widget_dict
    )

    return model, history


def evaluate_model(
        model: Any,
        h: np.ndarray,
        g: np.ndarray,
        probe_bank: np.ndarray,
        config: Dict
) -> Dict:  # FIXED: Added return type hint
    """Evaluate the trained model."""

    from evaluation.evaluator import evaluate_model as evaluator_evaluate_model

    # Evaluation parameters
    top_m_values = config.get('top_m_values', [1, 2, 4, 8])

    # Split test data
    n_train = config.get('n_train', 50000)
    n_val = config.get('n_val', 5000)

    h_test = h[:, n_train + n_val:]
    g_test = g[:, n_train + n_val:]

    # Evaluate
    results = evaluator_evaluate_model(
        model=model,
        h_test=h_test,
        g_test=g_test,
        probe_bank=probe_bank,
        config=config,
        top_m_values=top_m_values
    )

    return results


def update_status(widget_dict: Dict, message: str) -> None:  # FIXED: Added return type hint
    """
    Update status display in dashboard.

    Args:
        widget_dict: Dictionary of dashboard widgets
        message: Status message to display
    """

    if 'status_output' in widget_dict:
        with widget_dict['status_output']:
            print(message)

    logger.info(message)


def run_experiment_stack(stack: list, widget_dict: Dict) -> list:
    """
    Run a stack of experiments sequentially.

    Args:
        stack: List of experiment configurations
        widget_dict: Dashboard widgets

    Returns:
        results_list: List of results for each experiment
    """

    results_list = []

    for i, config in enumerate(stack):
        update_status(widget_dict, f"\n{'=' * 70}")
        update_status(widget_dict, f"EXPERIMENT {i + 1}/{len(stack)}")
        update_status(widget_dict, f"{'=' * 70}\n")

        result = run_single_experiment(config, widget_dict)
        results_list.append(result)

        # Log backend used
        backend = result['metadata'].get('backend', 'unknown')
        update_status(widget_dict, f"✓ Experiment {i + 1} completed using {backend.upper()} backend")

    update_status(widget_dict, f"\n{'=' * 70}")
    update_status(widget_dict, f"ALL {len(stack)} EXPERIMENTS COMPLETED")
    update_status(widget_dict, f"{'=' * 70}\n")

    return results_list


def run_multi_model_comparison(
        configs: list,
        widget_dict: Dict
) -> list:
    """
    Run comparison across multiple models.

    Args:
        configs: List of configurations (one per model)
        widget_dict: Dashboard widgets

    Returns:
        results_list: Results for each model
    """

    results_list = []

    for i, config in enumerate(configs):
        update_status(widget_dict, f"\n{'=' * 70}")
        update_status(widget_dict, f"MODEL {i + 1}/{len(configs)}: {config.get('model_preset')}")
        update_status(widget_dict, f"{'=' * 70}\n")

        result = run_single_experiment(config, widget_dict)
        results_list.append(result)

    return results_list


def run_multi_seed_experiment(
        base_config: Dict,
        seeds: list,
        widget_dict: Dict
) -> list:
    """
    Run experiment with multiple random seeds.

    Args:
        base_config: Base configuration
        seeds: List of random seeds
        widget_dict: Dashboard widgets

    Returns:
        results_list: Results for each seed
    """

    results_list = []

    for i, seed in enumerate(seeds):
        update_status(widget_dict, f"\n{'=' * 70}")
        update_status(widget_dict, f"SEED {i + 1}/{len(seeds)}: {seed}")
        update_status(widget_dict, f"{'=' * 70}\n")

        # Copy config and update seed
        config = base_config.copy()
        config['seed'] = seed

        result = run_single_experiment(config, widget_dict)
        results_list.append(result)

    return results_list