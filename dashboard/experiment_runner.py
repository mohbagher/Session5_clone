"""
Experiment Runner - Phase 2: Physics-First Architecture
========================================================
Refactored with composable physics components and factory pattern.
Maintains backward compatibility with existing dashboard.
"""

import numpy as np
import torch
import logging
from typing import Dict, Any, Optional, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass

# Phase 2: Import new physics architecture components
from src.ris_platform.core.interfaces import PhysicsModel, ChannelBackend, ProbingStrategy
from src.ris_platform.physics.models import IdealPhysicsModel, RealisticPhysicsModel
from src.ris_platform.physics.components.unit_cell import IdealUnitCell, VaractorUnitCell
from src.ris_platform.physics.components.coupling import NoCoupling, GeometricCoupling
from src.ris_platform.physics.components.wavefront import PlanarWavefront, SphericalWavefront
from src.ris_platform.physics.components.aging import JakesAging
from src.ris_platform.backend.matlab import MATLABBackend
from src.ris_platform.backend.python_synthetic import PythonSyntheticBackend
from src.ris_platform.probing.structured import RandomProbing, SobolProbing, HadamardProbing

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    config: Dict
    model_state: Any
    training_history: Any
    evaluation: Any
    execution_time: float
    metadata: Dict


# ============================================================================
# PHASE 2: FACTORY FUNCTIONS FOR COMPOSABLE ARCHITECTURE
# ============================================================================

def create_physics_model(config: Dict) -> PhysicsModel:
    """
    Create physics model from config using factory pattern.
    
    Translates dashboard config dict into instantiated physics components,
    maintaining backward compatibility while enabling new composable architecture.
    
    Args:
        config: Experiment configuration dictionary
        
    Returns:
        PhysicsModel instance (Ideal or Realistic)
    """
    realism_profile = config.get('realism_profile', 'ideal')
    
    if realism_profile == 'ideal':
        # Baseline ideal model
        return IdealPhysicsModel()
    else:
        # Realistic model with composable components
        
        # Unit cell: Check for varactor parameters
        if config.get('varactor_coupling_strength', 0.0) > 0:
            unit_cell = VaractorUnitCell(
                coupling_strength=config.get('varactor_coupling_strength', 0.3),
                thermal_drift_coeff=config.get('thermal_drift_coeff', 0.02),
                reference_temp=config.get('reference_temp', 25.0)
            )
        else:
            unit_cell = IdealUnitCell()
        
        # Coupling: Check for coupling strength
        if config.get('coupling_strength', 0.0) > 0:
            coupling = GeometricCoupling(
                coupling_strength=config.get('coupling_strength', 1.0),
                wavelength=config.get('wavelength', 0.125),
                distance_cutoff=config.get('distance_cutoff', None)
            )
        else:
            coupling = NoCoupling()
        
        # Wavefront: Check for near-field flag
        if config.get('enable_near_field', False):
            wavefront = SphericalWavefront(
                wavelength=config.get('wavelength', 0.125),
                include_path_loss=config.get('include_path_loss', True)
            )
        else:
            wavefront = PlanarWavefront()
        
        # Aging: Check for Doppler/mobility
        if config.get('enable_aging', False):
            aging = JakesAging(
                doppler_hz=config.get('doppler_hz', 10.0)
            )
        else:
            aging = None
        
        return RealisticPhysicsModel(
            unit_cell=unit_cell,
            coupling=coupling,
            wavefront=wavefront,
            aging=aging
        )


def create_backend(config: Dict) -> ChannelBackend:
    """
    Create channel backend from config.
    
    Args:
        config: Experiment configuration dictionary
        
    Returns:
        ChannelBackend instance
    """
    backend_name = config.get('physics_backend', 'python')
    
    if backend_name == 'matlab':
        return MATLABBackend(
            scenario=config.get('matlab_scenario', 'rayleigh_basic'),
            auto_fallback=True
        )
    else:
        return PythonSyntheticBackend(
            sigma_h_sq=config.get('sigma_h_sq', 1.0),
            sigma_g_sq=config.get('sigma_g_sq', 1.0)
        )


def create_probing_strategy(config: Dict) -> Optional[ProbingStrategy]:
    """
    Create probing strategy from config.
    
    Args:
        config: Experiment configuration dictionary
        
    Returns:
        ProbingStrategy instance or None
    """
    strategy_name = config.get('probing_strategy', 'random')
    seed = config.get('seed', 42)
    
    if strategy_name == 'sobol':
        return SobolProbing(seed=seed)
    elif strategy_name == 'hadamard':
        return HadamardProbing(seed=seed)
    else:
        return RandomProbing(seed=seed)


# ============================================================================
# ORIGINAL EXPERIMENT RUNNER (PRESERVED SIGNATURE)
# ============================================================================

def run_single_experiment(
    config: Dict,
    widget_dict: Optional[Dict] = None,
    progress_callback: Optional[Callable] = None,
    initial_weights: Optional[Dict] = None,
    verbose: bool = True
) -> ExperimentResult:
    """Run single experiment with full compatibility."""
    start_time = datetime.now()

    if verbose:
        logger.info("=" * 70)
        logger.info("STARTING EXPERIMENT")
        logger.info("=" * 70)

    results = {
        'config': config.copy(),
        'metadata': {
            'start_time': start_time.isoformat(),
            'backend': config.get('physics_backend', 'python'),
        }
    }

    try:
        backend = config.get('physics_backend', 'python')
        update_status(widget_dict, f"Initializing {backend.upper()} backend...", verbose)

        total_samples = config.get('n_train', 50000) + config.get('n_val', 5000) + config.get('n_test', 5000)

        if backend == 'matlab':
            h, g, channel_metadata = generate_channels_matlab(
                config=config,
                total_samples=total_samples,
                widget_dict=widget_dict,
                verbose=verbose
            )
            results['metadata']['channel_generation'] = channel_metadata
        else:
            h, g, channel_metadata = generate_channels_python(
                config=config,
                total_samples=total_samples,
                widget_dict=widget_dict,
                verbose=verbose
            )
            results['metadata']['channel_generation'] = channel_metadata

        if config.get('use_custom_impairments', False) or config.get('realism_profile') != 'ideal':
            h, g = apply_impairments(h, g, config, widget_dict, verbose)
            results['metadata']['impairments_applied'] = True
        else:
            results['metadata']['impairments_applied'] = False

        update_status(widget_dict, "Generating probe bank...", verbose)
        probe_bank = generate_probe_bank(config)

        update_status(widget_dict, "Generating training data...", verbose)
        train_data_dict = generate_training_data(h, g, probe_bank, config)

        update_status(widget_dict, "Training model...", verbose)
        model, training_history = train_model(
            train_data_dict,  # Pass the full dict with train/val/test
            config,
            widget_dict,
            progress_callback=progress_callback,
            initial_weights=initial_weights,
            verbose=verbose
        )

        update_status(widget_dict, "Evaluating performance...", verbose)
        eval_results = evaluate_model(model, train_data_dict['test'], config)

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        result = ExperimentResult(
            config=config,
            model_state=model.state_dict() if hasattr(model, 'state_dict') else None,
            training_history=training_history,
            evaluation=eval_results,
            execution_time=execution_time,
            metadata=results['metadata']
        )

        result.metadata['end_time'] = end_time.isoformat()
        result.metadata['success'] = True

        update_status(widget_dict, "Experiment completed successfully!", verbose)
        return result

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        result = ExperimentResult(
            config=config,
            model_state=None,
            training_history=None,
            evaluation=None,
            execution_time=execution_time,
            metadata=results.get('metadata', {})
        )

        result.metadata['success'] = False
        result.metadata['error'] = str(e)
        result.metadata['end_time'] = end_time.isoformat()

        update_status(widget_dict, f"Experiment failed: {e}", verbose)
        return result


def generate_channels_python(
        config: Dict,
        total_samples: int,
        widget_dict: Optional[Dict] = None,
        verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Generate channels using Python backend."""
    from physics import create_source_from_name

    N = config['N']
    sigma_h_sq = config.get('sigma_h_sq', 1.0)
    sigma_g_sq = config.get('sigma_g_sq', 1.0)
    seed = config.get('seed', 42)
    channel_source_name = config.get('channel_source', 'python_synthetic')

    update_status(widget_dict, f"Generating {total_samples} channels with Python...", verbose)

    source = create_source_from_name(channel_source_name)
    rng = np.random.RandomState(seed)

    h_all = []
    g_all = []

    for i in range(total_samples):
        h_single, g_single, _ = source.generate_channel(
            N=N,
            sigma_h_sq=sigma_h_sq,
            sigma_g_sq=sigma_g_sq,
            rng=rng
        )
        h_all.append(h_single)
        g_all.append(g_single)

    h = np.column_stack(h_all)
    g = np.column_stack(g_all)

    metadata = {
        'backend_name': 'python',
        'source': channel_source_name,
        'num_samples': total_samples,
        'N': N,
        'seed': seed
    }

    update_status(widget_dict, f"Python channels generated", verbose)
    return h, g, metadata


def generate_channels_matlab(
        config: Dict,
        total_samples: int,
        widget_dict: Optional[Dict] = None,
        verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Generate channels using MATLAB backend."""
    try:
        from physics.matlab_backend.matlab_source import MATLABEngineSource

        N = config['N']
        matlab_scenario = config.get('matlab_scenario', 'rayleigh_basic')

        update_status(widget_dict, f"Starting MATLAB Engine...", verbose)

        source = MATLABEngineSource(scenario=matlab_scenario)
        h, g, metadata = source.generate_channels(
            N=N,
            K=config['K'],
            num_samples=total_samples,
            seed=config.get('seed', 42)
        )

        update_status(widget_dict, f"MATLAB channels generated", verbose)
        return h, g, metadata

    except Exception as e:
        logger.error(f"MATLAB failed: {e}")
        update_status(widget_dict, f"MATLAB failed, using Python", verbose)
        return generate_channels_python(config, total_samples, widget_dict, verbose)


def apply_impairments(
        h: np.ndarray,
        g: np.ndarray,
        config: Dict,
        widget_dict: Optional[Dict] = None,
        verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply impairments to channels."""
    from physics import create_pipeline_from_profile

    realism_profile = config.get('realism_profile', 'ideal')
    if realism_profile == 'ideal':
        return h, g

    update_status(widget_dict, f"Applying impairments...", verbose)

    pipeline = create_pipeline_from_profile(realism_profile)
    num_samples = h.shape[1]
    h_impaired = np.zeros_like(h)
    g_impaired = np.zeros_like(g)

    rng = np.random.RandomState(config.get('seed', 42))

    for i in range(num_samples):
        h_impaired[:, i], g_impaired[:, i], _ = pipeline.apply_channel_impairments(
            h[:, i], g[:, i], rng
        )

    return h_impaired, g_impaired


def generate_probe_bank(config: Dict) -> Any:
    """Generate probe bank."""
    from data.probe_generators import get_probe_bank

    return get_probe_bank(
        probe_type=config.get('probe_type', 'continuous'),
        K=config['K'],
        N=config['N'],
        seed=config.get('seed', 42)
    )


def generate_training_data(
        h: np.ndarray,
        g: np.ndarray,
        probe_bank: Any,
        config: Dict
) -> Dict:
    """Generate training data - returns dict with train/val/test keys."""
    from data.data_generation import generate_limited_probing_dataset

    N = config['N']
    K = config['K']
    M = config['M']
    P_tx = config.get('P_tx', 1.0)

    n_train = config.get('n_train', 50000)
    n_val = config.get('n_val', 5000)
    n_test = config.get('n_test', 5000)

    # Split channels - CONVERT TO CORRECT FORMAT
    # h and g come in as (N, total_samples)
    # Need to convert to (samples, N) for matlab_data format
    h_transposed = h.T  # Now (total_samples, N)
    g_transposed = g.T

    h_train = h_transposed[:n_train]
    g_train = g_transposed[:n_train]
    h_val = h_transposed[n_train:n_train + n_val]
    g_val = g_transposed[n_train:n_train + n_val]
    h_test = h_transposed[n_train + n_val:]
    g_test = g_transposed[n_train + n_val:]

    # System config as dict
    system_config_dict = {
        'N': N,
        'K': K,
        'M': M,
        'P_tx': P_tx,
        'sigma_h_sq': config.get('sigma_h_sq', 1.0),
        'sigma_g_sq': config.get('sigma_g_sq', 1.0),
        'seed': config.get('seed', 42)
    }

    # CRITICAL FIX: Pass the channels!
    normalization_method = config.get('normalization_method', 'max_global')

    # Generate datasets with FIXED normalization AND pre-generated channels
    train_data = generate_limited_probing_dataset(
        probe_bank=probe_bank,
        n_samples=n_train,
        M=M,
        system_config=system_config_dict,
        normalize=True,
        normalization_method=normalization_method,
        matlab_data=(h_train, g_train),  # ← PASS THE CHANNELS!
        seed=config.get('seed', 42)
    )

    val_data = generate_limited_probing_dataset(
        probe_bank=probe_bank,
        n_samples=n_val,
        M=M,
        system_config=system_config_dict,
        normalize=True,
        normalization_method=normalization_method,
        matlab_data=(h_val, g_val),  # ← PASS THE CHANNELS!
        seed=config.get('seed', 42) + 1
    )

    test_data = generate_limited_probing_dataset(
        probe_bank=probe_bank,
        n_samples=n_test,
        M=M,
        system_config=system_config_dict,
        normalize=True,
        normalization_method=normalization_method,
        matlab_data=(h_test, g_test),  # ← PASS THE CHANNELS!
        seed=config.get('seed', 42) + 2
    )

    return {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }


def train_model(
        train_data_dict: Dict,  # Now expects dict with train/val/test
        config: Dict,
        widget_dict: Optional[Dict] = None,
        progress_callback: Optional[Callable] = None,
        initial_weights: Optional[Dict] = None,
        verbose: bool = True
) -> Tuple[Any, Any]:
    """Train the model."""
    from models.model_registry import get_model_class, get_model_architecture
    from data.data_generation import LimitedProbingDataset
    from torch.utils.data import DataLoader

    model_name = config.get('model_preset', 'Baseline_MLP')
    model_class = get_model_class(model_name)

    batch_size = config.get('batch_size', 128)
    learning_rate = config.get('learning_rate', 1e-3)
    n_epochs = config.get('n_epochs', 50)
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    input_size = 2 * config['K']
    output_size = config['K']

    if model_name == 'Custom':
        hidden_sizes = config.get('hidden_sizes', [512, 256, 128])
    else:
        hidden_sizes = get_model_architecture(model_name)

    model = model_class(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size
    )
    model = model.to(device)

    if initial_weights is not None:
        try:
            model.load_state_dict(initial_weights)
            if verbose:
                logger.info("Transfer learning weights loaded")
        except Exception as e:
            logger.warning(f"Could not load weights: {e}")

    # FIX: Access train_data_dict['train'] and ['val']
    train_data = train_data_dict['train']
    val_data = train_data_dict['val']

    train_dataset = LimitedProbingDataset(
        train_data['masked_powers'],
        train_data['masks'],
        train_data['labels'],
        powers_full=train_data.get('powers_full'),
        observed_indices=train_data.get('observed_indices'),
        optimal_powers=train_data.get('optimal_powers')
    )

    val_dataset = LimitedProbingDataset(
        val_data['masked_powers'],
        val_data['masks'],
        val_data['labels']
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = type('History', (), {
        'train_loss': [],
        'val_loss': [],
        'val_eta': [],
        'val_acc': []
    })()

    # EARLY STOPPING
    early_stopping = config.get('early_stopping', True)
    patience = config.get('early_stop_patience', 20)
    min_delta = config.get('early_stopping_min_delta', 1e-4)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct / total if total > 0 else 0.0

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)
        history.val_eta.append(val_acc)

        # EARLY STOPPING CHECK
        if early_stopping:
            # Check if validation loss improved
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                if verbose:
                    logger.info(f"New best model at epoch {epoch + 1}")
            else:
                patience_counter += 1
                if verbose and patience_counter > 0:
                    logger.info(f"No improvement for {patience_counter}/{patience} epochs")

                # Stop if patience exceeded
                if patience_counter >= patience:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch + 1}/{n_epochs}")
                        logger.info(f"   Best val_loss: {best_val_loss:.4f}")

                    # Restore best model
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                        if verbose:
                            logger.info(f"   Restored best model weights")

                    update_status(widget_dict, f"Early stopping (no improvement for {patience} epochs)", verbose)
                    break

        if progress_callback is not None:
            metrics = {'val_loss': val_loss, 'val_eta': val_acc}
            progress_callback(epoch + 1, n_epochs, metrics)

        if verbose and (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{n_epochs}: Loss={val_loss:.4f}, Acc={val_acc:.4f}")

    return model, history


# ============================================================================
# REPLACE THIS FUNCTION in dashboard/experiment_runner.py
# ============================================================================


def evaluate_model(
        model: Any,
        test_data: Dict,
        config: Dict
) -> Any:
    """
    Comprehensive evaluation with rigorous metrics for PhD research.

    Computes:
    - Top-m accuracy (oracle in top-m predictions)
    - Power ratio η for different m values
    - Multiple baseline comparisons
    - Statistical distributions

    Args:
        model: Trained PyTorch model
        test_data: Dict with keys:
            - 'masked_powers': (n_test, K)
            - 'masks': (n_test, K)
            - 'labels': (n_test,) - oracle best probe indices
            - 'powers_full': (n_test, K) - full power vectors
            - 'observed_indices': (n_test, M) - which probes were measured
            - 'optimal_powers': (n_test,) - theoretical optimal power
        config: Configuration dict

    Returns:
        EvaluationResults object with comprehensive metrics
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        logger.info("=" * 70)
        logger.info("Starting comprehensive evaluation...")
        logger.info("=" * 70)

        import torch
        import numpy as np
        from dataclasses import dataclass
        from typing import List

        # Extract configuration with safety checks
        K = config.get('K', 64)
        M = config.get('M', 8)
        device = config.get('device', 'cpu')
        batch_size = config.get('batch_size', 128)
        seed = config.get('seed', 42)

        logger.info(f"Configuration: K={K}, M={M}, device={device}")

        # SAFETY CHECK 1: Validate test_data structure
        required_keys = ['masked_powers', 'masks', 'labels', 'powers_full',
                         'observed_indices', 'optimal_powers']
        missing_keys = [key for key in required_keys if key not in test_data]

        if missing_keys:
            logger.error(f"Missing keys in test_data: {missing_keys}")
            logger.error("Returning placeholder results due to invalid test data")
            return create_placeholder_results(K, M, error=f"Missing keys: {missing_keys}")

        # SAFETY CHECK 2: Validate data shapes
        n_test = len(test_data['labels'])
        logger.info(f"Test dataset size: {n_test} samples")

        if n_test == 0:
            logger.error("Test data is empty!")
            return create_placeholder_results(K, M, error="Empty test dataset")

        # SAFETY CHECK 3: Verify data dimensions
        expected_shapes = {
            'masked_powers': (n_test, K),
            'masks': (n_test, K),
            'labels': (n_test,),
            'powers_full': (n_test, K),
            'observed_indices': (n_test, M),
            'optimal_powers': (n_test,)
        }

        for key, expected_shape in expected_shapes.items():
            actual_shape = test_data[key].shape
            if actual_shape != expected_shape:
                logger.error(f"Shape mismatch for {key}: expected {expected_shape}, got {actual_shape}")
                return create_placeholder_results(K, M, error=f"Shape mismatch: {key}")

        logger.info("✓ Data validation passed")

        # Top-m values to evaluate
        top_m_values = [1, 2, 4, 8, 16] if K >= 16 else [1, 2, 4, 8]
        logger.info(f"Evaluating top-m for m={top_m_values}")

        # =====================================================================
        # STEP 1: Prepare test data
        # =====================================================================
        logger.info("Step 1: Preparing test data...")

        from torch.utils.data import DataLoader, TensorDataset

        test_inputs = torch.cat([
            torch.FloatTensor(test_data['masked_powers']),
            torch.FloatTensor(test_data['masks'])
        ], dim=1)
        test_labels = torch.LongTensor(test_data['labels'])

        test_dataset = TensorDataset(test_inputs, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Extract metadata
        powers_full = test_data['powers_full']  # (n_test, K)
        labels = test_data['labels']  # (n_test,)
        observed_indices = test_data['observed_indices']  # (n_test, M)
        optimal_powers = test_data['optimal_powers']  # (n_test,)

        logger.info(f"✓ Test loader created with {len(test_loader)} batches")

        # =====================================================================
        # STEP 2: Get model predictions
        # =====================================================================
        logger.info("Step 2: Running model inference...")

        model.eval()
        model = model.to(device)

        all_logits = []
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(test_loader):
                inputs = inputs.to(device)
                logits = model(inputs)
                all_logits.append(logits.cpu().numpy())

                # Progress logging for large datasets
                if (batch_idx + 1) % 100 == 0:
                    logger.info(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")

        all_logits = np.concatenate(all_logits, axis=0)  # (n_test, K)
        logger.info(f"✓ Inference complete. Logits shape: {all_logits.shape}")

        # =====================================================================
        # STEP 3: Compute Top-m Accuracy and η
        # =====================================================================
        logger.info("Step 3: Computing top-m accuracy and power ratios...")

        accuracy = {m: 0 for m in top_m_values}
        eta_sum = {m: 0.0 for m in top_m_values}
        eta_distributions = {m: [] for m in top_m_values}

        for i in range(n_test):
            logits = all_logits[i]
            label = labels[i]
            P_best = powers_full[i, label]  # Power of oracle best probe

            # Sort predictions by confidence (descending)
            sorted_indices = np.argsort(logits)[::-1]

            for m in top_m_values:
                top_m_idx = sorted_indices[:m]

                # Top-m accuracy
                if label in top_m_idx:
                    accuracy[m] += 1

                # Top-m power ratio (η)
                P_selected = np.max(powers_full[i, top_m_idx])
                eta_val = P_selected / P_best if P_best > 1e-10 else 0.0
                eta_sum[m] += eta_val
                eta_distributions[m].append(eta_val)

            # Progress logging
            if (i + 1) % 1000 == 0:
                logger.info(f"  Processed {i + 1}/{n_test} samples")

        # Normalize
        for m in top_m_values:
            accuracy[m] /= n_test
            eta_sum[m] /= n_test

        logger.info("✓ Top-m metrics computed")

        # =====================================================================
        # STEP 4: Compute Baselines
        # =====================================================================
        logger.info("Step 4: Computing baseline comparisons...")

        rng = np.random.RandomState(seed + 1000)

        # Baseline 1: Random selection of 1 probe from K
        eta_random_1_sum = 0.0
        for i in range(n_test):
            random_idx = rng.randint(0, K)
            P_random = powers_full[i, random_idx]
            P_best = powers_full[i, labels[i]]
            eta_random_1_sum += P_random / P_best if P_best > 1e-10 else 0.0
        eta_random_1 = eta_random_1_sum / n_test
        logger.info(f"  Baseline 1 (Random-1): η={eta_random_1:.4f}")

        # Baseline 2: Best of random M probes (same sensing budget as ML)
        eta_random_M_sum = 0.0
        for i in range(n_test):
            random_M_idx = rng.choice(K, size=M, replace=False)
            P_random_M = np.max(powers_full[i, random_M_idx])
            P_best = powers_full[i, labels[i]]
            eta_random_M_sum += P_random_M / P_best if P_best > 1e-10 else 0.0
        eta_random_M = eta_random_M_sum / n_test
        logger.info(f"  Baseline 2 (Random-M): η={eta_random_M:.4f}")

        # Baseline 3: Best of actually observed M probes
        eta_best_observed_sum = 0.0
        eta_best_observed_dist = []
        for i in range(n_test):
            obs_idx = observed_indices[i]
            P_best_observed = np.max(powers_full[i, obs_idx])
            P_best = powers_full[i, labels[i]]
            eta_val = P_best_observed / P_best if P_best > 1e-10 else 0.0
            eta_best_observed_sum += eta_val
            eta_best_observed_dist.append(eta_val)
        eta_best_observed = eta_best_observed_sum / n_test
        logger.info(f"  Baseline 3 (Best-Observed): η={eta_best_observed:.4f}")

        # Baseline 4: Oracle (always = 1.0 by definition)
        eta_oracle = 1.0

        # Baseline 5: Best probe vs theoretical optimal
        eta_vs_theoretical_sum = 0.0
        for i in range(n_test):
            P_best = powers_full[i, labels[i]]
            P_optimal = optimal_powers[i]
            eta_vs_theoretical_sum += P_best / P_optimal if P_optimal > 1e-10 else 0.0
        eta_vs_theoretical = eta_vs_theoretical_sum / n_test
        logger.info(f"  Baseline 4 (vs Theoretical): η={eta_vs_theoretical:.4f}")

        logger.info("✓ Baselines computed")

        # =====================================================================
        # STEP 5: Assemble Results
        # =====================================================================
        logger.info("Step 5: Assembling results...")

        @dataclass
        class EvaluationResults:
            """PhD-quality evaluation results."""
            # Required fields (no defaults) first
            accuracy_top1: float
            accuracy_top2: float
            accuracy_top4: float
            accuracy_top8: float

            eta_top1: float
            eta_top2: float
            eta_top4: float
            eta_top8: float

            eta_random_1: float
            eta_random_M: float
            eta_best_observed: float
            eta_oracle: float
            eta_vs_theoretical: float

            eta_top1_distribution: np.ndarray
            eta_best_observed_distribution: np.ndarray

            M: int
            K: int
            n_test: int

            # Fields with defaults MUST come last
            accuracy_top16: float = 0.0
            eta_top16: float = 0.0

            def print_summary(self):
                """Print comprehensive evaluation summary."""
                print("\n" + "=" * 80)
                print("EVALUATION RESULTS - Limited Probing with ML Predictor")
                print(f"Configuration: M={self.M} observed probes, K={self.K} total probes, n_test={self.n_test}")
                print("=" * 80)

                print("\n📊 TOP-M ACCURACY (Oracle best probe in top-m predictions):")
                print(f"   Top-1:  {self.accuracy_top1 * 100:6.2f}%  (Theoretical max: {100 * self.M / self.K:.2f}%)")
                print(f"   Top-2:  {self.accuracy_top2 * 100:6.2f}%")
                print(f"   Top-4:  {self.accuracy_top4 * 100:6.2f}%")
                print(f"   Top-8:  {self.accuracy_top8 * 100:6.2f}%")
                if self.accuracy_top16 > 0:
                    print(f"   Top-16: {self.accuracy_top16 * 100:6.2f}%")

                print("\n⚡ POWER RATIO η = P_selected / P_oracle_best:")
                print(f"   η_top1 (ML):              {self.eta_top1:.4f}")
                print(f"   η_top2 (ML):              {self.eta_top2:.4f}")
                print(f"   η_top4 (ML):              {self.eta_top4:.4f}")
                print(f"   η_top8 (ML):              {self.eta_top8:.4f}")
                if self.eta_top16 > 0:
                    print(f"   η_top16 (ML):             {self.eta_top16:.4f}")

                print("\n📈 BASELINE COMPARISONS:")
                print(f"   η_random_1 (random):      {self.eta_random_1:.4f}  (pick 1 of K randomly)")
                print(f"   η_random_M (random):      {self.eta_random_M:.4f}  (best of M random)")
                print(f"   η_best_observed:          {self.eta_best_observed:.4f}  (best of M observed)")
                print(f"   η_oracle:                 {self.eta_oracle:.4f}  (perfect selection)")
                print(f"   η_vs_theoretical:         {self.eta_vs_theoretical:.4f}  (best probe / optimal)")

                print("\n📉 ML IMPROVEMENT OVER BASELINES:")
                if self.eta_random_1 > 1e-6:
                    gain = (self.eta_top1 / self.eta_random_1 - 1) * 100
                    print(f"   vs Random-1:      {gain:+6.1f}%  ({self.eta_top1 / self.eta_random_1:.2f}x)")
                if self.eta_random_M > 1e-6:
                    gain = (self.eta_top1 / self.eta_random_M - 1) * 100
                    print(f"   vs Random-M:      {gain:+6.1f}%  ({self.eta_top1 / self.eta_random_M:.2f}x)")
                if self.eta_best_observed > 1e-6:
                    gain = (self.eta_top1 / self.eta_best_observed - 1) * 100
                    efficiency = self.eta_top1 / self.eta_best_observed * 100
                    print(f"   vs Best-Observed: {gain:+6.1f}%  ({efficiency:.1f}% efficiency)")

                print("\n📊 STATISTICAL DISTRIBUTION (η_top1):")
                dist = self.eta_top1_distribution
                print(f"   Mean:   {np.mean(dist):.4f}")
                print(f"   Median: {np.median(dist):.4f}")
                print(f"   Std:    {np.std(dist):.4f}")
                print(f"   Min:    {np.min(dist):.4f}")
                print(f"   Max:    {np.max(dist):.4f}")
                print(f"   Q25:    {np.percentile(dist, 25):.4f}")
                print(f"   Q75:    {np.percentile(dist, 75):.4f}")

                print("\n💡 INTERPRETATION:")
                if self.eta_top1 / self.eta_best_observed > 0.9:
                    print("   ✓ Model achieves >90% of best-observed performance")
                    print("   ✓ Near-optimal selection from available measurements")
                elif self.eta_top1 / self.eta_best_observed > 0.7:
                    print("   ⚠ Model achieves 70-90% of best-observed performance")
                    print("   → Room for improvement in pattern learning")
                else:
                    print("   ❌ Model achieves <70% of best-observed performance")
                    print("   → Significant learning issues detected")

                if self.accuracy_top1 / (self.M / self.K) > 0.8:
                    print(
                        f"   ✓ Top-1 accuracy is {self.accuracy_top1 / (self.M / self.K) * 100:.0f}% of theoretical maximum")

                print("=" * 80)

            def to_dict(self) -> Dict:
                """Convert to dictionary for serialization."""
                return {
                    'accuracy_top1': float(self.accuracy_top1),
                    'accuracy_top2': float(self.accuracy_top2),
                    'accuracy_top4': float(self.accuracy_top4),
                    'accuracy_top8': float(self.accuracy_top8),
                    'accuracy_top16': float(self.accuracy_top16),
                    'eta_top1': float(self.eta_top1),
                    'eta_top2': float(self.eta_top2),
                    'eta_top4': float(self.eta_top4),
                    'eta_top8': float(self.eta_top8),
                    'eta_top16': float(self.eta_top16),
                    'eta_random_1': float(self.eta_random_1),
                    'eta_random_M': float(self.eta_random_M),
                    'eta_best_observed': float(self.eta_best_observed),
                    'eta_oracle': float(self.eta_oracle),
                    'eta_vs_theoretical': float(self.eta_vs_theoretical),
                    'M': int(self.M),
                    'K': int(self.K),
                    'n_test': int(self.n_test),
                    'eta_top1_stats': {
                        'mean': float(np.mean(self.eta_top1_distribution)),
                        'std': float(np.std(self.eta_top1_distribution)),
                        'min': float(np.min(self.eta_top1_distribution)),
                        'max': float(np.max(self.eta_top1_distribution)),
                        'median': float(np.median(self.eta_top1_distribution)),
                        'q25': float(np.percentile(self.eta_top1_distribution, 25)),
                        'q75': float(np.percentile(self.eta_top1_distribution, 75))
                    }
                }

        # Create results object
        results = EvaluationResults(
            accuracy_top1=accuracy.get(1, 0.0),
            accuracy_top2=accuracy.get(2, 0.0),
            accuracy_top4=accuracy.get(4, 0.0),
            accuracy_top8=accuracy.get(8, 0.0),
            accuracy_top16=accuracy.get(16, 0.0),
            eta_top1=eta_sum.get(1, 0.0),
            eta_top2=eta_sum.get(2, 0.0),
            eta_top4=eta_sum.get(4, 0.0),
            eta_top8=eta_sum.get(8, 0.0),
            eta_top16=eta_sum.get(16, 0.0),
            eta_random_1=eta_random_1,
            eta_random_M=eta_random_M,
            eta_best_observed=eta_best_observed,
            eta_oracle=eta_oracle,
            eta_vs_theoretical=eta_vs_theoretical,
            eta_top1_distribution=np.array(eta_distributions[1]),
            eta_best_observed_distribution=np.array(eta_best_observed_dist),
            M=M,
            K=K,
            n_test=n_test
        )

        logger.info("✓ Results assembled successfully")
        logger.info("=" * 70)

        # Print summary if verbose
        if config.get('verbose', True):
            results.print_summary()

        return results

    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user")
        raise

    except Exception as e:
        logger.error("=" * 70)
        logger.error(f"EVALUATION FAILED: {e}")
        logger.error("=" * 70)
        import traceback
        traceback.print_exc()
        logger.error("=" * 70)
        logger.error("Returning placeholder results to prevent crash")
        logger.error("=" * 70)

        # Return placeholder results instead of crashing
        return create_placeholder_results(
            config.get('K', 64),
            config.get('M', 8),
            error=str(e)
        )


def create_placeholder_results(K: int, M: int, error: str = "Unknown error"):
    """
    Create placeholder results when evaluation fails.

    This prevents the dashboard from crashing and provides
    diagnostic information about what went wrong.

    Args:
        K: Total number of probes
        M: Number of observed probes
        error: Error message describing what failed

    Returns:
        EvaluationResults object with placeholder values
    """
    from dataclasses import dataclass
    import numpy as np
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f"Creating placeholder results due to: {error}")

    @dataclass
    class EvaluationResults:
        """Placeholder evaluation results."""
        # Required fields
        accuracy_top1: float
        accuracy_top2: float
        accuracy_top4: float
        accuracy_top8: float

        eta_top1: float
        eta_top2: float
        eta_top4: float
        eta_top8: float

        eta_random_1: float
        eta_random_M: float
        eta_best_observed: float
        eta_oracle: float
        eta_vs_theoretical: float

        eta_top1_distribution: np.ndarray
        eta_best_observed_distribution: np.ndarray

        M: int
        K: int
        n_test: int

        # Optional fields
        accuracy_top16: float = 0.0
        eta_top16: float = 0.0
        error_message: str = ""

        def print_summary(self):
            """Print warning about placeholder results."""
            print("\n" + "=" * 80)
            print("⚠️  PLACEHOLDER EVALUATION RESULTS")
            print("=" * 80)
            print()
            print("Evaluation failed with error:")
            print(f"  {self.error_message}")
            print()
            print("These are NOT real results - they are placeholders to prevent crashes.")
            print()
            print("Action required:")
            print("  1. Check the error message above")
            print("  2. Review the training logs")
            print("  3. Verify your configuration")
            print("  4. Re-run the experiment")
            print("=" * 80)

        def to_dict(self):
            """Convert to dictionary."""
            return {
                'accuracy_top1': self.accuracy_top1,
                'eta_top1': self.eta_top1,
                'M': self.M,
                'K': self.K,
                'n_test': self.n_test,
                'error': self.error_message,
                'is_placeholder': True
            }

    # Create placeholder with reasonable dummy values
    return EvaluationResults(
        accuracy_top1=M / K,  # Theoretical random chance
        accuracy_top2=M / K,
        accuracy_top4=M / K,
        accuracy_top8=M / K,
        eta_top1=0.5,
        eta_top2=0.5,
        eta_top4=0.5,
        eta_top8=0.5,
        eta_random_1=0.1,
        eta_random_M=0.3,
        eta_best_observed=0.4,
        eta_oracle=1.0,
        eta_vs_theoretical=0.5,
        eta_top1_distribution=np.array([0.5]),
        eta_best_observed_distribution=np.array([0.4]),
        M=M,
        K=K,
        n_test=0,
        error_message=error
    )

def update_status(
        widget_dict: Optional[Dict],
        message: str,
        verbose: bool = True
) -> None:
    """Update status display."""
    if widget_dict is not None and 'status_output' in widget_dict:
        with widget_dict['status_output']:
            print(message)

    if verbose:
        logger.info(message)


def run_experiment_stack(stack: list, widget_dict: Dict) -> list:
    """Run stack of experiments."""
    results_list = []

    for i, config in enumerate(stack):
        update_status(widget_dict, f"\n{'=' * 70}")
        update_status(widget_dict, f"EXPERIMENT {i + 1}/{len(stack)}")
        update_status(widget_dict, f"{'=' * 70}\n")

        result = run_single_experiment(config, widget_dict)
        results_list.append(result)

    return results_list


def run_multi_model_comparison(configs: list, widget_dict: Dict) -> list:
    """Run multi-model comparison."""
    results_list = []
    for config in configs:
        result = run_single_experiment(config, widget_dict)
        results_list.append(result)
    return results_list


def run_multi_seed_experiment(base_config: Dict, seeds: list, widget_dict: Dict) -> list:
    """Run multi-seed experiment."""
    results_list = []
    for seed in seeds:
        config = base_config.copy()
        config['seed'] = seed
        result = run_single_experiment(config, widget_dict)
        results_list.append(result)
    return results_list