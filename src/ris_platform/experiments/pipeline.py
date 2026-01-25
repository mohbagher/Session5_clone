"""
Pipeline Logic (Ground Truth Preserved)
=======================================
Generates data and preserves full power vectors for evaluation.
"""
import numpy as np
import torch
from torch.utils.data import TensorDataset
from typing import Tuple, Dict, Any

def generate_channels(backend, config) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Wraps backend call with config extraction."""
    N = int(config.get('N', 32))
    K = int(config.get('K', 64))

    total_samples = int(config.get('n_train', 1000)) + \
                    int(config.get('n_val', 200)) + \
                    int(config.get('n_test', 200))

    seed = int(config.get('seed', 42))
    return backend.generate_channels(N=N, K=K, num_samples=total_samples, seed=seed)

def compute_physics_features(physics, probing, h_all, g_all, config):
    """Runs physics simulation and prepares datasets."""
    if h_all is None or g_all is None:
        raise ValueError("Cannot compute features: Channel matrices are None.")

    N = int(config.get('N', 32))
    K = int(config.get('K', 64))
    M = int(config.get('M', 8))

    num_samples = h_all.shape[1]

    # 1. Probing Strategy
    observed_indices = probing.select_probes(K, M)
    mask_vec = np.zeros(K, dtype=np.float32)
    mask_vec[observed_indices] = 1.0

    # 2. Codebook Generation
    rng = np.random.RandomState(config.get('seed', 42))
    probe_phases = rng.uniform(0, 2*np.pi, (K, N))

    # Containers
    input_vectors = []
    labels = []

    # NEW: Store Ground Truth for Evaluation
    powers_full_list = []
    optimal_powers_list = []

    # 3. Simulation Loop
    for i in range(num_samples):
        # Physics Engine: Compute power for ALL K probes
        powers = physics.compute_received_power(h_all[:, i], g_all[:, i], probe_phases)

        # Ground Truth
        best_idx = np.argmax(powers)
        optimal_power = powers[best_idx]

        # Store for Evaluator
        powers_full_list.append(powers)
        optimal_powers_list.append(optimal_power)

        # Sensing (Input Feature)
        sensed_powers = np.zeros(K)
        sensed_powers[observed_indices] = powers[observed_indices]

        # Normalize
        p_max = np.max(sensed_powers)
        if p_max > 1e-9:
            sensed_powers /= p_max

        # Feature Vector: [Powers, Mask]
        feat = np.concatenate([sensed_powers, mask_vec])

        input_vectors.append(feat)
        labels.append(best_idx)

    # Convert to Tensors
    X = torch.tensor(np.array(input_vectors), dtype=torch.float32)
    y = torch.tensor(np.array(labels), dtype=torch.long)

    # Ground Truth Arrays (for the Test Set)
    powers_full_arr = np.array(powers_full_list)
    optimal_powers_arr = np.array(optimal_powers_list)

    # Splits
    n_train = int(config.get('n_train', 1000))
    n_val = int(config.get('n_val', 200))
    n_test = int(config.get('n_test', 200)) # Explicitly use config size

    # Safety clamp
    n_test = min(n_test, num_samples - n_train - n_val)

    train_ds = TensorDataset(X[:n_train], y[:n_train])
    val_ds = TensorDataset(X[n_train:n_train+n_val], y[n_train:n_train+n_val])

    # Prepare Test Data Dict (Crucial for Evaluator)
    test_start = n_train + n_val
    test_end = test_start + n_test

    # Create Test Loader
    test_dataset = TensorDataset(X[test_start:test_end], y[test_start:test_end])
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size', 32), shuffle=False)

    test_data = {
        'loader': test_loader,
        'powers_full': powers_full_arr[test_start:test_end],
        'labels': np.array(labels)[test_start:test_end],
        'observed_indices': np.tile(observed_indices, (n_test, 1)), # Replicate for all samples
        'optimal_powers': optimal_powers_arr[test_start:test_end]
    }

    return train_ds, val_ds, test_data