"""
Data generation for RIS probe-based control with limited probing.

Key change from baseline:
- Instead of providing all K probe powers as input, we only observe M << K probes
- The model must predict the best probe among all K using only M observations
"""

import numpy as np
import math
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import qmc
from scipy.linalg import hadamard

from config import Config, SystemConfig, DataConfig
from experiments.probe_generators import ProbeBank


def generate_channel_realization(N: int,
                                 sigma_h_sq: float = 1.0,
                                 sigma_g_sq: float = 1.0,
                                 rng: Optional[np.random.RandomState] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate one channel realization (h, g) for Rayleigh fading.
    """
    if rng is None: 
        rng = np.random.RandomState()
    
    h = np.sqrt(sigma_h_sq / 2) * (rng.randn(N) + 1j * rng.randn(N))
    g = np.sqrt(sigma_g_sq / 2) * (rng.randn(N) + 1j * rng.randn(N))
    
    return h, g


def compute_probe_powers(h: np.ndarray,
                         g: np.ndarray,
                         probe_bank: ProbeBank,
                         P_tx: float = 1.0) -> np.ndarray:
    """
    Compute received power for all probes given a channel realization.
    """
    c = h * g
    reflection_coeffs = probe_bank.get_reflection_coefficients()
    h_eff = np.dot(reflection_coeffs, c)
    powers = P_tx * np.abs(h_eff) ** 2
    return powers


def compute_optimal_power(h: np.ndarray,
                          g: np.ndarray,
                          P_tx: float = 1.0) -> float:
    """
    Compute the theoretical optimal received power with perfect phase alignment.
    """
    c = h * g
    h_eff_opt = np.sum(np.abs(c))
    P_opt = P_tx * h_eff_opt ** 2
    return P_opt


def select_probing_subset(K: int, M: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Randomly select M probe indices from K total probes.
    
    Args:
        K: Total number of probes
        M:  Number of probes to select (sensing budget)
        rng: Random number generator
        
    Returns: 
        Array of M unique probe indices (sorted for consistency)
    """
    indices = rng.choice(K, size=M, replace=False)
    return np.sort(indices)


def create_masked_input(powers_full: np.ndarray,
                        observed_indices: np.ndarray,
                        K: int,
                        normalize:  bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create masked vector input from observed probe powers.
    
    This is the key function for the limited probing setup: 
    - We create a vector of size K with zeros for unobserved probes
    - We create a binary mask indicating which probes were observed
    - The model input is the concatenation [masked_powers, mask] of size 2K
    
    Args:
        powers_full: Full power vector of size K (only used at observed indices)
        observed_indices:  Indices of observed probes (size M)
        K: Total number of probes
        normalize:  Whether to normalize observed powers
        
    Returns: 
        masked_powers: Power vector with zeros at unobserved positions (size K)
        mask: Binary mask indicating observed positions (size K)
    """
    # Initialize with zeros
    masked_powers = np.zeros(K, dtype=np.float32)
    mask = np.zeros(K, dtype=np.float32)
    
    # Fill in observed values
    observed_powers = powers_full[observed_indices]
    
    # Normalize observed powers (important for training stability)
    if normalize and len(observed_powers) > 0:
        mean_power = np.mean(observed_powers)
        if mean_power > 1e-10: 
            observed_powers = observed_powers / mean_power
    
    masked_powers[observed_indices] = observed_powers
    mask[observed_indices] = 1.0
    
    return masked_powers, mask


def generate_limited_probing_dataset(
    probe_bank: ProbeBank,
    n_samples: int,
    M: int,
    system_config: SystemConfig,
    normalize: bool = True,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generate dataset for limited probing scenario.
    
    For each sample:
    1.Generate channel realization
    2.Compute full power vector (K powers) - for evaluation only
    3.Select random subset of M probes to observe
    4.Create masked input vector
    5.Label is the oracle best probe (argmax of full powers)
    
    Args:
        probe_bank: Fixed ProbeBank object
        n_samples: Number of samples to generate
        M:  Sensing budget (number of probes to measure)
        system_config: System configuration
        normalize: Whether to normalize observed powers
        seed: Random seed
        
    Returns:
        Dictionary containing:
            - 'masked_powers':  Masked power vectors, shape (n_samples, K)
            - 'masks': Binary masks, shape (n_samples, K)
            - 'observed_indices': Indices of observed probes, shape (n_samples, M)
            - 'powers_full': Full power vectors for evaluation, shape (n_samples, K)
            - 'labels': Oracle labels (best probe index), shape (n_samples,)
            - 'optimal_powers': Theoretical optimal powers, shape (n_samples,)
    """
    rng = np.random.RandomState(seed)
    
    K = probe_bank.K
    N = probe_bank.N
    
    # Allocate arrays
    masked_powers = np.zeros((n_samples, K), dtype=np.float32)
    masks = np.zeros((n_samples, K), dtype=np.float32)
    observed_indices = np.zeros((n_samples, M), dtype=np.int64)
    powers_full = np.zeros((n_samples, K), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int64)
    optimal_powers = np.zeros(n_samples, dtype=np.float32)
    
    for i in range(n_samples):
        # Generate channel realization
        h, g = generate_channel_realization(
            N,
            sigma_h_sq=system_config.sigma_h_sq,
            sigma_g_sq=system_config.sigma_g_sq,
            rng=rng
        )
        
        # Compute full power vector (for evaluation)
        p_full = compute_probe_powers(h, g, probe_bank, P_tx=system_config.P_tx)
        powers_full[i] = p_full
        
        # Oracle label:  best probe among all K
        labels[i] = np.argmax(p_full)
        
        # Theoretical optimal power
        optimal_powers[i] = compute_optimal_power(h, g, P_tx=system_config.P_tx)
        
        # Select random subset of M probes to observe
        obs_idx = select_probing_subset(K, M, rng)
        observed_indices[i] = obs_idx
        
        # Create masked input
        mp, m = create_masked_input(p_full, obs_idx, K, normalize=normalize)
        masked_powers[i] = mp
        masks[i] = m
    
    return {
        'masked_powers': masked_powers,
        'masks': masks,
        'observed_indices': observed_indices,
        'powers_full':  powers_full,
        'labels': labels,
        'optimal_powers': optimal_powers
    }


class LimitedProbingDataset(Dataset):
    """
    PyTorch Dataset for limited probing RIS control.
    
    Input to model:  concatenation of [masked_powers, mask] (size 2K)
    Label: oracle best probe index
    """
    
    def __init__(self,
                 masked_powers: np.ndarray,
                 masks: np.ndarray,
                 labels: np.ndarray,
                 powers_full: Optional[np.ndarray] = None,
                 observed_indices:  Optional[np.ndarray] = None):
        """
        Args:
            masked_powers:  Masked power vectors, shape (n_samples, K)
            masks: Binary masks, shape (n_samples, K)
            labels: Oracle labels, shape (n_samples,)
            powers_full: Full powers for evaluation, shape (n_samples, K)
            observed_indices: Observed probe indices, shape (n_samples, M)
        """
        # Concatenate masked_powers and mask as model input
        self.inputs = torch.FloatTensor(np.concatenate([masked_powers, masks], axis=1))
        self.labels = torch.LongTensor(labels)
        
        # Keep numpy arrays for evaluation
        self.powers_full = powers_full
        self.observed_indices = observed_indices
        self.masks = masks
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.labels[idx]


def create_dataloaders(config: Config,
                       probe_bank: ProbeBank) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train, validation, and test dataloaders for limited probing.
    
    Args:
        config: Configuration object
        probe_bank: Fixed probe bank
        
    Returns: 
        train_loader, val_loader, test_loader, metadata
    """
    data_config = config.data
    system_config = config.system
    training_config = config.training
    M = system_config.M
    
    print(f"Generating datasets with M={M} observed probes out of K={system_config.K}...")
    
    # Generate datasets with different seeds
    print("  Generating training data...")
    train_data = generate_limited_probing_dataset(
        probe_bank,
        data_config.n_train,
        M,
        system_config,
        normalize=data_config.normalize_input,
        seed=data_config.seed
    )
    
    print("  Generating validation data...")
    val_data = generate_limited_probing_dataset(
        probe_bank,
        data_config.n_val,
        M,
        system_config,
        normalize=data_config.normalize_input,
        seed=data_config.seed + 1
    )
    
    print("  Generating test data...")
    test_data = generate_limited_probing_dataset(
        probe_bank,
        data_config.n_test,
        M,
        system_config,
        normalize=data_config.normalize_input,
        seed=data_config.seed + 2
    )
    
    # Create datasets
    train_dataset = LimitedProbingDataset(
        train_data['masked_powers'],
        train_data['masks'],
        train_data['labels'],
        train_data['powers_full'],
        train_data['observed_indices']
    )
    val_dataset = LimitedProbingDataset(
        val_data['masked_powers'],
        val_data['masks'],
        val_data['labels'],
        val_data['powers_full'],
        val_data['observed_indices']
    )
    test_dataset = LimitedProbingDataset(
        test_data['masked_powers'],
        test_data['masks'],
        test_data['labels'],
        test_data['powers_full'],
        test_data['observed_indices']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if training_config.device == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Metadata for evaluation
    metadata = {
        'train_powers_full': train_data['powers_full'],
        'val_powers_full': val_data['powers_full'],
        'test_powers_full': test_data['powers_full'],
        'train_observed_indices': train_data['observed_indices'],
        'val_observed_indices': val_data['observed_indices'],
        'test_observed_indices':  test_data['observed_indices'],
        'train_optimal_powers': train_data['optimal_powers'],
        'val_optimal_powers': val_data['optimal_powers'],
        'test_optimal_powers': test_data['optimal_powers'],
        'train_labels': train_data['labels'],
        'val_labels': val_data['labels'],
        'test_labels': test_data['labels']
    }
    
    return train_loader, val_loader, test_loader, metadata
