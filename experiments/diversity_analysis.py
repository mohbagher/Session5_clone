"""
Diversity analysis for probe banks.

Computes diversity metrics:
- Cosine similarity (for continuous probes)
- Hamming distance (for binary probes)
- Statistical summaries (mean, std, min, max)
"""

import numpy as np
from typing import Dict
from .probe_generators import ProbeBank


def compute_cosine_similarity_matrix(probe_bank: ProbeBank) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix for probe phases.
    
    For continuous probes, we treat phases as vectors and compute
    cosine similarity between them.
    
    Args:
        probe_bank: ProbeBank object
        
    Returns:
        Similarity matrix of shape (K, K)
    """
    K = probe_bank.K
    N = probe_bank.N
    phases = probe_bank.phases
    
    # Convert to complex representation: exp(j*θ)
    complex_vectors = np.exp(1j * phases)  # Shape: (K, N)
    
    # Compute all pairwise inner products using matrix multiplication
    # Inner product: v1^H * v2 (conjugate transpose of v1 times v2)
    inner_products = np.abs(complex_vectors @ complex_vectors.conj().T)
    
    # Normalize by N (since all vectors have the same magnitude sqrt(N))
    similarity_matrix = inner_products / N
    
    return similarity_matrix


def compute_hamming_distance_matrix(probe_bank: ProbeBank) -> np.ndarray:
    """
    Compute pairwise Hamming distance matrix for binary/2-bit probes.
    
    Hamming distance counts the number of positions where probes differ.
    For binary probes, we count phase differences.
    
    Args:
        probe_bank: ProbeBank object
        
    Returns:
        Distance matrix of shape (K, K), normalized to [0, 1]
    """
    K = probe_bank.K
    N = probe_bank.N
    phases = probe_bank.phases
    
    # Tolerance for comparing floating point phases
    tol = 1e-6
    
    # Vectorized computation
    # Expand phases to compute all pairwise differences
    # phases[i, :] - phases[j, :] for all i, j
    phases_i = phases[:, np.newaxis, :]  # Shape: (K, 1, N)
    phases_j = phases[np.newaxis, :, :]  # Shape: (1, K, N)
    
    # Compute absolute differences
    diff = np.abs(phases_i - phases_j)  # Shape: (K, K, N)
    
    # Account for wrap-around (2π = 0)
    diff = np.minimum(diff, 2*np.pi - diff)
    
    # Count positions where difference > tolerance
    different = (diff > tol).sum(axis=2)  # Shape: (K, K)
    
    # Normalize by N
    distance_matrix = different / N
    
    return distance_matrix


def compute_diversity_metrics(probe_bank: ProbeBank) -> Dict[str, float]:
    """
    Compute diversity metrics for a probe bank.
    
    Returns mean, std, min, max of pairwise diversity measures.
    Uses cosine similarity for continuous probes, Hamming distance for others.
    
    Args:
        probe_bank: ProbeBank object
        
    Returns:
        Dictionary with diversity statistics
    """
    if probe_bank.probe_type == "continuous":
        # Use cosine similarity
        similarity_matrix = compute_cosine_similarity_matrix(probe_bank)
        
        # Extract upper triangle (excluding diagonal)
        K = probe_bank.K
        mask = np.triu(np.ones((K, K), dtype=bool), k=1)
        pairwise_similarities = similarity_matrix[mask]
        
        return {
            'metric_type': 'cosine_similarity',
            'mean': float(np.mean(pairwise_similarities)),
            'std': float(np.std(pairwise_similarities)),
            'min': float(np.min(pairwise_similarities)),
            'max': float(np.max(pairwise_similarities)),
            'median': float(np.median(pairwise_similarities))
        }
    else:
        # Use Hamming distance for binary/2bit/hadamard
        distance_matrix = compute_hamming_distance_matrix(probe_bank)
        
        # Extract upper triangle (excluding diagonal)
        K = probe_bank.K
        mask = np.triu(np.ones((K, K), dtype=bool), k=1)
        pairwise_distances = distance_matrix[mask]
        
        return {
            'metric_type': 'hamming_distance',
            'mean': float(np.mean(pairwise_distances)),
            'std': float(np.std(pairwise_distances)),
            'min': float(np.min(pairwise_distances)),
            'max': float(np.max(pairwise_distances)),
            'median': float(np.median(pairwise_distances))
        }
