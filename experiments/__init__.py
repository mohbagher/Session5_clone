"""
Experiments package for RIS probe-based ML research.
"""
from .probe_generators import (
    ProbeBank,
    generate_probe_bank_continuous,
    generate_probe_bank_binary,
    generate_probe_bank_2bit,
    generate_probe_bank_hadamard,
    get_probe_bank
)
from .diversity_analysis import (
    compute_cosine_similarity_matrix,
    compute_hamming_distance_matrix,
    compute_diversity_metrics
)
