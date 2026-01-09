"""
Evaluation metrics for RIS probe-based control with limited probing.

Key metrics:
- eta_top_m: Power ratio of best among top-m predictions vs oracle best
- top_m_accuracy: Fraction where oracle is in top-m predictions
- Baselines: random selection, best-of-observed, oracle
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from dataclasses import dataclass

from config import Config


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    # Top-m accuracy (oracle in top-m predictions)
    accuracy_top1: float
    accuracy_top2: float
    accuracy_top4: float
    accuracy_top8: float
    
    # Top-m power ratio (eta)
    eta_top1: float
    eta_top2: float
    eta_top4: float
    eta_top8: float
    
    # Baseline comparisons
    eta_random_1: float          # Random pick 1 from K
    eta_random_M: float          # Best of random M (same sensing budget)
    eta_best_observed: float     # Best among actually observed M probes
    eta_oracle: float            # Oracle best probe (should be 1.0)
    eta_vs_theoretical: float    # Best probe / theoretical optimal
    
    # Distributions for plotting
    eta_top1_distribution: np.ndarray
    eta_best_observed_distribution: np.ndarray
    
    # Additional info
    M: int  # Sensing budget used
    K: int  # Total probes
    
    def to_dict(self) -> Dict:
        return {
            'accuracy_top1': self.accuracy_top1,
            'accuracy_top2': self.accuracy_top2,
            'accuracy_top4': self.accuracy_top4,
            'accuracy_top8': self.accuracy_top8,
            'eta_top1':  self.eta_top1,
            'eta_top2': self.eta_top2,
            'eta_top4': self.eta_top4,
            'eta_top8': self.eta_top8,
            'eta_random_1': self.eta_random_1,
            'eta_random_M': self.eta_random_M,
            'eta_best_observed': self.eta_best_observed,
            'eta_oracle': self.eta_oracle,
            'eta_vs_theoretical': self.eta_vs_theoretical,
            'M': self.M,
            'K': self.K
        }
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS (Limited Probing)")
        print(f"Sensing Budget: M={self.M} probes observed out of K={self.K} total")
        print("=" * 70)
        
        print("\n📊 Top-m Accuracy (oracle best probe in top-m predictions):")
        print(f"   Top-1: {self.accuracy_top1:.4f} ({self.accuracy_top1*100:.2f}%)")
        print(f"   Top-2: {self.accuracy_top2:.4f} ({self.accuracy_top2*100:.2f}%)")
        print(f"   Top-4: {self.accuracy_top4:.4f} ({self.accuracy_top4*100:.2f}%)")
        print(f"   Top-8: {self.accuracy_top8:.4f} ({self.accuracy_top8*100:.2f}%)")
        
        print("\n⚡ Power Ratio η (P_selected / P_best_probe):")
        print(f"   η_top1 (ML model):        {self.eta_top1:.4f}")
        print(f"   η_top2 (ML model):        {self.eta_top2:.4f}")
        print(f"   η_top4 (ML model):        {self.eta_top4:.4f}")
        print(f"   η_top8 (ML model):        {self.eta_top8:.4f}")
        
        print("\n📈 Baselines:")
        print(f"   η_random_1 (random 1 of K):       {self.eta_random_1:.4f}")
        print(f"   η_random_M (best of random M):    {self.eta_random_M:.4f}")
        print(f"   η_best_observed (best of obs M):  {self.eta_best_observed:.4f}")
        print(f"   η_oracle (best probe):            {self.eta_oracle:.4f}")
        print(f"   η_vs_theoretical (probe/optimal): {self.eta_vs_theoretical:.4f}")
        
        print("\n📉 ML Model Improvement over Baselines:")
        if self.eta_random_1 > 0:
            improvement_vs_random = (self.eta_top1 - self.eta_random_1) / self.eta_random_1 * 100
            print(f"   vs random_1:       +{improvement_vs_random:.1f}%")
        if self.eta_random_M > 0:
            improvement_vs_randomM = (self.eta_top1 - self.eta_random_M) / self.eta_random_M * 100
            print(f"   vs random_M:       +{improvement_vs_randomM:.1f}%")
        if self.eta_best_observed > 0:
            improvement_vs_observed = (self.eta_top1 - self.eta_best_observed) / self.eta_best_observed * 100
            print(f"   vs best_observed:   +{improvement_vs_observed:.1f}%")
        
        print("\n📉 η_top1 Distribution:")
        print(f"   Mean:    {np.mean(self.eta_top1_distribution):.4f}")
        print(f"   Std:    {np.std(self.eta_top1_distribution):.4f}")
        print(f"   Min:    {np.min(self.eta_top1_distribution):.4f}")
        print(f"   Median: {np.median(self.eta_top1_distribution):.4f}")
        print(f"   Max:    {np.max(self.eta_top1_distribution):.4f}")
        
        print("=" * 70)


def evaluate_model(model: nn.Module,
                   test_loader: DataLoader,
                   config: Config,
                   powers_full: np.ndarray,
                   labels: np.ndarray,
                   observed_indices: np.ndarray,
                   optimal_powers: np.ndarray) -> EvaluationResults:
    """
    Comprehensive model evaluation with all baselines.
    
    Args:
        model:  Trained model
        test_loader:  Test data loader
        config: Configuration
        powers_full: Full power vectors, shape (n_test, K)
        labels: Oracle labels, shape (n_test,)
        observed_indices:  Indices of observed probes, shape (n_test, M)
        optimal_powers:  Theoretical optimal powers, shape (n_test,)
    """
    device = config.training.device
    model = model.to(device)
    model.eval()
    
    K = config.system.K
    M = config.system.M
    n_test = len(labels)
    top_m_values = config.eval.top_m_values
    
    # Collect all logits
    all_logits = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            all_logits.append(logits.cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0)
    
    # Initialize accumulators
    accuracy = {m: 0 for m in top_m_values}
    eta_sum = {m: 0.0 for m in top_m_values}
    eta_top1_dist = []
    
    # Baselines
    eta_best_observed_sum = 0.0
    eta_best_observed_dist = []
    eta_vs_theoretical_sum = 0.0
    
    rng = np.random.RandomState(config.data.seed + 100)
    eta_random_1_sum = 0.0
    eta_random_M_sum = 0.0
    
    for i in range(n_test):
        logits = all_logits[i]
        label = labels[i]
        P_best = powers_full[i, label]
        obs_idx = observed_indices[i]
        
        # Get top-m predictions
        sorted_indices = np.argsort(logits)[::-1]
        
        for m in top_m_values: 
            top_m_idx = sorted_indices[: m]
            if label in top_m_idx: 
                accuracy[m] += 1
            P_top_m = np.max(powers_full[i, top_m_idx])
            eta_sum[m] += P_top_m / P_best if P_best > 0 else 0
        
        # Store eta_top1 distribution
        top1_pred = sorted_indices[0]
        eta_top1_dist.append(powers_full[i, top1_pred] / P_best if P_best > 0 else 0)
        
        # Baseline:  best of observed M probes
        P_best_observed = np.max(powers_full[i, obs_idx])
        eta_best_obs = P_best_observed / P_best if P_best > 0 else 0
        eta_best_observed_sum += eta_best_obs
        eta_best_observed_dist.append(eta_best_obs)
        
        # Baseline: random pick 1 from K
        random_idx = rng.randint(0, K)
        P_random_1 = powers_full[i, random_idx]
        eta_random_1_sum += P_random_1 / P_best if P_best > 0 else 0
        
        # Baseline: best of random M from K
        random_M_idx = rng.choice(K, size=M, replace=False)
        P_random_M = np.max(powers_full[i, random_M_idx])
        eta_random_M_sum += P_random_M / P_best if P_best > 0 else 0
        
        # Theoretical comparison
        P_optimal = optimal_powers[i]
        eta_vs_theoretical_sum += P_best / P_optimal if P_optimal > 0 else 0
    
    # Normalize
    for m in top_m_values: 
        accuracy[m] /= n_test
        eta_sum[m] /= n_test
    
    results = EvaluationResults(
        accuracy_top1=accuracy.get(1, 0.0),
        accuracy_top2=accuracy.get(2, 0.0),
        accuracy_top4=accuracy.get(4, 0.0),
        accuracy_top8=accuracy.get(8, 0.0),
        eta_top1=eta_sum.get(1, 0.0),
        eta_top2=eta_sum.get(2, 0.0),
        eta_top4=eta_sum.get(4, 0.0),
        eta_top8=eta_sum.get(8, 0.0),
        eta_random_1=eta_random_1_sum / n_test,
        eta_random_M=eta_random_M_sum / n_test,
        eta_best_observed=eta_best_observed_sum / n_test,
        eta_oracle=1.0,
        eta_vs_theoretical=eta_vs_theoretical_sum / n_test,
        eta_top1_distribution=np.array(eta_top1_dist),
        eta_best_observed_distribution=np.array(eta_best_observed_dist),
        M=M,
        K=K
    )
    
    return results


def compute_baselines_only(powers_full: np.ndarray,
                           labels: np.ndarray,
                           observed_indices: np.ndarray,
                           K: int,
                           M: int,
                           seed: int = 42) -> Dict:
    """
    Compute baseline statistics without ML model.
    Useful for sanity checking.
    """
    n_samples = len(labels)
    rng = np.random.RandomState(seed)
    
    eta_random_1 = []
    eta_random_M = []
    eta_best_observed = []
    
    for i in range(n_samples):
        P_best = powers_full[i, labels[i]]
        if P_best <= 0:
            continue
        
        # Random 1
        random_idx = rng.randint(0, K)
        eta_random_1.append(powers_full[i, random_idx] / P_best)
        
        # Random M
        random_M_idx = rng.choice(K, size=M, replace=False)
        eta_random_M.append(np.max(powers_full[i, random_M_idx]) / P_best)
        
        # Best observed
        obs_idx = observed_indices[i]
        eta_best_observed.append(np.max(powers_full[i, obs_idx]) / P_best)
    
    return {
        'eta_random_1_mean': np.mean(eta_random_1),
        'eta_random_1_std': np.std(eta_random_1),
        'eta_random_M_mean': np.mean(eta_random_M),
        'eta_random_M_std':  np.std(eta_random_M),
        'eta_best_observed_mean': np.mean(eta_best_observed),
        'eta_best_observed_std': np.std(eta_best_observed),
        'expected_random_accuracy': 1.0 / K
    }