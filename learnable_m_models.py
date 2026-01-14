"""
Learnable M Selection - New Model Architectures
================================================

CONCEPT: Instead of randomly selecting M elements from K, let the neural network
         LEARN which M elements to activate for maximum efficiency.

APPROACHES:
1. Attention-based Selection (Soft)
2. Gumbel-Softmax Selection (Differentiable Hard)
3. Top-K Selection with Learned Scores (Hard)
4. Continuous Relaxation (Soft then Hard)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# APPROACH 1: Attention-Based Selection (Soft Attention)
# ============================================================================

class AttentionBasedMLP(nn.Module):
    """
    Uses attention mechanism to learn importance of each element.
    Softly weights elements (no hard selection).
    """
    def __init__(self, K, hidden_sizes, M=None, dropout_prob=0.0, use_batch_norm=False):
        super().__init__()
        self.K = K
        self.M = M  # Not used in soft attention, but kept for compatibility
        
        # Channel features (from real and imag parts)
        input_dim = 2 * K
        
        # Attention mechanism to learn element importance
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], K),  # Score for each of K elements
            nn.Softmax(dim=-1)  # Soft attention weights
        )
        
        # Main prediction network
        layers = []
        prev_size = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            prev_size = size
        
        layers.append(nn.Linear(prev_size, K))
        self.network = nn.Sequential(*layers)
        
    def forward(self, h_complex):
        """
        Args:
            h_complex: Complex channel (batch_size, K)
        Returns:
            predictions: Index predictions (batch_size, K)
        """
        # Convert complex to real input
        h_real = torch.cat([h_complex.real, h_complex.imag], dim=-1)
        
        # Compute attention weights (batch_size, K)
        attention_weights = self.attention(h_real)
        
        # Apply attention to input features
        # Repeat real and imag parts and multiply by attention
        h_real_weighted = h_complex.real * attention_weights
        h_imag_weighted = h_complex.imag * attention_weights
        h_weighted = torch.cat([h_real_weighted, h_imag_weighted], dim=-1)
        
        # Make prediction
        output = self.network(h_weighted)
        
        return output, attention_weights  # Return attention for visualization


# ============================================================================
# APPROACH 2: Gumbel-Softmax Selection (Differentiable Hard Selection)
# ============================================================================

class GumbelMLP(nn.Module):
    """
    Uses Gumbel-Softmax trick to make HARD selection of M elements
    while maintaining differentiability during training.
    """
    def __init__(self, K, hidden_sizes, M, dropout_prob=0.0, use_batch_norm=False,
                 temperature=1.0, hard=True):
        super().__init__()
        self.K = K
        self.M = M
        self.temperature = temperature
        self.hard = hard
        
        input_dim = 2 * K
        
        # Element selection network (learns logits for each element)
        self.selector = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_sizes[0], K)  # Logits for K elements
        )
        
        # Main prediction network (operates on selected elements)
        layers = []
        prev_size = 2 * M  # Only M elements selected
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            prev_size = size
        
        layers.append(nn.Linear(prev_size, K))
        self.network = nn.Sequential(*layers)
        
    def gumbel_softmax_topk(self, logits, k, temperature=1.0, hard=True):
        """
        Sample top-k using Gumbel-Softmax trick.
        """
        # Add Gumbel noise
        gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        gumbels = (logits + gumbels) / temperature
        
        # Get top-k
        topk_values, topk_indices = torch.topk(gumbels, k, dim=-1)
        
        # Create one-hot mask
        mask = torch.zeros_like(logits)
        mask.scatter_(-1, topk_indices, 1.0)
        
        if hard:
            # Straight-through estimator
            mask_hard = mask.detach()
            mask = mask_hard - mask.detach() + mask
        
        return mask, topk_indices
    
    def forward(self, h_complex):
        """
        Args:
            h_complex: Complex channel (batch_size, K)
        Returns:
            predictions: Index predictions (batch_size, K)
            selected_indices: Which M elements were selected (batch_size, M)
        """
        batch_size = h_complex.shape[0]
        
        # Convert to real
        h_real = torch.cat([h_complex.real, h_complex.imag], dim=-1)
        
        # Compute selection logits
        logits = self.selector(h_real)
        
        # Select top M elements using Gumbel-Softmax
        mask, selected_indices = self.gumbel_softmax_topk(
            logits, self.M, self.temperature, self.hard
        )
        
        # Extract selected elements
        h_real_selected = h_complex.real * mask
        h_imag_selected = h_complex.imag * mask
        
        # Gather only the selected M elements (for efficiency)
        h_real_m = torch.gather(h_complex.real, -1, selected_indices)
        h_imag_m = torch.gather(h_complex.imag, -1, selected_indices)
        h_selected = torch.cat([h_real_m, h_imag_m], dim=-1)
        
        # Make prediction
        output = self.network(h_selected)
        
        return output, selected_indices


# ============================================================================
# APPROACH 3: Top-K with Learned Importance Scores (Simple & Effective)
# ============================================================================

class LearnedTopKMLP(nn.Module):
    """
    Simple approach: Learn importance score for each element,
    then select top-M at inference (or softly during training).
    
    RECOMMENDED: This is the simplest and most interpretable approach.
    """
    def __init__(self, K, hidden_sizes, M, dropout_prob=0.0, use_batch_norm=False,
                 training_mode='soft'):
        super().__init__()
        self.K = K
        self.M = M
        self.training_mode = training_mode  # 'soft' or 'hard'
        
        input_dim = 2 * K
        
        # Importance scoring network
        self.importance_net = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], K),  # Score for each element
        )
        
        # Main prediction network
        layers = []
        prev_size = input_dim  # Uses all elements (weighted)
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            prev_size = size
        
        layers.append(nn.Linear(prev_size, K))
        self.network = nn.Sequential(*layers)
        
    def forward(self, h_complex):
        """
        Args:
            h_complex: Complex channel (batch_size, K)
        Returns:
            predictions: Index predictions (batch_size, K)
            selected_indices: Top-M indices (batch_size, M)
        """
        batch_size = h_complex.shape[0]
        
        # Convert to real
        h_real = torch.cat([h_complex.real, h_complex.imag], dim=-1)
        
        # Compute importance scores
        importance_scores = self.importance_net(h_real)  # (batch_size, K)
        
        if self.training and self.training_mode == 'soft':
            # During training: Use sigmoid-weighted (soft selection)
            weights = torch.sigmoid(importance_scores)
            
            # Scale weights so that sum ≈ M (similar to selecting M elements)
            weights = weights * (self.M / (weights.sum(dim=-1, keepdim=True) + 1e-6))
            
            # Apply weights
            h_real_weighted = h_complex.real * weights
            h_imag_weighted = h_complex.imag * weights
            h_weighted = torch.cat([h_real_weighted, h_imag_weighted], dim=-1)
            
            # Get top-M for returning (but not used in forward)
            _, selected_indices = torch.topk(importance_scores, self.M, dim=-1)
            
        else:
            # During inference: Hard top-M selection
            _, selected_indices = torch.topk(importance_scores, self.M, dim=-1)
            
            # Create binary mask
            mask = torch.zeros_like(importance_scores)
            mask.scatter_(-1, selected_indices, 1.0)
            
            # Apply mask
            h_real_weighted = h_complex.real * mask
            h_imag_weighted = h_complex.imag * mask
            h_weighted = torch.cat([h_real_weighted, h_imag_weighted], dim=-1)
        
        # Make prediction
        output = self.network(h_weighted)
        
        return output, selected_indices


# ============================================================================
# APPROACH 4: Reinforcement Learning-Based Selection (Advanced)
# ============================================================================

class RLBasedMLP(nn.Module):
    """
    Uses policy gradient to learn which M elements to select.
    More complex but can optimize directly for efficiency metric.
    """
    def __init__(self, K, hidden_sizes, M, dropout_prob=0.0, use_batch_norm=False):
        super().__init__()
        self.K = K
        self.M = M
        
        input_dim = 2 * K
        
        # Policy network (outputs action probabilities)
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], K),
        )
        
        # Value network (for baseline in REINFORCE)
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0] // 2),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0] // 2, 1),
        )
        
        # Prediction network
        layers = []
        prev_size = 2 * M
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            prev_size = size
        
        layers.append(nn.Linear(prev_size, K))
        self.network = nn.Sequential(*layers)
        
    def sample_action(self, logits):
        """Sample M elements without replacement."""
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample M indices without replacement
        batch_size = logits.shape[0]
        selected_indices = []
        log_probs = []
        
        for b in range(batch_size):
            indices = []
            log_prob = 0
            remaining_logits = logits[b].clone()
            
            for _ in range(self.M):
                # Sample one index
                dist = torch.distributions.Categorical(logits=remaining_logits)
                idx = dist.sample()
                
                indices.append(idx.item())
                log_prob += dist.log_prob(idx)
                
                # Mask out selected index
                remaining_logits[idx] = float('-inf')
            
            selected_indices.append(torch.tensor(indices, device=logits.device))
            log_probs.append(log_prob)
        
        selected_indices = torch.stack(selected_indices)
        log_probs = torch.stack(log_probs)
        
        return selected_indices, log_probs
    
    def forward(self, h_complex):
        """
        Args:
            h_complex: Complex channel (batch_size, K)
        Returns:
            predictions: Index predictions (batch_size, K)
            selected_indices: Sampled M indices (batch_size, M)
            log_probs: Log probabilities of actions
            value: Value estimate for baseline
        """
        # Convert to real
        h_real = torch.cat([h_complex.real, h_complex.imag], dim=-1)
        
        # Compute policy logits
        logits = self.policy_net(h_real)
        
        # Compute value estimate
        value = self.value_net(h_real).squeeze(-1)
        
        # Sample action (which M elements)
        selected_indices, log_probs = self.sample_action(logits)
        
        # Extract selected elements
        h_real_m = torch.gather(h_complex.real, -1, selected_indices)
        h_imag_m = torch.gather(h_complex.imag, -1, selected_indices)
        h_selected = torch.cat([h_real_m, h_imag_m], dim=-1)
        
        # Make prediction
        output = self.network(h_selected)
        
        return output, selected_indices, log_probs, value


# ============================================================================
# MODEL REGISTRY ADDITIONS
# ============================================================================

# Add to your model_registry.py:

LEARNABLE_M_MODELS = {
    'Attention_MLP': {
        'type': 'attention',
        'hidden_sizes': [512, 256, 128],
        'description': 'Soft attention-based element selection'
    },
    'Gumbel_MLP': {
        'type': 'gumbel',
        'hidden_sizes': [512, 256, 128],
        'description': 'Hard selection with Gumbel-Softmax'
    },
    'LearnedTopK_MLP': {
        'type': 'learned_topk',
        'hidden_sizes': [512, 256, 128],
        'description': 'Learn importance scores, select top-M (RECOMMENDED)'
    },
    'RL_MLP': {
        'type': 'rl',
        'hidden_sizes': [512, 256, 128],
        'description': 'Reinforcement learning-based selection'
    }
}


def get_learnable_m_model(model_name, K, M, **kwargs):
    """
    Factory function to create learnable-M models.
    
    Usage:
        model = get_learnable_m_model('LearnedTopK_MLP', K=64, M=8)
    """
    if model_name not in LEARNABLE_M_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    
    config = LEARNABLE_M_MODELS[model_name]
    hidden_sizes = config['hidden_sizes']
    model_type = config['type']
    
    if model_type == 'attention':
        return AttentionBasedMLP(K, hidden_sizes, M, **kwargs)
    elif model_type == 'gumbel':
        return GumbelMLP(K, hidden_sizes, M, **kwargs)
    elif model_type == 'learned_topk':
        return LearnedTopKMLP(K, hidden_sizes, M, **kwargs)
    elif model_type == 'rl':
        return RLBasedMLP(K, hidden_sizes, M, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
