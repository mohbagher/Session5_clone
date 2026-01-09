"""
MLP model for RIS probe-based control with limited probing.

Architecture:  Masked Vector Approach
- Input: concatenation of [masked_powers, binary_mask] (size 2K)
  - masked_powers: K-dimensional vector with observed powers at their indices, zeros elsewhere
  - binary_mask:  K-dimensional vector with 1s at observed indices, 0s elsewhere
- Output: logits over K probes

Why this approach:
1.Simple and interpretable - the mask explicitly tells the model what was observed
2.No learned embeddings needed for probe indices
3.Position-aware - the model knows which specific probes were measured
4.Handles variable observation patterns naturally
"""

import torch
import torch.nn as nn
from typing import List

from config import Config


class LimitedProbingMLP(nn.Module):
    """
    MLP for probe selection with limited probing.
    
    Input: [masked_powers, mask] ∈ ℝ^{2K}
    Output: logits ∈ ℝ^K
    """
    
    def __init__(self,
                 K: int,
                 hidden_sizes: List[int],
                 dropout_prob: float = 0.1,
                 use_batch_norm: bool = True):
        """
        Args:
            K:  Number of probes (output size, input size is 2K)
            hidden_sizes: List of hidden layer sizes
            dropout_prob:  Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super(LimitedProbingMLP, self).__init__()
        
        self.K = K
        self.input_size = 2 * K  # [masked_powers, mask]
        self.output_size = K
        
        layers = []
        prev_size = self.input_size
        
        for hidden_size in hidden_sizes: 
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            prev_size = hidden_size
        
        # Output layer (raw logits, no activation)
        layers.append(nn.Linear(prev_size, self.output_size))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x:  torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [masked_powers, mask], shape (batch_size, 2K)
            
        Returns: 
            Logits, shape (batch_size, K)
        """
        return self.network(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted probe indices (top-1)."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)
    
    def predict_top_m(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """Get top-m predicted probe indices."""
        logits = self.forward(x)
        _, top_indices = torch.topk(logits, m, dim=1)
        return top_indices


def create_model(config: Config) -> LimitedProbingMLP: 
    """Create model from configuration."""
    K = config.system.K
    model_config = config.model
    
    model = LimitedProbingMLP(
        K=K,
        hidden_sizes=model_config.hidden_sizes,
        dropout_prob=model_config.dropout_prob,
        use_batch_norm=model_config.use_batch_norm
    )
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)