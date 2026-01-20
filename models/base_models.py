"""
Base Model Architectures
========================
Core neural network models for RIS probe-based control.
"""

import torch
import torch.nn as nn
from typing import List


class BaselineMLPPredictor(nn.Module):
    """
    Baseline MLP for probe selection with limited probing.

    Input: [masked_powers, mask] ∈ ℝ^{2K}
    Output: logits ∈ ℝ^K
    """

    def __init__(self,
                 input_size: int,
                 hidden_sizes: List[int],
                 output_size: int,
                 dropout_prob: float = 0.1,
                 use_batch_norm: bool = True):
        """
        Args:
            input_size: Input dimension (typically 2K)
            hidden_sizes: List of hidden layer sizes
            output_size: Output dimension (typically K)
            dropout_prob: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super(BaselineMLPPredictor, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            prev_size = hidden_size

        # Output layer (raw logits, no activation)
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch_size, input_size)

        Returns:
            Logits, shape (batch_size, output_size)
        """
        return self.network(x)


class AttentionMLPPredictor(nn.Module):
    """
    MLP with attention mechanism over observed probes.
    """

    def __init__(self,
                 input_size: int,
                 hidden_sizes: List[int],
                 output_size: int,
                 dropout_prob: float = 0.1,
                 use_batch_norm: bool = True,
                 num_attention_heads: int = 4):
        super(AttentionMLPPredictor, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_heads = num_attention_heads

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_sizes[0],
            num_heads=num_attention_heads,
            dropout=dropout_prob,
            batch_first=True
        )

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_sizes[0])

        # MLP after attention
        layers = []
        prev_size = hidden_sizes[0]

        for hidden_size in hidden_sizes[1:]:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.mlp = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project input
        x_proj = self.input_proj(x)  # (batch, hidden_sizes[0])

        # Add sequence dimension for attention
        x_seq = x_proj.unsqueeze(1)  # (batch, 1, hidden_sizes[0])

        # Apply self-attention
        attn_out, _ = self.attention(x_seq, x_seq, x_seq)
        attn_out = attn_out.squeeze(1)  # (batch, hidden_sizes[0])

        # Pass through MLP
        return self.mlp(attn_out)


class ResidualMLPPredictor(nn.Module):
    """
    MLP with residual connections for deeper architectures.
    """

    def __init__(self,
                 input_size: int,
                 hidden_sizes: List[int],
                 output_size: int,
                 dropout_prob: float = 0.1,
                 use_batch_norm: bool = True):
        super(ResidualMLPPredictor, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        # Input projection to first hidden size
        self.input_proj = nn.Linear(input_size, hidden_sizes[0])

        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.residual_blocks.append(
                ResidualBlock(
                    hidden_sizes[i],
                    hidden_sizes[i + 1],
                    dropout_prob,
                    use_batch_norm
                )
            )

        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection
        x = self.input_proj(x)

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Output
        return self.output_layer(x)


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, in_features: int, out_features: int, dropout_prob: float, use_batch_norm: bool):
        super(ResidualBlock, self).__init__()

        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features) if use_batch_norm else nn.Identity()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features) if use_batch_norm else nn.Identity()

        # Skip connection
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out