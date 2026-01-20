"""
learnable_m_models.py

Repo-compatible "learnable M usage" models for the limited probing setup.

Key compatibility requirements (matches your training/evaluation):
- Forward input: x in R^(2K) where:
    x[:, :K]  = masked_powers
    x[:, K:]  = mask (0/1)
- Forward output: logits in R^K (single tensor)

Important note:
- Your current pipeline chooses the observed subset (size M) in data_generation.py.
  So these models do NOT change which probes are observed.
  They learn how to best exploit the observed probes (attention/gating), which is
  the only thing possible without changing data generation.

If later you want the model to truly choose which M probes to measure, you must
move selection into data_generation (or an online/active sensing loop).
"""

from __future__ import annotations
from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _xavier_init(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _build_mlp(in_dim: int,
               hidden_sizes: List[int],
               out_dim: int,
               dropout_prob: float,
               use_batch_norm: bool) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU())
        if dropout_prob and dropout_prob > 0:
            layers.append(nn.Dropout(dropout_prob))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    net = nn.Sequential(*layers)
    _xavier_init(net)
    return net


class AttentionBasedMLP(nn.Module):
    """
    Soft attention over the observed probes (mask-aware).
    Produces logits over K probes.

    This is safe and fully compatible with your current training/evaluation code.
    """
    def __init__(self,
                 K: int,
                 hidden_sizes: List[int],
                 M: Optional[int] = None,
                 dropout_prob: float = 0.1,
                 use_batch_norm: bool = True,
                 attention_hidden: int = 128,
                 temperature: float = 1.0):
        super().__init__()
        self.K = int(K)
        self.M = M
        self.temperature = float(temperature)

        # Attention score head: (2K) -> K
        self.attn_net = nn.Sequential(
            nn.Linear(2 * self.K, attention_hidden),
            nn.ReLU(),
            nn.Linear(attention_hidden, self.K),
        )
        _xavier_init(self.attn_net)

        # Predictor: input is (gated_powers, mask) => 2K
        self.predictor = _build_mlp(
            in_dim=2 * self.K,
            hidden_sizes=hidden_sizes,
            out_dim=self.K,
            dropout_prob=dropout_prob,
            use_batch_norm=use_batch_norm
        )

        self._last_attention: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2K)
        K = self.K
        masked_powers = x[:, :K]
        mask = x[:, K:]

        # raw scores
        scores = self.attn_net(x)  # (B, K)

        # mask-aware softmax (only on observed entries)
        very_neg = torch.finfo(scores.dtype).min
        scores_masked = scores.masked_fill(mask <= 0.0, very_neg)
        attn = F.softmax(scores_masked / max(self.temperature, 1e-6), dim=-1)

        # scale attention so sum(attn on observed) ~= 1, then gate powers
        # (soft focus among observed probes)
        gated_powers = masked_powers * attn

        # final input keeps mask so model knows what was observed
        x_gated = torch.cat([gated_powers, mask], dim=-1)

        self._last_attention = attn.detach()
        logits = self.predictor(x_gated)
        return logits

    def get_last_attention(self) -> Optional[torch.Tensor]:
        return self._last_attention


class LearnedTopKMLP(nn.Module):
    """
    "LearnedTopK" for your repo means: learn a sparse weighting over observed probes,
    then predict logits over all K.

    It stays fully compatible: forward returns logits only.
    """
    def __init__(self,
                 K: int,
                 hidden_sizes: List[int],
                 M: int,
                 dropout_prob: float = 0.1,
                 use_batch_norm: bool = True,
                 temperature: float = 0.7,
                 sparsity_strength: float = 0.0):
        super().__init__()
        self.K = int(K)
        self.M = int(M)
        self.temperature = float(temperature)
        self.sparsity_strength = float(sparsity_strength)

        self.score_net = nn.Sequential(
            nn.Linear(2 * self.K, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], self.K)
        )
        _xavier_init(self.score_net)

        self.predictor = _build_mlp(
            in_dim=2 * self.K,
            hidden_sizes=hidden_sizes,
            out_dim=self.K,
            dropout_prob=dropout_prob,
            use_batch_norm=use_batch_norm
        )

        self._last_scores: Optional[torch.Tensor] = None
        self._last_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K = self.K
        masked_powers = x[:, :K]
        mask = x[:, K:]

        scores = self.score_net(x)  # (B, K)

        # mask-aware softmax
        very_neg = torch.finfo(scores.dtype).min
        scores_masked = scores.masked_fill(mask <= 0.0, very_neg)
        weights = F.softmax(scores_masked / max(self.temperature, 1e-6), dim=-1)

        # Optional sparsity encouragement (does not break training)
        # This is a gentle regularizer signal via internal attribute only.
        # If you want it in the loss, you must add it in training.py.
        # Kept here for future extension.
        self._last_scores = scores.detach()
        self._last_weights = weights.detach()

        gated_powers = masked_powers * weights
        x_gated = torch.cat([gated_powers, mask], dim=-1)
        logits = self.predictor(x_gated)
        return logits

    def get_last_weights(self) -> Optional[torch.Tensor]:
        return self._last_weights

    def get_last_scores(self) -> Optional[torch.Tensor]:
        return self._last_scores


class GumbelMLP(nn.Module):
    """
    Repo-compatible "Gumbel" variant:
    creates a sharper (more discrete-like) weighting over observed probes using
    Gumbel-Softmax during training, but still returns logits only.

    Note: This does NOT change which probes are observed, it only sharpens usage.
    """
    def __init__(self,
                 K: int,
                 hidden_sizes: List[int],
                 M: int,
                 dropout_prob: float = 0.1,
                 use_batch_norm: bool = True,
                 temperature: float = 0.5):
        super().__init__()
        self.K = int(K)
        self.M = int(M)
        self.temperature = float(temperature)

        self.score_net = nn.Sequential(
            nn.Linear(2 * self.K, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_sizes[0], self.K)
        )
        _xavier_init(self.score_net)

        self.predictor = _build_mlp(
            in_dim=2 * self.K,
            hidden_sizes=hidden_sizes,
            out_dim=self.K,
            dropout_prob=dropout_prob,
            use_batch_norm=use_batch_norm
        )

        self._last_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K = self.K
        masked_powers = x[:, :K]
        mask = x[:, K:]

        scores = self.score_net(x)

        very_neg = torch.finfo(scores.dtype).min
        scores_masked = scores.masked_fill(mask <= 0.0, very_neg)

        if self.training:
            # Gumbel noise then softmax
            u = torch.rand_like(scores_masked)
            g = -torch.log(-torch.log(u + 1e-10) + 1e-10)
            logits = (scores_masked + g) / max(self.temperature, 1e-6)
            weights = F.softmax(logits, dim=-1)
        else:
            weights = F.softmax(scores_masked / max(self.temperature, 1e-6), dim=-1)

        self._last_weights = weights.detach()

        gated_powers = masked_powers * weights
        x_gated = torch.cat([gated_powers, mask], dim=-1)
        out = self.predictor(x_gated)
        return out

    def get_last_weights(self) -> Optional[torch.Tensor]:
        return self._last_weights


class RLBasedMLP(nn.Module):
    """
    Placeholder that remains compatible with your pipeline.
    True RL-based probe selection requires changing the environment loop and loss.

    For now: behaves like AttentionBasedMLP so it trains and runs without breaking.
    """
    def __init__(self,
                 K: int,
                 hidden_sizes: List[int],
                 M: int,
                 dropout_prob: float = 0.1,
                 use_batch_norm: bool = True):
        super().__init__()
        self.inner = AttentionBasedMLP(
            K=K,
            hidden_sizes=hidden_sizes,
            M=M,
            dropout_prob=dropout_prob,
            use_batch_norm=use_batch_norm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)


def build_model_from_name(model_name: str,
                          K: int,
                          hidden_sizes: List[int],
                          M: int,
                          dropout_prob: float = 0.1,
                          use_batch_norm: bool = True) -> nn.Module:
    """
    Small factory (optional). Useful if you later modify experiment_runner.py
    to instantiate the selected model class (not just hidden_sizes).
    """
    name = str(model_name)

    if name == "Attention_MLP":
        return AttentionBasedMLP(K, hidden_sizes, M, dropout_prob, use_batch_norm)
    if name == "LearnedTopK_MLP":
        return LearnedTopKMLP(K, hidden_sizes, M, dropout_prob, use_batch_norm)
    if name == "Gumbel_MLP":
        return GumbelMLP(K, hidden_sizes, M, dropout_prob, use_batch_norm)
    if name == "RL_MLP":
        return RLBasedMLP(K, hidden_sizes, M, dropout_prob, use_batch_norm)

    # Fallback: simple MLP like your baseline behaviour
    return _build_mlp(2 * int(K), hidden_sizes, int(K), dropout_prob, use_batch_norm)
