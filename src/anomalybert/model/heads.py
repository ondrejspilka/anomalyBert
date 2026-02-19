import torch
import torch.nn as nn


class AnomalyScoringHead(nn.Module):
    """Maps encoder output to per-timestep anomaly score in [0, 1]."""

    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, d_model) -> (batch, seq_len, 1) -> (batch, seq_len)
        return self.mlp(encoder_output).squeeze(-1)


class FinetuneHead(nn.Module):
    """Replacement head for domain-specific finetuning.

    Can be used for classification (anomaly/normal) or regression (anomaly probability).
    """

    def __init__(self, d_model: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        return self.mlp(encoder_output).squeeze(-1)
