import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with pre-norm residual connections."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed)
        x = x + self.dropout1(attn_out)

        # Pre-norm feedforward with residual
        normed = self.norm2(x)
        ff_out = self.ffn(normed)
        x = x + self.dropout2(ff_out)

        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""

    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)
