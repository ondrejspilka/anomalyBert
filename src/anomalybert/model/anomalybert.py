import torch
import torch.nn as nn

from .config import ModelConfig
from .embedding import PositionalEncoding, ValueEmbedding
from .encoder import TransformerEncoder
from .heads import AnomalyScoringHead


class AnomalyBertModel(nn.Module):
    """Encoder-only transformer for timeseries anomaly detection."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embedding = ValueEmbedding(config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, dropout=config.dropout)
        self.encoder = TransformerEncoder(
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )
        self.head = AnomalyScoringHead(config.d_model, config.head_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) normalized timeseries values
        Returns:
            (batch, seq_len) anomaly scores in [0, 1]
        """
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        scores = self.head(x)
        return scores

    def freeze_encoder(self) -> None:
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.pos_encoding.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for param in self.embedding.parameters():
            param.requires_grad = True
        for param in self.pos_encoding.parameters():
            param.requires_grad = True
        for param in self.encoder.parameters():
            param.requires_grad = True

    def set_head(self, head: nn.Module) -> None:
        self.head = head
