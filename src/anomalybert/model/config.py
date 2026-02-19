from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    # Tokenizer
    window_size: int = 64
    stride: int = 1

    # Embedding
    d_model: int = 128

    # Transformer encoder
    n_layers: int = 3
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.1

    # Anomaly scoring head
    head_hidden_dim: int = 64

    # Normalization
    normalization: str = "minmax"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
