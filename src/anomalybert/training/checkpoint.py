from pathlib import Path

import torch

from ..model.anomalybert import AnomalyBertModel
from ..model.config import ModelConfig
from ..data.normalization import MinMaxNormalizer, ZScoreNormalizer, create_normalizer


def save_checkpoint(
    model: AnomalyBertModel,
    normalizer: MinMaxNormalizer | ZScoreNormalizer | None,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "normalizer": None,
    }
    if normalizer is not None:
        if isinstance(normalizer, MinMaxNormalizer):
            checkpoint["normalizer"] = {
                "type": "minmax",
                "min": normalizer.min_val,
                "max": normalizer.max_val,
            }
        elif isinstance(normalizer, ZScoreNormalizer):
            checkpoint["normalizer"] = {
                "type": "zscore",
                "mean": normalizer.mean,
                "std": normalizer.std,
            }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
) -> tuple[AnomalyBertModel, MinMaxNormalizer | ZScoreNormalizer | None]:
    checkpoint = torch.load(path, weights_only=False)
    config = ModelConfig.from_dict(checkpoint["config"])
    model = AnomalyBertModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    normalizer = None
    if checkpoint.get("normalizer") is not None:
        norm_data = checkpoint["normalizer"]
        if norm_data["type"] == "minmax":
            normalizer = MinMaxNormalizer()
            normalizer.min_val = norm_data["min"]
            normalizer.max_val = norm_data["max"]
        elif norm_data["type"] == "zscore":
            normalizer = ZScoreNormalizer()
            normalizer.mean = norm_data["mean"]
            normalizer.std = norm_data["std"]

    return model, normalizer
