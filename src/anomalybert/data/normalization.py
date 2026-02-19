import json
from pathlib import Path

import numpy as np


class MinMaxNormalizer:
    def __init__(self):
        self.min_val: float | None = None
        self.max_val: float | None = None

    def fit(self, values: np.ndarray) -> "MinMaxNormalizer":
        self.min_val = float(np.min(values))
        self.max_val = float(np.max(values))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.min_val is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        range_val = self.max_val - self.min_val
        if range_val == 0:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - self.min_val) / range_val).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.min_val is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        range_val = self.max_val - self.min_val
        return (values * range_val + self.min_val).astype(np.float32)

    def save(self, path: str | Path) -> None:
        data = {"type": "minmax", "min": self.min_val, "max": self.max_val}
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "MinMaxNormalizer":
        data = json.loads(Path(path).read_text())
        n = cls()
        n.min_val = data["min"]
        n.max_val = data["max"]
        return n


class ZScoreNormalizer:
    def __init__(self):
        self.mean: float | None = None
        self.std: float | None = None

    def fit(self, values: np.ndarray) -> "ZScoreNormalizer":
        self.mean = float(np.mean(values))
        self.std = float(np.std(values))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        if self.std == 0:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return (values * self.std + self.mean).astype(np.float32)

    def save(self, path: str | Path) -> None:
        data = {"type": "zscore", "mean": self.mean, "std": self.std}
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "ZScoreNormalizer":
        data = json.loads(Path(path).read_text())
        n = cls()
        n.mean = data["mean"]
        n.std = data["std"]
        return n


def create_normalizer(normalization: str = "minmax"):
    if normalization == "minmax":
        return MinMaxNormalizer()
    elif normalization == "zscore":
        return ZScoreNormalizer()
    elif normalization == "none":
        return None
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
