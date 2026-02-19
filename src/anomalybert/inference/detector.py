from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data.normalization import MinMaxNormalizer, ZScoreNormalizer
from ..data.tokenizer import TimeseriesTokenizer
from ..training.checkpoint import load_checkpoint


class AnomalyDetector:
    """High-level anomaly detection interface."""

    def __init__(self, model_path: str | Path):
        self.model, self.normalizer = load_checkpoint(model_path)
        self.model.eval()
        self.tokenizer = TimeseriesTokenizer(
            window_size=self.model.config.window_size,
            stride=self.model.config.stride,
        )

    def detect(
        self,
        timestamps: np.ndarray,
        values: np.ndarray,
        top_n: int = 10,
    ) -> list[dict]:
        # Normalize
        if self.normalizer is not None:
            norm_values = self.normalizer.transform(values.astype(np.float64))
        else:
            norm_values = values.astype(np.float32)

        # Tokenize
        idx_timestamps = np.arange(len(values), dtype=np.int64)
        windows = self.tokenizer.tokenize(idx_timestamps, norm_values)

        # Run inference on each window
        window_scores = []
        with torch.no_grad():
            for w in windows:
                input_tensor = torch.tensor(w["values"], dtype=torch.float32).unsqueeze(0)
                scores = self.model(input_tensor)
                window_scores.append(scores.squeeze(0).numpy())

        # Aggregate overlapping windows
        per_ts_scores = self.tokenizer.aggregate_scores_simple(
            window_scores, total_len=len(values)
        )

        # Build results sorted by score
        results = []
        for i in range(len(values)):
            results.append(
                {
                    "timestamp": timestamps[i],
                    "value": float(values[i]),
                    "score": float(per_ts_scores[i]),
                }
            )

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_n]

    def detect_from_csv(self, csv_path: str | Path, top_n: int = 10) -> list[dict]:
        df = pd.read_csv(csv_path)
        timestamps = df["timestamp"].values
        values = df["value"].values.astype(np.float64)
        return self.detect(timestamps, values, top_n)
