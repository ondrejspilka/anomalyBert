import json
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort

from ..data.normalization import MinMaxNormalizer, ZScoreNormalizer
from ..data.tokenizer import TimeseriesTokenizer
from ..model.config import ModelConfig


class OnnxAnomalyDetector:
    """Anomaly detection using an ONNX model (no PyTorch dependency at inference)."""

    def __init__(self, onnx_path: str | Path):
        onnx_path = Path(onnx_path)
        meta_path = Path(str(onnx_path) + ".json")

        # Load ONNX session
        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        # Load metadata
        metadata = json.loads(meta_path.read_text())
        self.config = ModelConfig.from_dict(metadata["config"])

        # Reconstruct normalizer
        self.normalizer = None
        if metadata.get("normalizer") is not None:
            norm_data = metadata["normalizer"]
            if norm_data["type"] == "minmax":
                self.normalizer = MinMaxNormalizer()
                self.normalizer.min_val = norm_data["min"]
                self.normalizer.max_val = norm_data["max"]
            elif norm_data["type"] == "zscore":
                self.normalizer = ZScoreNormalizer()
                self.normalizer.mean = norm_data["mean"]
                self.normalizer.std = norm_data["std"]

        self.tokenizer = TimeseriesTokenizer(
            window_size=self.config.window_size,
            stride=self.config.stride,
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
        for w in windows:
            input_array = w["values"].reshape(1, -1).astype(np.float32)
            (scores,) = self.session.run(None, {self.input_name: input_array})
            window_scores.append(scores.squeeze(0))

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
