from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .normalization import MinMaxNormalizer, ZScoreNormalizer, create_normalizer
from .tokenizer import TimeseriesTokenizer


class TimeseriesDataset(Dataset):
    """PyTorch dataset for timeseries anomaly detection."""

    def __init__(
        self,
        values: np.ndarray,
        labels: np.ndarray | None,
        tokenizer: TimeseriesTokenizer,
        normalizer: MinMaxNormalizer | ZScoreNormalizer | None = None,
    ):
        self.tokenizer = tokenizer

        # Normalize
        if normalizer is not None:
            values = normalizer.transform(values)

        # Create dummy timestamps as indices for tokenization
        timestamps = np.arange(len(values), dtype=np.int64)

        # Tokenize into windows
        windows = tokenizer.tokenize(timestamps, values)
        self.windows = []
        for w in windows:
            item = {"values": torch.tensor(w["values"], dtype=torch.float32)}
            if labels is not None:
                start = int(w["timestamps"][0])
                end = start + w["valid_len"]
                win_labels = labels[start:end]
                if len(win_labels) < tokenizer.window_size:
                    win_labels = np.pad(win_labels, (0, tokenizer.window_size - len(win_labels)))
                item["labels"] = torch.tensor(win_labels[:tokenizer.window_size], dtype=torch.float32)
            self.windows.append(item)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        return self.windows[idx]

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        tokenizer: TimeseriesTokenizer,
        normalizer: MinMaxNormalizer | ZScoreNormalizer | None = None,
        fit_normalizer: bool = False,
        is_training: bool = True,
    ) -> "TimeseriesDataset":
        df = pd.read_csv(path)
        values = df["value"].values.astype(np.float64)

        if fit_normalizer and normalizer is not None:
            normalizer.fit(values)

        labels = None
        if is_training and "probability" in df.columns:
            labels = df["probability"].values.astype(np.float64)

        return cls(values, labels, tokenizer, normalizer)

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        tokenizer: TimeseriesTokenizer,
        normalizer: MinMaxNormalizer | ZScoreNormalizer | None = None,
        is_training: bool = True,
    ) -> "TimeseriesDataset":
        """Load and concatenate all CSV files in a directory."""
        directory = Path(directory)
        all_values = []
        all_labels = []

        for csv_path in sorted(directory.glob("*.csv")):
            df = pd.read_csv(csv_path)
            all_values.append(df["value"].values.astype(np.float64))
            if is_training and "probability" in df.columns:
                all_labels.append(df["probability"].values.astype(np.float64))

        values = np.concatenate(all_values)
        labels = np.concatenate(all_labels) if all_labels else None

        if normalizer is not None:
            normalizer.fit(values)

        return cls(values, labels, tokenizer, normalizer)
