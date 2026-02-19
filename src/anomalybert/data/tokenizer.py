import numpy as np


class TimeseriesTokenizer:
    """Converts raw timeseries into sliding-window token sequences."""

    def __init__(self, window_size: int = 64, stride: int = 1):
        self.window_size = window_size
        self.stride = stride

    def tokenize(
        self, timestamps: np.ndarray, values: np.ndarray
    ) -> list[dict]:
        n = len(values)
        if n < self.window_size:
            # Pad with zeros if series is shorter than window
            pad_len = self.window_size - n
            padded_values = np.pad(values, (0, pad_len), mode="constant", constant_values=0)
            padded_ts = np.pad(timestamps, (0, pad_len), mode="constant", constant_values=0)
            return [
                {
                    "values": padded_values.astype(np.float32),
                    "timestamps": padded_ts,
                    "positions": np.arange(self.window_size),
                    "valid_len": n,
                }
            ]

        windows = []
        for start in range(0, n - self.window_size + 1, self.stride):
            end = start + self.window_size
            windows.append(
                {
                    "values": values[start:end].astype(np.float32),
                    "timestamps": timestamps[start:end],
                    "positions": np.arange(self.window_size),
                    "valid_len": self.window_size,
                }
            )
        return windows

    def aggregate_scores(
        self,
        window_scores: list[np.ndarray],
        window_timestamps: list[np.ndarray],
        total_len: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate overlapping window scores back to per-timestamp scores."""
        score_sums = np.zeros(total_len, dtype=np.float64)
        score_counts = np.zeros(total_len, dtype=np.float64)

        for scores, timestamps in zip(window_scores, window_timestamps):
            valid = timestamps > 0
            for i, (ts_idx, s) in enumerate(zip(range(len(scores)), scores)):
                if i < len(timestamps) and valid[i]:
                    # Find position by timestamp offset
                    pos = i
                    # Use first window's start as reference
                    if len(window_timestamps) > 0 and len(window_timestamps[0]) > 0:
                        offset = int(timestamps[0] - window_timestamps[0][0])
                        pos = offset + i
                    if 0 <= pos < total_len:
                        score_sums[pos] += s
                        score_counts[pos] += 1

        # Avoid division by zero
        score_counts = np.maximum(score_counts, 1)
        return score_sums / score_counts

    def aggregate_scores_simple(
        self,
        window_scores: list[np.ndarray],
        total_len: int,
    ) -> np.ndarray:
        """Simpler aggregation using stride-based position tracking."""
        score_sums = np.zeros(total_len, dtype=np.float64)
        score_counts = np.zeros(total_len, dtype=np.float64)

        for win_idx, scores in enumerate(window_scores):
            start = win_idx * self.stride
            for i, s in enumerate(scores):
                pos = start + i
                if pos < total_len:
                    score_sums[pos] += s
                    score_counts[pos] += 1

        score_counts = np.maximum(score_counts, 1)
        return (score_sums / score_counts).astype(np.float32)
