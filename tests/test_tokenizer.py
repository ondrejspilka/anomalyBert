import numpy as np
import pytest

from anomalybert.data.tokenizer import TimeseriesTokenizer


class TestTimeseriesTokenizer:
    def test_basic_tokenization(self):
        tok = TimeseriesTokenizer(window_size=10, stride=1)
        timestamps = np.arange(20, dtype=np.int64)
        values = np.random.randn(20).astype(np.float32)
        windows = tok.tokenize(timestamps, values)
        assert len(windows) == 11  # 20 - 10 + 1

    def test_window_size(self):
        tok = TimeseriesTokenizer(window_size=10, stride=1)
        timestamps = np.arange(20, dtype=np.int64)
        values = np.random.randn(20).astype(np.float32)
        windows = tok.tokenize(timestamps, values)
        for w in windows:
            assert len(w["values"]) == 10

    def test_stride(self):
        tok = TimeseriesTokenizer(window_size=10, stride=5)
        timestamps = np.arange(30, dtype=np.int64)
        values = np.random.randn(30).astype(np.float32)
        windows = tok.tokenize(timestamps, values)
        # Starts: 0, 5, 10, 15, 20 -> 5 windows
        assert len(windows) == 5

    def test_short_series_padding(self):
        tok = TimeseriesTokenizer(window_size=10, stride=1)
        timestamps = np.arange(5, dtype=np.int64)
        values = np.random.randn(5).astype(np.float32)
        windows = tok.tokenize(timestamps, values)
        assert len(windows) == 1
        assert len(windows[0]["values"]) == 10
        assert windows[0]["valid_len"] == 5

    def test_exact_window_size(self):
        tok = TimeseriesTokenizer(window_size=10, stride=1)
        timestamps = np.arange(10, dtype=np.int64)
        values = np.random.randn(10).astype(np.float32)
        windows = tok.tokenize(timestamps, values)
        assert len(windows) == 1
        assert windows[0]["valid_len"] == 10

    def test_aggregate_scores_simple(self):
        tok = TimeseriesTokenizer(window_size=4, stride=1)
        # 3 overlapping windows of size 4, stride 1 -> total_len = 6
        window_scores = [
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.2, 0.3, 0.4, 0.5]),
            np.array([0.3, 0.4, 0.5, 0.6]),
        ]
        result = tok.aggregate_scores_simple(window_scores, total_len=6)
        assert len(result) == 6
        # Position 0: only window 0 -> 0.1
        assert result[0] == pytest.approx(0.1, abs=1e-5)
        # Position 5: only window 2 -> 0.6
        assert result[5] == pytest.approx(0.6, abs=1e-5)
