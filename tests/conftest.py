import pytest
import numpy as np

from anomalybert.model.config import ModelConfig


@pytest.fixture
def small_config():
    """Small model config for fast tests."""
    return ModelConfig(
        window_size=16,
        stride=1,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
        head_hidden_dim=16,
        normalization="minmax",
    )


@pytest.fixture
def sample_values():
    """Simple timeseries values."""
    rng = np.random.RandomState(42)
    return rng.normal(0, 1, 100).astype(np.float64)


@pytest.fixture
def sample_timestamps():
    """Timestamps for 100 data points."""
    return np.arange(1000000000, 1000000000 + 100 * 60, 60, dtype=np.int64)
