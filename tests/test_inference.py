import numpy as np
import pytest

from anomalybert.model.config import ModelConfig
from anomalybert.model.anomalybert import AnomalyBertModel
from anomalybert.data.normalization import MinMaxNormalizer
from anomalybert.data.synthetic import SyntheticScenario, generate_scenario
from anomalybert.training.checkpoint import save_checkpoint
from anomalybert.inference.detector import AnomalyDetector


@pytest.fixture
def model_path(tmp_path):
    config = ModelConfig(
        window_size=16, d_model=32, n_layers=1, n_heads=4,
        d_ff=64, head_hidden_dim=16, dropout=0.0,
    )
    model = AnomalyBertModel(config)
    normalizer = MinMaxNormalizer()
    normalizer.fit(np.array([0.0, 10.0]))
    path = tmp_path / "model.pt"
    save_checkpoint(model, normalizer, path)
    return path


class TestAnomalyDetector:
    def test_detect_returns_top_n(self, model_path):
        detector = AnomalyDetector(model_path)
        timestamps = np.arange(50, dtype=np.int64)
        values = np.random.randn(50)
        results = detector.detect(timestamps, values, top_n=5)
        assert len(results) == 5

    def test_detect_result_structure(self, model_path):
        detector = AnomalyDetector(model_path)
        timestamps = np.arange(50, dtype=np.int64)
        values = np.random.randn(50)
        results = detector.detect(timestamps, values, top_n=3)
        for r in results:
            assert "timestamp" in r
            assert "value" in r
            assert "score" in r
            assert 0 <= r["score"] <= 1

    def test_detect_sorted_by_score(self, model_path):
        detector = AnomalyDetector(model_path)
        timestamps = np.arange(50, dtype=np.int64)
        values = np.random.randn(50)
        results = detector.detect(timestamps, values, top_n=10)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_detect_from_csv(self, model_path, tmp_path):
        df = generate_scenario(SyntheticScenario.POINT_ANOMALY, n_samples=50)
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        detector = AnomalyDetector(model_path)
        results = detector.detect_from_csv(csv_path, top_n=5)
        assert len(results) == 5
