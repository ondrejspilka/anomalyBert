import numpy as np
import pytest

from anomalybert.model.config import ModelConfig
from anomalybert.model.anomalybert import AnomalyBertModel
from anomalybert.data.normalization import MinMaxNormalizer
from anomalybert.data.synthetic import SyntheticScenario, generate_scenario
from anomalybert.training.checkpoint import save_checkpoint
from anomalybert.inference.detector import AnomalyDetector
from anomalybert.inference.onnx_export import export_to_onnx
from anomalybert.inference.onnx_detector import OnnxAnomalyDetector


@pytest.fixture
def model_and_paths(tmp_path):
    config = ModelConfig(
        window_size=16, d_model=32, n_layers=1, n_heads=4,
        d_ff=64, head_hidden_dim=16, dropout=0.0,
    )
    model = AnomalyBertModel(config)
    normalizer = MinMaxNormalizer()
    normalizer.fit(np.array([0.0, 10.0]))
    pt_path = tmp_path / "model.pt"
    save_checkpoint(model, normalizer, pt_path)
    onnx_path = tmp_path / "model.onnx"
    export_to_onnx(pt_path, onnx_path)
    return pt_path, onnx_path


class TestOnnxExport:
    def test_export_creates_files(self, model_and_paths):
        _, onnx_path = model_and_paths
        assert onnx_path.exists()
        meta_path = onnx_path.parent / (onnx_path.name + ".json")
        assert meta_path.exists()

    def test_metadata_content(self, model_and_paths):
        _, onnx_path = model_and_paths
        import json
        meta = json.loads((onnx_path.parent / (onnx_path.name + ".json")).read_text())
        assert "config" in meta
        assert "normalizer" in meta
        assert meta["config"]["window_size"] == 16
        assert meta["config"]["d_model"] == 32
        assert meta["normalizer"]["type"] == "minmax"


class TestOnnxDetector:
    def test_detect_returns_top_n(self, model_and_paths):
        _, onnx_path = model_and_paths
        detector = OnnxAnomalyDetector(onnx_path)
        timestamps = np.arange(50, dtype=np.int64)
        values = np.random.randn(50)
        results = detector.detect(timestamps, values, top_n=5)
        assert len(results) == 5

    def test_detect_result_structure(self, model_and_paths):
        _, onnx_path = model_and_paths
        detector = OnnxAnomalyDetector(onnx_path)
        timestamps = np.arange(50, dtype=np.int64)
        values = np.random.randn(50)
        results = detector.detect(timestamps, values, top_n=3)
        for r in results:
            assert "timestamp" in r
            assert "value" in r
            assert "score" in r
            assert 0 <= r["score"] <= 1

    def test_detect_sorted_by_score(self, model_and_paths):
        _, onnx_path = model_and_paths
        detector = OnnxAnomalyDetector(onnx_path)
        timestamps = np.arange(50, dtype=np.int64)
        values = np.random.randn(50)
        results = detector.detect(timestamps, values, top_n=10)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_detect_from_csv(self, model_and_paths, tmp_path):
        _, onnx_path = model_and_paths
        df = generate_scenario(SyntheticScenario.POINT_ANOMALY, n_samples=50, seed=42)
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        detector = OnnxAnomalyDetector(onnx_path)
        results = detector.detect_from_csv(csv_path, top_n=5)
        assert len(results) == 5


class TestOnnxVsPytorch:
    def test_scores_match(self, model_and_paths):
        pt_path, onnx_path = model_and_paths
        np.random.seed(123)
        timestamps = np.arange(50, dtype=np.int64)
        values = np.random.randn(50)

        pt_detector = AnomalyDetector(pt_path)
        pt_results = pt_detector.detect(timestamps, values.copy(), top_n=50)

        onnx_detector = OnnxAnomalyDetector(onnx_path)
        onnx_results = onnx_detector.detect(timestamps, values.copy(), top_n=50)

        pt_scores = {r["timestamp"]: r["score"] for r in pt_results}
        onnx_scores = {r["timestamp"]: r["score"] for r in onnx_results}

        for ts in pt_scores:
            assert abs(pt_scores[ts] - onnx_scores[ts]) < 1e-5, (
                f"Mismatch at timestamp {ts}: pt={pt_scores[ts]}, onnx={onnx_scores[ts]}"
            )
