import pytest
from click.testing import CliRunner

from anomalybert.cli.main import cli
from anomalybert.data.synthetic import SyntheticScenario, generate_scenario
from anomalybert.model.config import ModelConfig
from anomalybert.model.anomalybert import AnomalyBertModel
from anomalybert.data.normalization import MinMaxNormalizer
from anomalybert.training.checkpoint import save_checkpoint

import numpy as np


@pytest.fixture
def runner():
    return CliRunner()


class TestGenerateCommand:
    def test_generate_all(self, runner, tmp_path):
        result = runner.invoke(cli, ["generate", "--output-dir", str(tmp_path), "--n-samples", "30"])
        assert result.exit_code == 0
        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) == len(SyntheticScenario)

    def test_generate_single(self, runner, tmp_path):
        result = runner.invoke(
            cli, ["generate", "--output-dir", str(tmp_path), "--scenario", "point_anomaly", "--n-samples", "50"]
        )
        assert result.exit_code == 0
        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) == 1

    def test_generate_unknown_scenario(self, runner, tmp_path):
        result = runner.invoke(
            cli, ["generate", "--output-dir", str(tmp_path), "--scenario", "nonexistent"]
        )
        assert result.exit_code != 0


class TestTrainCommand:
    def test_train_basic(self, runner, tmp_path):
        # Generate data
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        df = generate_scenario(SyntheticScenario.POINT_ANOMALY, n_samples=50)
        df.to_csv(data_dir / "train.csv", index=False)

        model_path = tmp_path / "model.pt"
        result = runner.invoke(cli, [
            "train",
            "--data", str(data_dir / "train.csv"),
            "--output", str(model_path),
            "--epochs", "2",
            "--window-size", "16",
            "--d-model", "32",
            "--n-layers", "1",
            "--n-heads", "4",
            "--d-ff", "64",
            "--batch-size", "4",
            "--val-split", "0.0",
        ])
        assert result.exit_code == 0
        assert model_path.exists()


class TestDetectCommand:
    def test_detect_table(self, runner, tmp_path):
        # Create a model
        config = ModelConfig(
            window_size=16, d_model=32, n_layers=1, n_heads=4,
            d_ff=64, head_hidden_dim=16, dropout=0.0,
        )
        model = AnomalyBertModel(config)
        normalizer = MinMaxNormalizer()
        normalizer.fit(np.array([0.0, 10.0]))
        model_path = tmp_path / "model.pt"
        save_checkpoint(model, normalizer, model_path)

        # Create input
        df = generate_scenario(SyntheticScenario.POINT_ANOMALY, n_samples=50)
        input_path = tmp_path / "input.csv"
        df.to_csv(input_path, index=False)

        result = runner.invoke(cli, [
            "detect", "--model", str(model_path),
            "--input", str(input_path), "--top-n", "5",
        ])
        assert result.exit_code == 0
        assert "Rank" in result.output

    def test_detect_json(self, runner, tmp_path):
        config = ModelConfig(
            window_size=16, d_model=32, n_layers=1, n_heads=4,
            d_ff=64, head_hidden_dim=16, dropout=0.0,
        )
        model = AnomalyBertModel(config)
        normalizer = MinMaxNormalizer()
        normalizer.fit(np.array([0.0, 10.0]))
        model_path = tmp_path / "model.pt"
        save_checkpoint(model, normalizer, model_path)

        df = generate_scenario(SyntheticScenario.POINT_ANOMALY, n_samples=50)
        input_path = tmp_path / "input.csv"
        df.to_csv(input_path, index=False)

        result = runner.invoke(cli, [
            "detect", "--model", str(model_path),
            "--input", str(input_path), "--top-n", "3", "--format", "json",
        ])
        assert result.exit_code == 0
        import json
        data = json.loads(result.output)
        assert len(data) == 3

    def test_detect_csv_format(self, runner, tmp_path):
        config = ModelConfig(
            window_size=16, d_model=32, n_layers=1, n_heads=4,
            d_ff=64, head_hidden_dim=16, dropout=0.0,
        )
        model = AnomalyBertModel(config)
        normalizer = MinMaxNormalizer()
        normalizer.fit(np.array([0.0, 10.0]))
        model_path = tmp_path / "model.pt"
        save_checkpoint(model, normalizer, model_path)

        df = generate_scenario(SyntheticScenario.POINT_ANOMALY, n_samples=50)
        input_path = tmp_path / "input.csv"
        df.to_csv(input_path, index=False)

        result = runner.invoke(cli, [
            "detect", "--model", str(model_path),
            "--input", str(input_path), "--top-n", "3", "--format", "csv",
        ])
        assert result.exit_code == 0
        assert "timestamp,value,score" in result.output
