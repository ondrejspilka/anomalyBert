import numpy as np
import torch
import pytest

from anomalybert.model.config import ModelConfig
from anomalybert.model.anomalybert import AnomalyBertModel
from anomalybert.data.normalization import MinMaxNormalizer
from anomalybert.data.tokenizer import TimeseriesTokenizer
from anomalybert.data.dataset import TimeseriesDataset
from anomalybert.data.synthetic import SyntheticScenario, generate_scenario
from anomalybert.training.trainer import Trainer
from anomalybert.training.checkpoint import save_checkpoint, load_checkpoint


@pytest.fixture
def training_setup():
    config = ModelConfig(
        window_size=16, d_model=32, n_layers=1, n_heads=4,
        d_ff=64, head_hidden_dim=16, dropout=0.0,
    )
    model = AnomalyBertModel(config)
    normalizer = MinMaxNormalizer()
    tokenizer = TimeseriesTokenizer(window_size=16, stride=8)

    df = generate_scenario(SyntheticScenario.POINT_ANOMALY, n_samples=100, seed=42)
    values = df["value"].values
    labels = df["probability"].values
    normalizer.fit(values)
    dataset = TimeseriesDataset(values, labels, tokenizer, normalizer)

    return model, dataset, normalizer, config


class TestTrainer:
    def test_loss_decreases(self, training_setup):
        model, dataset, normalizer, config = training_setup
        trainer = Trainer(model, dataset, normalizer, lr=1e-2, batch_size=4, val_split=0.0)
        losses = trainer.train(epochs=5, checkpoint_path="models/test_model.pt")
        # Loss should generally decrease
        assert losses[-1] <= losses[0] + 0.1  # Allow small fluctuations

    def test_checkpoint_save_load(self, training_setup, tmp_path):
        model, dataset, normalizer, config = training_setup
        path = tmp_path / "model.pt"
        save_checkpoint(model, normalizer, path)
        loaded_model, loaded_norm = load_checkpoint(path)
        assert loaded_model.config.d_model == config.d_model
        assert loaded_model.config.n_layers == config.n_layers
        assert loaded_norm.min_val == normalizer.min_val
        assert loaded_norm.max_val == normalizer.max_val

    def test_finetune_freezes_encoder(self, training_setup):
        model, dataset, normalizer, config = training_setup
        model.freeze_encoder()
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert all("head" in n for n in trainable)


class TestCheckpoint:
    def test_roundtrip(self, training_setup, tmp_path):
        model, _, normalizer, _ = training_setup
        model.eval()
        x = torch.randn(1, 16)
        with torch.no_grad():
            out1 = model(x)

        path = tmp_path / "ckpt.pt"
        save_checkpoint(model, normalizer, path)
        loaded_model, _ = load_checkpoint(path)
        loaded_model.eval()
        with torch.no_grad():
            out2 = loaded_model(x)

        torch.testing.assert_close(out1, out2)
