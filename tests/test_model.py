import torch
import pytest

from anomalybert.model.config import ModelConfig
from anomalybert.model.anomalybert import AnomalyBertModel
from anomalybert.model.heads import FinetuneHead


class TestAnomalyBertModel:
    def test_forward_shape(self, small_config):
        model = AnomalyBertModel(small_config)
        x = torch.randn(4, small_config.window_size)
        scores = model(x)
        assert scores.shape == (4, small_config.window_size)

    def test_output_range(self, small_config):
        model = AnomalyBertModel(small_config)
        model.eval()
        x = torch.randn(2, small_config.window_size)
        with torch.no_grad():
            scores = model(x)
        assert (scores >= 0).all()
        assert (scores <= 1).all()

    def test_different_configs(self):
        for n_layers in [1, 2, 4]:
            for d_model in [16, 32]:
                config = ModelConfig(
                    window_size=8, d_model=d_model, n_layers=n_layers,
                    n_heads=4, d_ff=d_model * 2, head_hidden_dim=d_model // 2, dropout=0.0,
                )
                model = AnomalyBertModel(config)
                x = torch.randn(2, 8)
                scores = model(x)
                assert scores.shape == (2, 8)

    def test_freeze_encoder(self, small_config):
        model = AnomalyBertModel(small_config)
        model.freeze_encoder()
        for name, param in model.named_parameters():
            if "head" in name:
                assert param.requires_grad
            else:
                assert not param.requires_grad

    def test_unfreeze_encoder(self, small_config):
        model = AnomalyBertModel(small_config)
        model.freeze_encoder()
        model.unfreeze_encoder()
        for param in model.parameters():
            assert param.requires_grad

    def test_set_head(self, small_config):
        model = AnomalyBertModel(small_config)
        new_head = FinetuneHead(small_config.d_model, small_config.head_hidden_dim)
        model.set_head(new_head)
        x = torch.randn(2, small_config.window_size)
        scores = model(x)
        assert scores.shape == (2, small_config.window_size)

    def test_serialization(self, small_config, tmp_path):
        model = AnomalyBertModel(small_config)
        path = tmp_path / "model.pt"
        torch.save(model.state_dict(), path)
        model2 = AnomalyBertModel(small_config)
        model2.load_state_dict(torch.load(path, weights_only=True))
        x = torch.randn(1, small_config.window_size)
        model.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)
        torch.testing.assert_close(out1, out2)
