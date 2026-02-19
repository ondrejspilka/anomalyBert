import torch
import pytest

from anomalybert.model.embedding import ValueEmbedding, PositionalEncoding


class TestValueEmbedding:
    def test_output_shape(self):
        emb = ValueEmbedding(d_model=32)
        x = torch.randn(4, 16)  # batch=4, seq_len=16
        out = emb(x)
        assert out.shape == (4, 16, 32)

    def test_3d_input(self):
        emb = ValueEmbedding(d_model=32)
        x = torch.randn(4, 16, 1)
        out = emb(x)
        assert out.shape == (4, 16, 32)

    def test_gradient_flows(self):
        emb = ValueEmbedding(d_model=32)
        x = torch.randn(2, 8, requires_grad=True)
        out = emb(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestPositionalEncoding:
    def test_output_shape(self):
        pe = PositionalEncoding(d_model=32, dropout=0.0)
        x = torch.randn(4, 16, 32)
        out = pe(x)
        assert out.shape == (4, 16, 32)

    def test_deterministic(self):
        pe = PositionalEncoding(d_model=32, dropout=0.0)
        x = torch.zeros(1, 10, 32)
        out1 = pe(x)
        out2 = pe(x)
        torch.testing.assert_close(out1, out2)

    def test_adds_position_info(self):
        pe = PositionalEncoding(d_model=32, dropout=0.0)
        x = torch.zeros(1, 10, 32)
        out = pe(x)
        # Output should not be all zeros since PE adds sinusoidal values
        assert out.abs().sum() > 0
