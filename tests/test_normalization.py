import numpy as np
import pytest

from anomalybert.data.normalization import MinMaxNormalizer, ZScoreNormalizer, create_normalizer


class TestMinMaxNormalizer:
    def test_fit_transform(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        norm = MinMaxNormalizer().fit(values)
        result = norm.transform(values)
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)

    def test_inverse_transform(self):
        values = np.array([10.0, 20.0, 30.0])
        norm = MinMaxNormalizer().fit(values)
        transformed = norm.transform(values)
        recovered = norm.inverse_transform(transformed)
        np.testing.assert_allclose(recovered, values, atol=1e-5)

    def test_constant_series(self):
        values = np.array([5.0, 5.0, 5.0])
        norm = MinMaxNormalizer().fit(values)
        result = norm.transform(values)
        np.testing.assert_array_equal(result, np.zeros(3, dtype=np.float32))

    def test_not_fitted_raises(self):
        norm = MinMaxNormalizer()
        with pytest.raises(RuntimeError):
            norm.transform(np.array([1.0]))

    def test_save_load(self, tmp_path):
        values = np.array([0.0, 10.0, 5.0])
        norm = MinMaxNormalizer().fit(values)
        path = tmp_path / "norm.json"
        norm.save(path)
        loaded = MinMaxNormalizer.load(path)
        assert loaded.min_val == norm.min_val
        assert loaded.max_val == norm.max_val


class TestZScoreNormalizer:
    def test_fit_transform(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        norm = ZScoreNormalizer().fit(values)
        result = norm.transform(values)
        assert np.abs(np.mean(result)) < 1e-5
        assert np.abs(np.std(result) - 1.0) < 1e-5

    def test_inverse_transform(self):
        values = np.array([10.0, 20.0, 30.0])
        norm = ZScoreNormalizer().fit(values)
        transformed = norm.transform(values)
        recovered = norm.inverse_transform(transformed)
        np.testing.assert_allclose(recovered, values, atol=1e-5)

    def test_constant_series(self):
        values = np.array([5.0, 5.0, 5.0])
        norm = ZScoreNormalizer().fit(values)
        result = norm.transform(values)
        np.testing.assert_array_equal(result, np.zeros(3, dtype=np.float32))


class TestCreateNormalizer:
    def test_minmax(self):
        assert isinstance(create_normalizer("minmax"), MinMaxNormalizer)

    def test_zscore(self):
        assert isinstance(create_normalizer("zscore"), ZScoreNormalizer)

    def test_none(self):
        assert create_normalizer("none") is None

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            create_normalizer("unknown")
