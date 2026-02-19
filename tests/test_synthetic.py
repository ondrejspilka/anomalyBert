import pytest

from anomalybert.data.synthetic import SyntheticScenario, generate_scenario, generate_all_scenarios


class TestGenerateScenario:
    @pytest.mark.parametrize("scenario", list(SyntheticScenario))
    def test_columns(self, scenario):
        df = generate_scenario(scenario, n_samples=50)
        assert list(df.columns) == ["timestamp", "value", "probability", "anomaly_tag"]

    @pytest.mark.parametrize("scenario", list(SyntheticScenario))
    def test_length(self, scenario):
        df = generate_scenario(scenario, n_samples=50)
        assert len(df) == 50

    @pytest.mark.parametrize("scenario", list(SyntheticScenario))
    def test_anomaly_tag_binary(self, scenario):
        df = generate_scenario(scenario, n_samples=50)
        assert set(df["anomaly_tag"].unique()).issubset({0, 1})

    @pytest.mark.parametrize("scenario", list(SyntheticScenario))
    def test_probability_range(self, scenario):
        df = generate_scenario(scenario, n_samples=50)
        assert (df["probability"] >= 0).all()
        assert (df["probability"] <= 1).all()

    @pytest.mark.parametrize("scenario", list(SyntheticScenario))
    def test_has_anomalies(self, scenario):
        df = generate_scenario(scenario, n_samples=100, anomaly_ratio=0.1)
        assert df["anomaly_tag"].sum() > 0

    def test_deterministic_with_seed(self):
        df1 = generate_scenario(SyntheticScenario.POINT_ANOMALY, seed=42)
        df2 = generate_scenario(SyntheticScenario.POINT_ANOMALY, seed=42)
        assert df1.equals(df2)

    @pytest.mark.parametrize("size", [10, 30, 100])
    def test_various_sizes(self, size):
        df = generate_scenario(SyntheticScenario.POINT_ANOMALY, n_samples=size)
        assert len(df) == size


class TestGenerateAllScenarios:
    def test_generates_files(self, tmp_path):
        files = generate_all_scenarios(tmp_path, sizes=[30])
        assert len(files) == len(SyntheticScenario)
        for f in files:
            assert f.exists()
            assert f.suffix == ".csv"

    def test_multiple_sizes(self, tmp_path):
        files = generate_all_scenarios(tmp_path, sizes=[30, 50])
        assert len(files) == len(SyntheticScenario) * 2
