from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd


class SyntheticScenario(Enum):
    POINT_ANOMALY = "point_anomaly"
    CONTEXTUAL_ANOMALY = "contextual_anomaly"
    COLLECTIVE_ANOMALY = "collective_anomaly"
    SEASONAL_WITH_ANOMALY = "seasonal_with_anomaly"
    TREND_SHIFT = "trend_shift"
    NOISE_BURST = "noise_burst"
    FREQUENCY_CHANGE = "frequency_change"
    FLAT_SIGNAL = "flat_signal"


def _make_timestamps(n: int, start: int = 1000000000, interval: int = 60) -> np.ndarray:
    return np.arange(start, start + n * interval, interval, dtype=np.int64)


def generate_scenario(
    scenario: SyntheticScenario,
    n_samples: int = 100,
    anomaly_ratio: float = 0.05,
    seed: int | None = None,
) -> pd.DataFrame:
    if seed is None:
        rng = np.random.RandomState()
    else:
        rng = np.random.RandomState(seed)
    n_anomalies = max(1, int(n_samples * anomaly_ratio))
    ts_start = int(1e9) + rng.randint(0, int(1e7))
    ts_interval = rng.choice([1, 5, 10, 30, 60, 300, 3600])
    timestamps = _make_timestamps(n_samples, start=ts_start, interval=ts_interval)
    values = np.zeros(n_samples, dtype=np.float64)
    probability = np.zeros(n_samples, dtype=np.float64)
    anomaly_tag = np.zeros(n_samples, dtype=np.int32)

    if scenario == SyntheticScenario.POINT_ANOMALY:
        base_mean = rng.uniform(-10, 10)
        base_std = rng.uniform(0.5, 3.0)
        values = rng.normal(base_mean, base_std, n_samples)
        anomaly_idx = rng.choice(n_samples, n_anomalies, replace=False)
        for idx in anomaly_idx:
            spike_mag = rng.uniform(3, 8) * base_std
            values[idx] += rng.choice([-1, 1]) * spike_mag
            probability[idx] = rng.uniform(0.8, 1.0)
            anomaly_tag[idx] = 1

    elif scenario == SyntheticScenario.CONTEXTUAL_ANOMALY:
        n_cycles = rng.uniform(2, 8)
        amplitude = rng.uniform(0.5, 5.0)
        noise_std = rng.uniform(0.02, 0.2) * amplitude
        t = np.linspace(0, n_cycles * 2 * np.pi, n_samples)
        values = amplitude * np.sin(t) + rng.normal(0, noise_std, n_samples)
        anomaly_idx = rng.choice(n_samples, n_anomalies, replace=False)
        for idx in anomaly_idx:
            values[idx] = -values[idx] + rng.normal(0, noise_std)
            probability[idx] = rng.uniform(0.7, 1.0)
            anomaly_tag[idx] = 1

    elif scenario == SyntheticScenario.COLLECTIVE_ANOMALY:
        base_mean = rng.uniform(-5, 5)
        base_std = rng.uniform(0.5, 2.0)
        values = rng.normal(base_mean, base_std, n_samples)
        seg_len = max(2, rng.randint(n_anomalies, max(n_anomalies + 1, n_samples // 5)))
        start = rng.randint(0, max(1, n_samples - seg_len))
        shift = rng.uniform(2, 6) * base_std
        values[start : start + seg_len] += rng.choice([-1, 1]) * shift
        prob = rng.uniform(0.75, 1.0)
        probability[start : start + seg_len] = prob
        anomaly_tag[start : start + seg_len] = 1

    elif scenario == SyntheticScenario.SEASONAL_WITH_ANOMALY:
        n_cycles = rng.uniform(2, 10)
        amp1 = rng.uniform(0.5, 5.0)
        amp2 = rng.uniform(0.1, 0.8) * amp1
        harmonic = rng.randint(2, 6)
        noise_std = rng.uniform(0.02, 0.15) * amp1
        t = np.linspace(0, n_cycles * 2 * np.pi, n_samples)
        values = amp1 * np.sin(t) + amp2 * np.sin(harmonic * t) + rng.normal(0, noise_std, n_samples)
        anomaly_idx = rng.choice(n_samples, n_anomalies, replace=False)
        for idx in anomaly_idx:
            spike_mag = rng.uniform(2, 5) * amp1
            values[idx] += rng.choice([-1, 1]) * spike_mag
            probability[idx] = rng.uniform(0.8, 1.0)
            anomaly_tag[idx] = 1

    elif scenario == SyntheticScenario.TREND_SHIFT:
        mid = rng.randint(n_samples // 4, 3 * n_samples // 4)
        mean1 = rng.uniform(-5, 5)
        std1 = rng.uniform(0.2, 1.5)
        shift = rng.uniform(2, 8) * std1
        mean2 = mean1 + rng.choice([-1, 1]) * shift
        std2 = rng.uniform(0.2, 1.5)
        values[:mid] = rng.normal(mean1, std1, mid)
        values[mid:] = rng.normal(mean2, std2, n_samples - mid)
        trans_width = max(1, rng.randint(1, max(2, n_anomalies)))
        trans_start = max(0, mid - trans_width // 2)
        trans_end = min(n_samples, mid + trans_width // 2 + 1)
        probability[trans_start:trans_end] = rng.uniform(0.6, 0.9)
        anomaly_tag[trans_start:trans_end] = 1

    elif scenario == SyntheticScenario.NOISE_BURST:
        base_std = rng.uniform(0.1, 1.0)
        base_mean = rng.uniform(-5, 5)
        values = rng.normal(base_mean, base_std, n_samples)
        burst_len = max(2, rng.randint(n_anomalies, max(n_anomalies + 1, n_samples // 5)))
        start = rng.randint(0, max(1, n_samples - burst_len))
        burst_std = rng.uniform(3, 10) * base_std
        values[start : start + burst_len] = rng.normal(base_mean, burst_std, burst_len)
        prob = rng.uniform(0.7, 1.0)
        probability[start : start + burst_len] = prob
        anomaly_tag[start : start + burst_len] = 1

    elif scenario == SyntheticScenario.FREQUENCY_CHANGE:
        mid = rng.randint(n_samples // 4, 3 * n_samples // 4)
        amplitude = rng.uniform(0.5, 5.0)
        noise_std = rng.uniform(0.02, 0.15) * amplitude
        freq1 = rng.uniform(1, 4)
        freq2 = rng.uniform(4, 12)
        t1 = np.linspace(0, freq1 * 2 * np.pi, mid)
        t2 = np.linspace(0, freq2 * 2 * np.pi, n_samples - mid)
        values[:mid] = amplitude * np.sin(t1) + rng.normal(0, noise_std, mid)
        values[mid:] = amplitude * np.sin(t2) + rng.normal(0, noise_std, n_samples - mid)
        trans_width = max(1, rng.randint(1, max(2, n_anomalies)))
        trans_start = max(0, mid - trans_width // 2)
        trans_end = min(n_samples, mid + trans_width // 2 + 1)
        probability[trans_start:trans_end] = rng.uniform(0.6, 0.9)
        anomaly_tag[trans_start:trans_end] = 1

    elif scenario == SyntheticScenario.FLAT_SIGNAL:
        base_val = rng.uniform(-10, 10)
        base_noise = rng.uniform(0.001, 0.05) * abs(base_val + 1)
        values = np.full(n_samples, base_val) + rng.normal(0, base_noise, n_samples)
        anomaly_idx = rng.choice(n_samples, n_anomalies, replace=False)
        for idx in anomaly_idx:
            deviation = rng.uniform(0.3, 2.0) * abs(base_val + 1)
            values[idx] += rng.choice([-1, 1]) * deviation
            probability[idx] = rng.uniform(0.7, 1.0)
            anomaly_tag[idx] = 1

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "value": values,
            "probability": probability,
            "anomaly_tag": anomaly_tag,
        }
    )


def generate_all_scenarios(
    output_dir: str | Path,
    sizes: list[int] | None = None,
    anomaly_ratio: float = 0.05,
    seed: int | None = None,
) -> list[Path]:
    if sizes is None:
        sizes = [30, 50, 100, 200]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for i, scenario in enumerate(SyntheticScenario):
        for j, size in enumerate(sizes):
            s = (seed + i * len(sizes) + j) if seed is not None else None
            df = generate_scenario(scenario, n_samples=size, anomaly_ratio=anomaly_ratio, seed=s)
            filename = f"{scenario.value}_{size}.csv"
            path = output_dir / filename
            df.to_csv(path, index=False)
            generated.append(path)
    return generated
