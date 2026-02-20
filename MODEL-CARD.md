---
license: mit
tags:
  - anomaly-detection
  - timeseries
  - transformer
  - onnx
library_name: onnxruntime
pipeline_tag: time-series-forecasting
---

# AnomalyBERT

Encoder-only transformer for anomaly detection in univariate timeseries data. Exported as ONNX for lightweight inference without PyTorch.

## Model Description

- **Architecture**: Encoder-only transformer (BERT-style) with custom multi-head self-attention, sinusoidal positional encoding, and an MLP anomaly scoring head
- **Input**: Univariate timeseries as `(timestamp, value)` pairs
- **Output**: Top-N anomalous points ranked by anomaly score in `[0, 1]`
- **Training**: Supervised with BCE loss on labeled synthetic data (8 anomaly scenarios)
- **Framework**: PyTorch (training), ONNX Runtime (inference)
- **CPU only**: No GPU required

## Architecture

```
Normalized values → Sliding Window (size=16, stride=1)
  → ValueEmbedding: Linear(1 → 64)
  → Sinusoidal Positional Encoding
  → Transformer Encoder (2 layers, 4 heads, d_ff=128)
  → AnomalyScoringHead: MLP → Sigmoid
  → Overlapping Window Aggregation → Top-N
```

## How to Use

### With ONNX Runtime (Python)

```python
import numpy as np
import json
import onnxruntime as ort

# Load model and metadata
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
meta = json.load(open("model.onnx.json"))

# Normalize input (using saved min/max from metadata)
norm = meta["normalizer"]
values = np.array([...], dtype=np.float32)
normalized = (values - norm["min"]) / (norm["max"] - norm["min"])

# Run inference on a window
window = normalized[:16].reshape(1, -1).astype(np.float32)
scores = session.run(None, {"input": window})[0]
```

### With AnomalyBERT Library

```bash
pip install -e ".[onnx]"
anomalybert detect --model model.onnx --input data.csv --top-n 10
```

```python
from anomalybert.inference.onnx_detector import OnnxAnomalyDetector

detector = OnnxAnomalyDetector("model.onnx")
results = detector.detect(timestamps, values, top_n=10)
```

## Files

| File | Description |
|------|-------------|
| `model.onnx` | ONNX model graph and weights |
| `model.onnx.json` | Model config and normalizer parameters |

## Training Data

Trained on 8 synthetic anomaly scenarios: point anomaly, contextual anomaly, collective anomaly, seasonal with anomaly, trend shift, noise burst, frequency change, and flat signal. Each scenario generates randomized timeseries with labeled anomaly positions and probabilities.

## Limitations

- Trained on synthetic data only; may require finetuning for real-world domains
- Univariate timeseries only (single value column)
- Designed for small-to-medium series (10s to 100s of samples)

## Source

GitHub: [anomalyBert](https://github.com/onsp327/anomalyBert)
