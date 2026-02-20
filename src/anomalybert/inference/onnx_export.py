import json
from pathlib import Path

import torch

from ..training.checkpoint import load_checkpoint


def export_to_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Export a PyTorch AnomalyBERT checkpoint to ONNX format.

    Produces two files:
    - output_path (.onnx): the ONNX model
    - output_path.json: metadata (config + normalizer params)
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, normalizer = load_checkpoint(checkpoint_path)
    model.eval()

    # Dummy input matching expected shape: (batch=1, window_size)
    dummy_input = torch.randn(1, model.config.window_size)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        input_names=["input"],
        output_names=["scores"],
        dynamic_shapes={
            "x": {0: torch.export.Dim("batch_size")},
        },
    )

    # Save metadata alongside
    metadata = {
        "config": model.config.to_dict(),
        "normalizer": None,
    }
    if normalizer is not None:
        from ..data.normalization import MinMaxNormalizer, ZScoreNormalizer

        if isinstance(normalizer, MinMaxNormalizer):
            metadata["normalizer"] = {
                "type": "minmax",
                "min": normalizer.min_val,
                "max": normalizer.max_val,
            }
        elif isinstance(normalizer, ZScoreNormalizer):
            metadata["normalizer"] = {
                "type": "zscore",
                "mean": normalizer.mean,
                "std": normalizer.std,
            }

    meta_path = Path(str(output_path) + ".json")
    meta_path.write_text(json.dumps(metadata, indent=2))

    return output_path
