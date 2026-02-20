import json

import click


@click.command()
@click.option("--model", required=True, help="Path to model checkpoint.")
@click.option("--input", "input_path", required=True, help="Input CSV with timestamp,value columns.")
@click.option("--top-n", default=10, type=int, help="Number of top anomalies to return.")
@click.option("--output", default=None, help="Output file path (prints to stdout if omitted).")
@click.option(
    "--format",
    "output_format",
    default="table",
    type=click.Choice(["table", "csv", "json"]),
    help="Output format.",
)
def detect(model, input_path, top_n, output, output_format):
    """Detect anomalies in timeseries data."""
    if model.endswith(".onnx"):
        from ..inference.onnx_detector import OnnxAnomalyDetector

        detector = OnnxAnomalyDetector(model)
    else:
        from ..inference.detector import AnomalyDetector

        detector = AnomalyDetector(model)
    results = detector.detect_from_csv(input_path, top_n=top_n)

    if output_format == "json":
        text = json.dumps(results, indent=2, default=str)
    elif output_format == "csv":
        lines = ["timestamp,value,score"]
        for r in results:
            lines.append(f"{r['timestamp']},{r['value']},{r['score']:.6f}")
        text = "\n".join(lines)
    else:
        # Table format
        lines = [f"{'Rank':<6}{'Timestamp':<20}{'Value':<15}{'Score':<10}"]
        lines.append("-" * 51)
        for i, r in enumerate(results, 1):
            lines.append(f"{i:<6}{str(r['timestamp']):<20}{r['value']:<15.4f}{r['score']:<10.6f}")
        text = "\n".join(lines)

    if output:
        from pathlib import Path

        Path(output).write_text(text)
        click.echo(f"Results written to {output}")
    else:
        click.echo(text)
