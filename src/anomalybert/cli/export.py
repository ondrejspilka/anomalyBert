import click


@click.command()
@click.option("--model", required=True, help="Path to PyTorch model checkpoint (.pt).")
@click.option("--output", required=True, help="Output path for ONNX model (.onnx).")
def export(model, output):
    """Export a trained model to ONNX format."""
    from ..inference.onnx_export import export_to_onnx

    path = export_to_onnx(model, output)
    click.echo(f"ONNX model exported to {path}")
    click.echo(f"Metadata saved to {path}.json")
