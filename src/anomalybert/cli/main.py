import click

from .generate import generate
from .train import train
from .detect import detect
from .export import export


@click.group()
@click.version_option(package_name="anomalybert")
def cli():
    """AnomalyBERT: Anomaly detection in timeseries data."""
    pass


cli.add_command(generate)
cli.add_command(train)
cli.add_command(detect)
cli.add_command(export)
