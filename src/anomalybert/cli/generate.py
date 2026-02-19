import click

from ..data.synthetic import SyntheticScenario, generate_all_scenarios, generate_scenario


@click.command()
@click.option("--output-dir", default="data/synthetic", help="Output directory for CSV files.")
@click.option(
    "--scenario",
    default="all",
    help="Scenario name or 'all'. Options: " + ", ".join(s.value for s in SyntheticScenario),
)
@click.option("--n-samples", default=100, type=int, help="Number of data points per file.")
@click.option("--anomaly-ratio", default=0.05, type=float, help="Fraction of anomalous points.")
@click.option("--seed", default=42, type=int, help="Random seed.")
def generate(output_dir, scenario, n_samples, anomaly_ratio, seed):
    """Generate synthetic timeseries datasets."""
    if scenario == "all":
        files = generate_all_scenarios(
            output_dir, sizes=[n_samples], anomaly_ratio=anomaly_ratio, seed=seed
        )
        click.echo(f"Generated {len(files)} files in {output_dir}/")
        for f in files:
            click.echo(f"  {f}")
    else:
        try:
            sc = SyntheticScenario(scenario)
        except ValueError:
            click.echo(f"Unknown scenario: {scenario}")
            click.echo("Available: " + ", ".join(s.value for s in SyntheticScenario))
            raise SystemExit(1)

        from pathlib import Path

        df = generate_scenario(sc, n_samples=n_samples, anomaly_ratio=anomaly_ratio, seed=seed)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        filename = out_path / f"{scenario}_{n_samples}.csv"
        df.to_csv(filename, index=False)
        click.echo(f"Generated {filename}")
