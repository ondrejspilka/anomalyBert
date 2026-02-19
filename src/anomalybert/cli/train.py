from pathlib import Path

import click


@click.command()
@click.option("--data", required=True, help="Training CSV file or directory of CSVs.")
@click.option("--output", default="models/model.pt", help="Model checkpoint output path.")
@click.option("--epochs", default=50, type=int, help="Number of training epochs.")
@click.option("--batch-size", default=32, type=int, help="Batch size.")
@click.option("--window-size", default=64, type=int, help="Sliding window size.")
@click.option("--d-model", default=128, type=int, help="Embedding dimension.")
@click.option("--n-layers", default=3, type=int, help="Number of transformer layers.")
@click.option("--n-heads", default=4, type=int, help="Number of attention heads.")
@click.option("--d-ff", default=256, type=int, help="Feedforward hidden dimension.")
@click.option("--lr", default=1e-3, type=float, help="Learning rate.")
@click.option("--normalization", default="minmax", help="Normalization: minmax, zscore, none.")
@click.option("--finetune", default=None, help="Path to pre-trained model for finetuning.")
@click.option("--val-split", default=0.2, type=float, help="Validation split ratio.")
def train(data, output, epochs, batch_size, window_size, d_model, n_layers, n_heads, d_ff, lr, normalization, finetune, val_split):
    """Train or finetune an AnomalyBERT model."""
    from ..model.config import ModelConfig
    from ..model.anomalybert import AnomalyBertModel
    from ..model.heads import FinetuneHead
    from ..data.normalization import create_normalizer
    from ..data.tokenizer import TimeseriesTokenizer
    from ..data.dataset import TimeseriesDataset
    from ..training.trainer import Trainer
    from ..training.checkpoint import load_checkpoint

    data_path = Path(data)
    tokenizer = TimeseriesTokenizer(window_size=window_size, stride=1)
    normalizer = create_normalizer(normalization)

    if finetune:
        click.echo(f"Loading pre-trained model from {finetune}")
        model, loaded_normalizer = load_checkpoint(finetune)
        if loaded_normalizer is not None:
            normalizer = loaded_normalizer
        model.freeze_encoder()
        model.set_head(FinetuneHead(model.config.d_model, model.config.head_hidden_dim))
        click.echo("Encoder frozen. Training finetune head only.")
    else:
        config = ModelConfig(
            window_size=window_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            head_hidden_dim=d_ff // 4,
            normalization=normalization,
        )
        model = AnomalyBertModel(config)
        click.echo(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")

    # Load dataset
    if data_path.is_dir():
        dataset = TimeseriesDataset.from_directory(
            data_path, tokenizer, normalizer, is_training=True
        )
    else:
        dataset = TimeseriesDataset.from_csv(
            data_path, tokenizer, normalizer, fit_normalizer=True, is_training=True
        )

    click.echo(f"Dataset: {len(dataset)} windows")

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        normalizer=normalizer,
        lr=lr,
        batch_size=batch_size,
        val_split=val_split,
    )
    losses = trainer.train(epochs=epochs, checkpoint_path=output)
    click.echo(f"Training complete. Final loss: {losses[-1]:.4f}")
