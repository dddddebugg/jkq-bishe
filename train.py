import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from timeseries_transformer import ModelConfig, SlidingWindowDataset, TimeSeriesTransformer


def build_sine_series(length: int, noise: float = 0.1) -> torch.Tensor:
    t = torch.linspace(0, 30, steps=length)
    series = torch.sin(t) + 0.5 * torch.sin(3 * t)
    series += noise * torch.randn_like(series)
    return series.unsqueeze(-1)


def train(epochs: int, batch_size: int, lr: float) -> None:
    cfg = ModelConfig()
    series = build_sine_series(2000)

    dataset = SlidingWindowDataset(series, cfg.input_window, cfg.horizon)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TimeSeriesTransformer(
        input_dim=cfg.input_dim,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        output_dim=cfg.output_dim,
        dropout=cfg.dropout,
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"epoch={epoch:02d} loss={total_loss / len(loader):.6f}")

    model.eval()
    with torch.no_grad():
        x, y = dataset[-1]
        pred = model(x.unsqueeze(0)).squeeze(0)
    print(f"last target: {y.tolist()}")
    print(f"prediction : {pred.tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transformer for time series forecast")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
