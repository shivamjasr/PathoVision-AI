import argparse

import torch

from src.config import BATCH_SIZE, get_device, make_output_dirs
from src.datasets.pcam import make_loaders
from src.models.cnn import TumorCNN


def main():
    parser = argparse.ArgumentParser(
        description="Verify DataLoader -> CNN forward pass without training."
    )
    parser.add_argument("--batch-size", type=int, default=min(BATCH_SIZE, 16))
    args = parser.parse_args()

    make_output_dirs()
    device = get_device()

    print(f"Device: {device}")

    train_loader, _, _ = make_loaders(
        download=False,
        limit_train=args.batch_size,
        limit_val=1,
        limit_test=1,
        batch_size=args.batch_size,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))

    print(f"Images shape: {tuple(images.shape)}")
    print(f"Labels shape: {tuple(labels.shape)}")
    print(f"Labels dtype: {labels.dtype}")

    if images.ndim != 4:
        raise RuntimeError("Images must be a 4D batch: [B, C, H, W].")

    if images.shape[1:] != (3, 96, 96):
        raise RuntimeError(
            f"Expected [B, 3, 96, 96], received {tuple(images.shape)}."
        )

    model = TumorCNN().to(device)
    images = images.to(device)

    with torch.no_grad():
        logits = model(images)

    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Logits dtype: {logits.dtype}")

    expected = (images.shape[0],)
    if tuple(logits.shape) != expected:
        raise RuntimeError(
            f"Expected output shape {expected}, received {tuple(logits.shape)}."
        )

    probabilities = torch.sigmoid(logits)
    print(
        f"Probability range: "
        f"{probabilities.min().item():.4f} → {probabilities.max().item():.4f}"
    )

    print("\nDataLoader -> CNN forward pass PASSED.")


if __name__ == "__main__":
    main()
