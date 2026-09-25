import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CHECKPOINT_DIR, DATA_DIR, get_device
from src.datasets.pcam import get_pcam_dataset
from src.models.cnn import TumorCNN
from src.models.resnet18 import build_resnet18


def collect_cnn(split, checkpoint, limit, batch_size, num_workers, device):
    dataset = get_pcam_dataset(
        split,
        download=False,
        image_size=96,
    )

    if limit is not None and limit > 0:
        from torch.utils.data import Subset

        dataset = Subset(
            dataset,
            range(min(limit, len(dataset))),
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = TumorCNN().to(device)

    checkpoint = torch.load(
        checkpoint,
        map_location=device,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    labels = []
    probabilities = []

    with torch.no_grad():
        for images, batch_labels in tqdm(
            loader,
            desc=f"cnn:{split}",
        ):
            images = images.to(
                device,
                non_blocking=True,
            )

            logits = model(images)
            probs = torch.sigmoid(logits)

            labels.append(
                batch_labels.numpy()
            )
            probabilities.append(
                probs.cpu().numpy()
            )

    return (
        np.concatenate(labels).astype(np.int64),
        np.concatenate(probabilities).astype(np.float32),
    )


def collect_resnet(split, checkpoint, limit, batch_size, num_workers, device):
    dataset = get_pcam_dataset(
        split,
        download=False,
        image_size=224,
    )

    if limit is not None and limit > 0:
        from torch.utils.data import Subset

        dataset = Subset(
            dataset,
            range(min(limit, len(dataset))),
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    saved = torch.load(
        checkpoint,
        map_location=device,
    )

    model = build_resnet18(
        pretrained=False,
        strategy="full",
    ).to(device)
    model.load_state_dict(
        saved["model_state_dict"]
    )
    model.eval()

    labels = []
    probabilities = []

    with torch.no_grad():
        for images, batch_labels in tqdm(
            loader,
            desc=f"resnet18:{split}",
        ):
            images = images.to(
                device,
                non_blocking=True,
            )

            logits = model(images)
            probs = torch.softmax(
                logits,
                dim=1,
            )[:, 1]

            labels.append(
                batch_labels.numpy()
            )
            probabilities.append(
                probs.cpu().numpy()
            )

    return (
        np.concatenate(labels).astype(np.int64),
        np.concatenate(probabilities).astype(np.float32),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collect model probabilities once so evaluation "
            "can analyze thresholds, ROC/PR curves and errors."
        )
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "resnet18"],
        default="resnet18",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    if args.checkpoint is None:
        args.checkpoint = str(
            CHECKPOINT_DIR
            / (
                "best_cnn.pt"
                if args.model == "cnn"
                else "best_resnet18.pt"
            )
        )

    checkpoint = Path(
        args.checkpoint
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    if args.model == "cnn":
        labels, probs = collect_cnn(
            args.split,
            checkpoint,
            args.limit,
            args.batch_size,
            args.num_workers,
            device,
        )
    else:
        labels, probs = collect_resnet(
            args.split,
            checkpoint,
            args.limit,
            args.batch_size,
            args.num_workers,
            device,
        )

    if args.output is None:
        filename = (
            f"{args.model}_{args.split}_predictions.npz"
        )
        output = (
            DATA_DIR
            / "experiments"
            / filename
        )
    else:
        output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        output,
        labels=labels,
        probabilities=probs,
    )

    print(
        f"Saved {len(labels):,} predictions to {output}"
    )


if __name__ == "__main__":
    main()
