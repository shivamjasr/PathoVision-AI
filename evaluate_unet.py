import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    CHECKPOINT_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    get_device,
    make_output_dirs,
)
from src.models.unet import UNet
from src.segmentation.dataset import SegmentationDataset
from src.segmentation.losses import segmentation_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate U-Net segmentation model on validation patches."
    )
    parser.add_argument(
        "--manifest",
        default="data/segmentation_demo/manifest.csv",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            CHECKPOINT_DIR / "best_unet.pt"
        ),
    )
    parser.add_argument(
        "--split",
        default="val",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    args = parser.parse_args()

    make_output_dirs()
    device = get_device()

    checkpoint_path = (
        args.checkpoint
    )

    if not (
        __import__("pathlib").Path(
            checkpoint_path
        ).exists()
    ):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    image_size = checkpoint.get(
        "image_size",
        args.image_size,
    )
    base_channels = checkpoint.get(
        "base_channels",
        args.base_channels,
    )

    dataset = SegmentationDataset(
        args.manifest,
        split=args.split,
        image_size=image_size,
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = UNet(
        base_channels=base_channels
    ).to(device)
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    accum = {
        "dice": [],
        "iou": [],
        "precision": [],
        "recall": [],
    }

    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc="Evaluating",
        ):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)

            metrics = segmentation_metrics(
                logits,
                masks,
                threshold=args.threshold,
            )

            for key in accum:
                accum[key].append(metrics[key])

    metrics = {
        key: float(np.mean(values))
        for key, values in accum.items()
    }

    metrics["threshold"] = args.threshold
    metrics["num_patches"] = len(dataset)

    output = (
        METRICS_DIR
        / "unet_val_metrics.json"
    )
    output.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nU-Net validation metrics")
    for key, value in metrics.items():
        print(f"{key:>12}: {value}")

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
