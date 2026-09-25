from pathlib import Path
from typing import List

import csv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.config import CHECKPOINT_DIR, get_device
from src.datasets.pcam import MEAN, STD
from src.models.unet import UNet
from src.segmentation.dataset import SegmentationDataset
from src.segmentation.losses import segmentation_metrics


class WSISegmentationPatchDataset(Dataset):
    """Lazy WSI patch dataset for U-Net inference."""

    def __init__(self, records, image_size=256):
        self.records = records
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]

        path = Path(row["image_path"])
        if not path.exists():
            raise FileNotFoundError(
                f"WSI segmentation patch not found: {path}"
            )

        image = Image.open(path).convert("RGB")

        return (
            self.transform(image),
            index,
        )


def read_manifest(
    manifest_path,
    split="val",
):
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    records = []

    with manifest_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "split",
            "image_path",
            "mask_path",
            "slide_id",
            "x",
            "y",
            "tumor_fraction",
        }

        if not required.issubset(
            set(reader.fieldnames or [])
        ):
            raise ValueError(
                "Segmentation manifest does not contain "
                f"required columns: {sorted(required)}"
            )

        for row in reader:
            if row["split"] == split:
                records.append(row)

    if not records:
        raise ValueError(
            f"No WSI patch records for split='{split}'"
        )

    return records


def load_unet(
    checkpoint_path=None,
    device=None,
):
    if device is None:
        device = get_device()

    if checkpoint_path is None:
        checkpoint_path = (
            CHECKPOINT_DIR
            / "best_unet.pt"
        )

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"U-Net checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model = UNet(
        in_channels=3,
        out_channels=1,
        base_channels=checkpoint.get(
            "base_channels",
            16,
        ),
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    return model, checkpoint


def predict_patch_masks(
    records,
    model,
    device,
    image_size=256,
    batch_size=4,
    threshold=0.5,
    num_workers=0,
):
    """Predict a binary mask for each patch.

    Returns uint8 masks with 0/255 values.
    """
    dataset = WSISegmentationPatchDataset(
        records,
        image_size=image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    results = [None] * len(records)

    with torch.no_grad():
        for images, indices in tqdm(
            loader,
            desc="U-Net WSI inference",
        ):
            images = images.to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            )

            masks = (
                probabilities >= threshold
            ).to(torch.uint8) * 255

            masks = masks.cpu().numpy()

            for local_idx, sample_index in enumerate(
                indices.numpy()
            ):
                results[int(sample_index)] = (
                    masks[local_idx, 0]
                )

    return results
