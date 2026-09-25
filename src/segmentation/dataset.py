import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from src.datasets.pcam import MEAN, STD


class SegmentationDataset(Dataset):
    """Image/mask dataset described by a CSV manifest.

    Expected columns:
        split,image_path,mask_path,slide_id,x,y,tumor_fraction

    Image and mask transformations are kept synchronized.
    """

    def __init__(
        self,
        manifest_path,
        split="train",
        image_size=256,
        augment=False,
    ):
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.image_size = int(image_size)
        self.augment = augment

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Segmentation manifest not found: {self.manifest_path}"
            )

        self.records = []
        with self.manifest_path.open(
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

            fields = set(reader.fieldnames or [])
            missing = required - fields
            if missing:
                raise ValueError(
                    f"Manifest missing columns: {sorted(missing)}"
                )

            for row in reader:
                if row["split"] == split:
                    self.records.append(row)

        if not self.records:
            raise ValueError(
                f"No records found for split='{split}' "
                f"in {self.manifest_path}"
            )

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_size, self.image_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

        self.mask_resize = transforms.Resize(
            (self.image_size, self.image_size),
            interpolation=transforms.InterpolationMode.NEAREST,
        )

    def __len__(self):
        return len(self.records)

    def _augment_pair(self, image, mask):
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # 0/90/180/270 degree rotation keeps dimensions exact
        # and requires no interpolation for the mask.
        quarter_turns = random.randint(0, 3)
        if quarter_turns:
            angle = quarter_turns * 90
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        return image, mask

    def __getitem__(self, index):
        row = self.records[index]

        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image patch not found: {image_path}"
            )
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask patch not found: {mask_path}"
            )

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            image, mask = self._augment_pair(image, mask)

        image = self.image_transform(image)
        mask = self.mask_resize(mask)
        mask = transforms.PILToTensor()(mask).float() / 255.0
        mask = (mask > 0.5).float()

        return {
            "image": image,
            "mask": mask,
            "slide_id": row["slide_id"],
            "x": int(row["x"]),
            "y": int(row["y"]),
            "tumor_fraction": float(row["tumor_fraction"]),
            "image_path": str(image_path),
        }


def split_group_counts(dataset):
    """Summarize number of patches per source slide."""
    counts = {}
    for row in dataset.records:
        slide_id = row["slide_id"]
        counts[slide_id] = counts.get(slide_id, 0) + 1
    return counts
