from dataclasses import dataclass
from pathlib import Path
from typing import List
import csv

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.config import CHECKPOINT_DIR, get_device
from src.datasets.pcam import MEAN, STD
from src.models.resnet18 import build_resnet18


@dataclass(frozen=True)
class TileRecord:
    patch_id: str
    x: int
    y: int
    width: int
    height: int
    tissue_fraction: float
    path: Path


class WSIPatchDataset(Dataset):
    """Lazy-loading dataset for Phase-2 extracted WSI patches."""

    def __init__(self, records: List[TileRecord], image_size: int = 224):
        self.records = records
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        if not record.path.exists():
            raise FileNotFoundError(f"Patch file not found: {record.path}")
        image = Image.open(record.path).convert("RGB")
        return self.transform(image), index


def read_tile_records(metadata_csv: str | Path) -> List[TileRecord]:
    path = Path(metadata_csv)
    if not path.exists():
        raise FileNotFoundError(f"Tile metadata not found: {path}")

    required = {"patch_id", "x", "y", "width", "height", "tissue_fraction", "path"}
    records = []

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise ValueError(f"tiles.csv is missing columns: {sorted(missing)}")

        for row in reader:
            records.append(TileRecord(
                patch_id=row["patch_id"],
                x=int(row["x"]),
                y=int(row["y"]),
                width=int(row["width"]),
                height=int(row["height"]),
                tissue_fraction=float(row["tissue_fraction"]),
                path=Path(row["path"]),
            ))

    if not records:
        raise ValueError("tiles.csv contains no patch records.")
    return records


def load_resnet_checkpoint(checkpoint=None, device=None):
    if device is None:
        device = get_device()
    path = Path(checkpoint) if checkpoint else CHECKPOINT_DIR / "best_resnet18.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"ResNet18 checkpoint not found: {path}. "
            "Train it first with train_resnet.py."
        )

    saved = torch.load(path, map_location=device, weights_only=False)
    model = build_resnet18(pretrained=False, strategy="full").to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model, saved, path


def predict_tiles(records, model, device, batch_size=32, num_workers=0):
    dataset = WSIPatchDataset(records, image_size=224)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    probabilities = np.zeros(len(records), dtype=np.float32)

    with torch.no_grad():
        for images, indices in tqdm(loader, desc="WSI inference"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            probabilities[indices.numpy()] = probs.cpu().numpy()

    return probabilities
